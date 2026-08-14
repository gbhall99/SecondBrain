"""Always-on supervisor (launchd entrypoint).

Runs three cooperating loops, each with its own SQLite connection:
  1. capture  — the rolling recorder (room audio -> FLAC chunks -> queue)
  2. worker   — drains transcription jobs (VAD -> transcribe -> store -> index)
  3. maintenance — periodic raw-audio retention sweep, daily backup, job hygiene

A watchdog in ``run_forever`` restarts any loop whose thread dies (with a
growing backoff), so one crashed loop can't silently disable the daemon.

The local web API is run separately via ``sb serve`` (or its own launchd job) so
capture keeps running even if the UI is restarted.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from datetime import datetime

from secondbrain.capture.recorder import Recorder
from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.pipeline import worker
from secondbrain.storage import retention, state
from secondbrain.storage.db import init_db
from secondbrain.storage.models import utcnow_iso

log = logging.getLogger("secondbrain.daemon")

WORKER_IDLE_SLEEP = 2.0
RETENTION_INTERVAL_S = 3600
# Maintenance ticks on a short cadence so conversation-stale-closing (and the
# heartbeat) stay fresh; the heavy sweeps only run once per RETENTION_INTERVAL_S.
MAINTENANCE_TICK_S = 60.0
DONE_JOB_KEEP_DAYS = 30
BACKUP_RUN_KEY = "backup:last_run"
# Watchdog: restart a dead loop after a growing delay (never gives up).
RESTART_BACKOFF_S = 5.0
RESTART_BACKOFF_MAX_S = 300.0

# Single-instance lease: two daemons on one DB double-capture the mic and race
# the queue. The lease (pid + start time) lives in app_state; a dead pid means
# a crashed daemon, whose stale lease is overridden with a log line.
LEASE_KEY = "daemon:lease"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def acquire_lease(conn) -> bool:
    """Take the single-daemon lease. False when a live daemon already holds it."""
    raw = state.get_state(conn, LEASE_KEY)
    if raw:
        try:
            info = json.loads(raw)
        except (ValueError, TypeError):
            info = None
        if info and int(info.get("pid", 0)) != os.getpid() and _pid_alive(int(info["pid"])):
            log.error(
                "another daemon (pid %s, started %s) already holds the lease — refusing to start",
                info["pid"], info.get("started_at"),
            )
            return False
        if info:
            log.warning("overriding stale daemon lease (pid %s not running)", info.get("pid"))
    state.set_state(
        conn, LEASE_KEY, json.dumps({"pid": os.getpid(), "started_at": utcnow_iso()})
    )
    return True


def release_lease(conn) -> None:
    """Drop the lease if this process holds it (best-effort, on clean shutdown)."""
    raw = state.get_state(conn, LEASE_KEY)
    try:
        holder = int(json.loads(raw).get("pid", 0)) if raw else 0
    except (ValueError, TypeError, AttributeError):
        holder = 0
    if holder == os.getpid():
        state.set_state(conn, LEASE_KEY, "")


class Daemon:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._stop = threading.Event()
        self._loops: dict[str, Callable[[], None]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._restart_delay: dict[str, float] = {}
        self._next_restart_at: dict[str, float] = {}
        self._recorder: Recorder | None = None

    # --- loops ---------------------------------------------------------------

    def _capture_loop(self) -> None:
        conn = init_db(settings=self.settings)
        self._recorder = Recorder(conn, self.settings)
        try:
            self._recorder.run()
        except Exception:  # noqa: BLE001
            log.exception("capture loop crashed")
        finally:
            conn.close()

    def _worker_loop(self) -> None:
        conn = init_db(settings=self.settings)
        try:
            while not self._stop.is_set():
                state.set_state(conn, "heartbeat:worker", utcnow_iso())
                try:
                    ran = worker.run_once(conn, settings=self.settings)
                except Exception:  # noqa: BLE001
                    log.exception("worker iteration failed")
                    ran = False
                if not ran:
                    self._stop.wait(WORKER_IDLE_SLEEP)
        finally:
            conn.close()

    def _maintenance_loop(self) -> None:
        conn = init_db(settings=self.settings)
        next_hourly = 0.0  # run the hourly block immediately on startup
        try:
            while not self._stop.is_set():
                state.set_state(conn, "heartbeat:maintenance", utcnow_iso())
                # Short cadence (~60s): close idle conversations promptly so
                # their diarization/extraction doesn't wait for the next hourly
                # sweep. Runs regardless of diarization: conversations are the
                # unit extraction consumes, and closing is cheap.
                self._conversation_maintenance(conn)
                if time.monotonic() >= next_hourly:
                    next_hourly = time.monotonic() + RETENTION_INTERVAL_S
                    self._hourly_maintenance(conn)
                self._stop.wait(MAINTENANCE_TICK_S)
        finally:
            conn.close()

    def _conversation_maintenance(self, conn) -> None:
        """Close idle conversations for diarization (runs every maintenance tick)."""
        from secondbrain.pipeline import conversation

        try:
            closed = conversation.close_stale_conversations(conn, self.settings)
            if closed:
                log.info("closed %d idle conversation(s) for diarization", closed)
        except Exception:  # noqa: BLE001
            log.exception("conversation close failed")

    def _hourly_maintenance(self, conn) -> None:
        """The heavy periodic work: sweeps, job hygiene, daily jobs, backup."""
        try:
            n = retention.sweep_expired_audio(conn, self.settings)
            if n:
                log.info("retention: deleted %d expired raw-audio files", n)
            reclaimed = q.reclaim_stale(conn)
            if reclaimed:
                log.warning("reclaimed %d stale 'running' job(s)", reclaimed)
            pruned = q.prune_done_jobs(conn, keep_days=DONE_JOB_KEEP_DAYS)
            if pruned:
                log.info("pruned %d completed job row(s)", pruned)
        except Exception:  # noqa: BLE001
            log.exception("retention/reclaim failed")
        if self.settings.diarization.enabled:
            self._diarization_maintenance(conn)
        if self.settings.extraction.enabled:
            self._extraction_catchup(conn)
        if self.settings.proactive.enabled:
            self._proactive_maintenance(conn)
        if self.settings.backup.auto_enabled:
            self._backup_maintenance(conn)

    def _backup_maintenance(self, conn) -> None:
        """Daily DB snapshot + prune, date-gated so it runs once per day."""
        from secondbrain.storage import backup

        today = utcnow_iso()[:10]
        last = (state.get_state(conn, BACKUP_RUN_KEY) or "")[:10]
        if last == today:
            return
        try:
            path = backup.backup_database(settings=self.settings)
            backup.prune_backups(settings=self.settings, keep=self.settings.backup.keep)
            state.set_state(conn, BACKUP_RUN_KEY, utcnow_iso())
            log.info("daily backup written to %s", path)
        except Exception:  # noqa: BLE001
            log.exception("scheduled backup failed")

    def _proactive_maintenance(self, conn) -> None:
        """Enqueue the daily morning brief and the weekly review when due."""
        from secondbrain.proactive import engine

        # digest_hour / weekly_review_weekday are LOCAL-time as documented; gate the
        # schedule on local time (and store matching local-date run keys). Digest
        # content generation keeps its own UTC clock, so only *when* it fires changes.
        now = datetime.now().astimezone()
        try:
            if engine.due_daily(conn, self.settings, now):
                q.enqueue(conn, engine.JOB_PROACTIVE, {"kind": "daily"}, dedupe_key="kind")
                state.set_state(conn, engine.DAILY_RUN_KEY, now.strftime("%Y-%m-%d"))
            if engine.due_weekly(conn, self.settings, now):
                q.enqueue(conn, engine.JOB_PROACTIVE, {"kind": "weekly"}, dedupe_key="kind")
                state.set_state(conn, engine.WEEKLY_RUN_KEY, now.strftime("%Y-W%W"))
        except Exception:  # noqa: BLE001
            log.exception("proactive enqueue failed")

    def _diarization_maintenance(self, conn) -> None:
        """Enqueue the daily clustering/reattribution jobs.

        (Idle-conversation closing runs on the faster maintenance tick via
        :meth:`_conversation_maintenance`; extraction catch-up runs from
        :meth:`_extraction_catchup` so it works with diarization disabled too.)
        """
        from secondbrain.pipeline import worker
        from secondbrain.speaker import cluster

        try:
            today = utcnow_iso()[:10]
            last = (state.get_state(conn, cluster.LAST_RUN_KEY) or "")[:10]
            if last != today:
                q.enqueue(conn, worker.JOB_CLUSTER, {}, dedupe_key=None)
                state.set_state(conn, cluster.LAST_RUN_KEY, utcnow_iso())
        except Exception:  # noqa: BLE001
            log.exception("clustering enqueue failed")
        try:
            from secondbrain.speaker import reattribute

            today = utcnow_iso()[:10]
            last = (state.get_state(conn, reattribute.LAST_RUN_KEY) or "")[:10]
            if last != today:
                q.enqueue(conn, worker.JOB_REATTRIBUTE, {}, dedupe_key=None)
                state.set_state(conn, reattribute.LAST_RUN_KEY, utcnow_iso())
        except Exception:  # noqa: BLE001
            log.exception("reattribution enqueue failed")

    def _extraction_catchup(self, conn) -> None:
        """Enqueue extraction for finished conversations not yet processed.

        Includes conversations whose diarization was skipped for missing chunk
        audio ('skipped_incomplete') — extraction only needs the transcript.
        """
        try:
            from secondbrain.knowledge.extract import enqueue_extraction

            rows = conn.execute(
                "SELECT id FROM conversations "
                "WHERE status IN ('diarized', 'skipped_incomplete') "
                "AND knowledge_status='pending'"
            ).fetchall()
            for r in rows:
                enqueue_extraction(conn, r["id"])
        except Exception:  # noqa: BLE001
            log.exception("extraction catch-up failed")

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.settings.ensure_dirs()
        init_db(settings=self.settings).close()  # create schema once up front
        # Self-heal on (re)start: reclaim crashed jobs, checkpoint WAL, fix dirs/schema.
        try:
            from secondbrain import repair
            from secondbrain.storage.db import db_session

            with db_session(settings=self.settings) as conn:
                for a in repair.repair(conn, self.settings):
                    if a.fixed:
                        log.info("self-heal: %s — %s", a.name, a.detail)
                    elif not a.ok:
                        log.warning("self-heal: %s — %s", a.name, a.detail)
        except Exception:  # noqa: BLE001 - repair is best-effort; never block startup
            log.warning("self-heal step failed", exc_info=True)
        self._loops = {
            "capture": self._capture_loop,
            "worker": self._worker_loop,
            "maintenance": self._maintenance_loop,
        }
        for name in self._loops:
            self._spawn(name)
        log.info("SecondBrain daemon started (%d loops)", len(self._threads))

    def _spawn(self, name: str) -> None:
        t = threading.Thread(target=self._loops[name], name=name, daemon=True)
        t.start()
        self._threads[name] = t

    def _check_threads(self) -> None:
        """Watchdog: restart any dead loop thread (with a growing backoff)."""
        if self._stop.is_set():
            return
        now = time.monotonic()
        for name, t in list(self._threads.items()):
            if t.is_alive():
                continue
            if now < self._next_restart_at.get(name, 0.0):
                continue  # still backing off
            delay = self._restart_delay.get(name, RESTART_BACKOFF_S)
            log.error("%s loop thread died — restarting (next backoff %.0fs)", name, delay)
            self._next_restart_at[name] = now + delay
            self._restart_delay[name] = min(RESTART_BACKOFF_MAX_S, delay * 2.0)
            self._spawn(name)

    def stop(self) -> None:
        log.info("stopping daemon…")
        self._stop.set()
        if self._recorder:
            self._recorder.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                self._check_threads()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> None:
    from secondbrain.logging_setup import configure_logging
    from secondbrain.storage.db import db_session

    configure_logging()
    daemon = Daemon()
    daemon.settings.ensure_dirs()
    init_db(settings=daemon.settings).close()
    with db_session(settings=daemon.settings) as conn:
        if not acquire_lease(conn):
            raise SystemExit(
                "Another SecondBrain daemon is already running against this database "
                "(see the log for its pid). Stop it first, or wait for launchd to."
            )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: daemon.stop())
    try:
        daemon.run_forever()
    finally:
        with db_session(settings=daemon.settings) as conn:
            release_lease(conn)


if __name__ == "__main__":
    main()
