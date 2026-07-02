"""Always-on supervisor (launchd entrypoint).

Runs three cooperating loops, each with its own SQLite connection:
  1. capture  — the rolling recorder (room audio -> FLAC chunks -> queue)
  2. worker   — drains transcription jobs (VAD -> transcribe -> store -> index)
  3. maintenance — periodic raw-audio retention sweep

The local web API is run separately via ``sb serve`` (or its own launchd job) so
capture keeps running even if the UI is restarted.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
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


class Daemon:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
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
        try:
            while not self._stop.is_set():
                state.set_state(conn, "heartbeat:maintenance", utcnow_iso())
                try:
                    n = retention.sweep_expired_audio(conn, self.settings)
                    if n:
                        log.info("retention: deleted %d expired raw-audio files", n)
                    reclaimed = q.reclaim_stale(conn)
                    if reclaimed:
                        log.warning("reclaimed %d stale 'running' job(s)", reclaimed)
                except Exception:  # noqa: BLE001
                    log.exception("retention/reclaim failed")
                if self.settings.diarization.enabled:
                    self._diarization_maintenance(conn)
                if self.settings.proactive.enabled:
                    self._proactive_maintenance(conn)
                self._stop.wait(RETENTION_INTERVAL_S)
        finally:
            conn.close()

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
        """Close idle conversations for diarization; enqueue clustering daily."""
        from secondbrain.pipeline import conversation, worker
        from secondbrain.speaker import cluster

        try:
            closed = conversation.close_stale_conversations(conn, self.settings)
            if closed:
                log.info("closed %d idle conversation(s) for diarization", closed)
        except Exception:  # noqa: BLE001
            log.exception("conversation close failed")
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
        # Catch-up: enqueue extraction for diarized conversations not yet processed.
        if self.settings.extraction.enabled:
            try:
                from secondbrain.knowledge.extract import enqueue_extraction

                rows = conn.execute(
                    "SELECT id FROM conversations WHERE status='diarized' "
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
        for target in (self._capture_loop, self._worker_loop, self._maintenance_loop):
            t = threading.Thread(target=target, name=target.__name__, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("SecondBrain daemon started (%d loops)", len(self._threads))

    def stop(self) -> None:
        log.info("stopping daemon…")
        self._stop.set()
        if self._recorder:
            self._recorder.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> None:
    from secondbrain.logging_setup import configure_logging

    configure_logging()
    daemon = Daemon()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: daemon.stop())
    daemon.run_forever()


if __name__ == "__main__":
    main()
