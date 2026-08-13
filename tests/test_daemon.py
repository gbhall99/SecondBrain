"""Daemon maintenance loops — enqueue the right jobs, and only once per period."""

from __future__ import annotations

from secondbrain.config import Settings
from secondbrain.daemon import Daemon
from secondbrain.pipeline import worker
from secondbrain.proactive import engine
from secondbrain.speaker import cluster, reattribute
from secondbrain.storage import state


def _settings(tmp_path, **over) -> Settings:
    base = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "transcription": {"backend": "mock"},
        "search": {"semantic_enabled": False},
    }
    base.update(over)
    return Settings(**base)


def _pending_types(conn) -> list[str]:
    return [r["type"] for r in conn.execute("SELECT type FROM jobs WHERE state='pending'")]


def test_diarization_maintenance_enqueues_cluster_and_reattribute_once(conn, tmp_path):
    d = Daemon(settings=_settings(tmp_path))

    d._diarization_maintenance(conn)
    types = _pending_types(conn)
    assert worker.JOB_CLUSTER in types
    assert worker.JOB_REATTRIBUTE in types

    # date-gated: a second run the same day enqueues nothing new
    before = len(_pending_types(conn))
    d._diarization_maintenance(conn)
    assert len(_pending_types(conn)) == before


def test_diarization_maintenance_reenqueues_after_day_rolls_over(conn, tmp_path):
    d = Daemon(settings=_settings(tmp_path))
    d._diarization_maintenance(conn)
    n_after_first = len(_pending_types(conn))

    # backdate the last-run markers to "yesterday" → next run re-enqueues
    state.set_state(conn, cluster.LAST_RUN_KEY, "2000-01-01T00:00:00.000Z")
    state.set_state(conn, reattribute.LAST_RUN_KEY, "2000-01-01T00:00:00.000Z")
    d._diarization_maintenance(conn)
    assert len(_pending_types(conn)) > n_after_first


def test_proactive_maintenance_disabled_by_default_enqueues_when_due(conn, tmp_path):
    # proactive enabled; digest_hour=0 so it's always "due" by hour
    d = Daemon(settings=_settings(tmp_path, proactive={"enabled": True, "digest_hour": 0}))
    d._proactive_maintenance(conn)
    assert engine.JOB_PROACTIVE in _pending_types(conn)

    # idempotent within the day
    before = len(_pending_types(conn))
    d._proactive_maintenance(conn)
    assert len(_pending_types(conn)) == before


def test_proactive_maintenance_swallows_errors(conn, tmp_path, monkeypatch):
    d = Daemon(settings=_settings(tmp_path, proactive={"enabled": True, "digest_hour": 0}))
    monkeypatch.setattr(engine, "due_daily", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # must not raise — maintenance is best-effort
    d._proactive_maintenance(conn)


def test_conversation_maintenance_closes_stale_open_conversations(conn, tmp_path):
    d = Daemon(settings=_settings(tmp_path, diarization={"enabled": True, "backend": "mock"}))
    conn.execute(
        "INSERT INTO conversations (started_at, ended_at, status) "
        "VALUES ('2026-06-16T09:00:00.000Z', '2026-06-16T09:05:00.000Z', 'open')"
    )
    d._conversation_maintenance(conn)  # runs each ~60s tick, not just hourly
    row = conn.execute("SELECT status FROM conversations").fetchone()
    assert row["status"] in ("closed", "diarized")  # short ones are finalized directly


def test_hourly_maintenance_prunes_old_done_jobs(conn, tmp_path):
    from secondbrain.pipeline import queue as q

    d = Daemon(settings=_settings(tmp_path))
    q.enqueue(conn, "transcribe", {"audio_file_id": 1})
    job = q.claim_next(conn)
    q.complete(conn, job.id)
    conn.execute("UPDATE jobs SET finished_at='2000-01-01T00:00:00.000Z'")
    d._hourly_maintenance(conn)
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


def test_backup_maintenance_runs_once_per_day(conn, tmp_path, settings):
    # use the shared `settings` fixture so the live DB (conn) is the one backed up
    d = Daemon(settings=settings)
    backups_dir = settings.data_path / "backups"
    d._backup_maintenance(conn)
    assert len(list(backups_dir.glob("secondbrain-*.db"))) == 1
    d._backup_maintenance(conn)  # date-gated: same day → no second snapshot
    assert len(list(backups_dir.glob("secondbrain-*.db"))) == 1

    # backdate the marker → next run snapshots (and prunes to [backup].keep)
    state.set_state(conn, "backup:last_run", "2000-01-01T00:00:00.000Z")
    d._backup_maintenance(conn)
    snaps = list(backups_dir.glob("secondbrain-*.db"))
    assert 1 <= len(snaps) <= settings.backup.keep


def test_hourly_maintenance_skips_backup_when_disabled(conn, settings):
    settings.backup.auto_enabled = False
    d = Daemon(settings=settings)
    d._hourly_maintenance(conn)
    assert not list((settings.data_path / "backups").glob("secondbrain-*.db"))


def test_watchdog_restarts_dead_thread_with_backoff(tmp_path):
    import threading
    import time

    d = Daemon(settings=_settings(tmp_path))
    runs = []

    def flaky_loop():
        runs.append(time.monotonic())  # dies immediately after starting

    d._loops = {"flaky": flaky_loop}
    d._spawn("flaky")
    d._threads["flaky"].join(timeout=5)
    assert not d._threads["flaky"].is_alive()

    d._check_threads()  # dead → restarted
    d._threads["flaky"].join(timeout=5)
    assert len(runs) == 2

    d._check_threads()  # dead again but within the backoff window → NOT restarted
    assert len(runs) == 2
    assert d._restart_delay["flaky"] > 5.0  # backoff grew

    d._next_restart_at["flaky"] = 0.0  # backoff elapsed → restarts again
    d._check_threads()
    d._threads["flaky"].join(timeout=5)
    assert len(runs) == 3

    # a stopping daemon never restarts threads
    d._stop.set()
    d._next_restart_at["flaky"] = 0.0
    d._check_threads()
    assert len(runs) == 3
    assert isinstance(d._threads["flaky"], threading.Thread)
