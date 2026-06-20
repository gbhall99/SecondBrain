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
