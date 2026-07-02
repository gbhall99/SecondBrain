"""Self-healing repair: safe, idempotent auto-remediation."""

from __future__ import annotations

import pytest

from secondbrain import repair
from secondbrain.pipeline import queue as q
from secondbrain.storage import schema


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # repair()'s config-seed writes ./config.local.toml — keep it out of the repo.
    monkeypatch.chdir(tmp_path)


def _names(actions):
    return {a.name: a for a in actions}


def test_repair_is_clean_on_healthy_db(conn, settings):
    settings.ensure_dirs()
    actions = _names(repair.repair(conn, settings))
    assert actions["schema"].detail.startswith("at head")
    assert actions["integrity"].detail == "ok"
    assert actions["stale jobs"].fixed is False  # nothing to reclaim


def test_repair_creates_missing_dirs(conn, settings):
    import shutil

    shutil.rmtree(settings.audio_raw_dir, ignore_errors=True)
    assert not settings.audio_raw_dir.exists()
    actions = _names(repair.repair(conn, settings))
    assert actions["data dirs"].fixed
    assert settings.audio_raw_dir.exists()


def test_repair_reclaims_crashed_jobs(conn, settings):
    # A job stuck in 'running' (worker died mid-job) is re-queued.
    jid = q.enqueue(conn, "transcribe", {"x": 1})
    q.claim_next(conn)  # → 'running', started_at = now
    conn.execute(
        "UPDATE jobs SET started_at='2000-01-01T00:00:00.000Z' WHERE id=?", (jid,)
    )
    actions = _names(repair.repair(conn, settings))
    assert actions["stale jobs"].fixed
    assert conn.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()["state"] == "pending"


def test_repair_upgrades_stale_schema(conn, settings):
    conn.execute("UPDATE alembic_version SET version_num='0001_initial'")
    actions = _names(repair.repair(conn, settings))
    assert actions["schema"].fixed
    ver = conn.execute("SELECT version_num FROM alembic_version").fetchone()["version_num"]
    assert ver == schema.SCHEMA_VERSION


def test_repair_flags_corruption_without_deleting():
    # A failed integrity check is surfaced (ok=False), not silently "fixed".
    class _Cur:
        def fetchone(self):
            return ("malformed database page",)

    class _FakeConn:
        def execute(self, sql, *a, **k):
            return _Cur()

    action = repair._integrity(_FakeConn())
    assert action.ok is False and action.fixed is False and "restore" in action.detail
