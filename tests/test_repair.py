"""Self-healing repair: safe, idempotent auto-remediation."""

from __future__ import annotations

import pytest

from secondbrain import repair
from secondbrain.pipeline import queue as q
from secondbrain.storage import schema


@pytest.fixture(autouse=True)
def _isolate_repo_root(tmp_path, monkeypatch):
    # repair()'s config-seed writes <repo>/config.local.toml — keep it out of
    # the real repo by pointing the resolved repo root at a temp dir.
    monkeypatch.setattr(repair, "REPO_ROOT", tmp_path)


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


class _CorruptConn:
    class _Cur:
        def fetchone(self):
            return ("malformed database page",)

    def execute(self, sql, *a, **k):
        return self._Cur()


def test_repair_flags_corruption_without_deleting(settings):
    # A failed integrity check is surfaced (ok=False), not silently "fixed".
    action = repair._integrity(_CorruptConn(), settings)
    assert action.ok is False and action.fixed is False and "restore" in action.detail
    assert "no backups found" in action.detail  # nothing to restore from yet


def test_corruption_message_names_newest_backup(settings):
    backups_dir = settings.data_path / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    (backups_dir / "secondbrain-20260101-000000.db").write_bytes(b"x")
    newest = backups_dir / "secondbrain-20260201-000000.db"
    newest.write_bytes(b"x")
    action = repair._integrity(_CorruptConn(), settings)
    assert action.ok is False
    assert str(newest) in action.detail
    assert "d old" in action.detail


def test_repair_reenqueues_recorded_chunks_with_no_job(conn, settings):
    from secondbrain.storage import models
    from secondbrain.storage.models import AudioFile

    # a 'recorded' chunk whose transcription job vanished
    af_orphan = models.insert_audio_file(
        conn, AudioFile(path="/tmp/orphan.flac", started_at="2026-06-16T09:00:00.000Z",
                        sample_rate=16000, status="recorded"))
    # a 'recorded' chunk with a live pending job — must NOT be double-enqueued
    af_queued = models.insert_audio_file(
        conn, AudioFile(path="/tmp/queued.flac", started_at="2026-06-16T09:01:00.000Z",
                        sample_rate=16000, status="recorded"))
    q.enqueue(conn, "transcribe", {"audio_file_id": af_queued}, dedupe_key="audio_file_id")

    actions = _names(repair.repair(conn, settings))
    assert actions["orphan chunks"].fixed
    jobs = conn.execute(
        "SELECT json_extract(payload, '$.audio_file_id') AS af FROM jobs "
        "WHERE type='transcribe' AND state='pending'"
    ).fetchall()
    assert sorted(j["af"] for j in jobs) == sorted([af_orphan, af_queued])

    # idempotent: a second run finds nothing new
    actions = _names(repair.repair(conn, settings))
    assert actions["orphan chunks"].fixed is False


def test_repair_reenqueues_stuck_conversations(conn, settings):
    # stuck mid-pipeline with no live diarize job
    stuck = conn.execute(
        "INSERT INTO conversations (started_at, status) "
        "VALUES ('2026-06-16T09:00:00.000Z', 'diarizing')"
    ).lastrowid
    # closed but its job is still pending — must not be double-enqueued
    queued = conn.execute(
        "INSERT INTO conversations (started_at, status) "
        "VALUES ('2026-06-16T10:00:00.000Z', 'closed')"
    ).lastrowid
    q.enqueue(conn, "diarize_conversation", {"conversation_id": queued},
              dedupe_key="conversation_id")
    # terminal states are left alone
    conn.execute(
        "INSERT INTO conversations (started_at, status) "
        "VALUES ('2026-06-16T11:00:00.000Z', 'diarized')"
    )

    actions = _names(repair.repair(conn, settings))
    assert actions["stuck conversations"].fixed
    jobs = conn.execute(
        "SELECT json_extract(payload, '$.conversation_id') AS cid FROM jobs "
        "WHERE type='diarize_conversation' AND state='pending'"
    ).fetchall()
    assert sorted(j["cid"] for j in jobs) == sorted([stuck, queued])

    actions = _names(repair.repair(conn, settings))
    assert actions["stuck conversations"].fixed is False


def test_local_config_seeds_from_repo_root_not_cwd(tmp_path, monkeypatch):
    # CWD points elsewhere; the seed must still land next to the example.
    monkeypatch.setattr(repair, "REPO_ROOT", tmp_path)
    (tmp_path / "config.local.toml.example").write_text("[capture]\ninput_device = \"\"\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    action = repair._local_config()
    assert action.fixed
    assert (tmp_path / "config.local.toml").exists()
    assert not (elsewhere / "config.local.toml").exists()
