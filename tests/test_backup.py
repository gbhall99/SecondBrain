"""Backup & export — snapshot integrity + portable dumps that honour opt-out."""

from __future__ import annotations

import json
import sqlite3

import pytest

from secondbrain.query import service
from secondbrain.storage import backup
from secondbrain.storage.backup import RestoreError


def _seed(conn) -> None:
    # owner + a normal speaker + an opted-out speaker
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) "
        "VALUES (1, 'Me', 'owner', 1, 0)"
    )
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) "
        "VALUES (2, 'Alice', 'known', 0, 0)"
    )
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) "
        "VALUES (3, 'Mallory', 'known', 0, 1)"
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (1, '/tmp/a.flac', '2026-06-16T09:00:00.000Z', 16000, 'transcribed')"
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (1, 1, 'mock')"
    )
    rows = [
        (1, "2026-06-16T09:00:00.000Z", "hello from me", 1, 0.99),
        (2, "2026-06-16T09:00:05.000Z", "alice speaking", 2, 0.91),
        (3, "2026-06-16T09:00:10.000Z", "secret from mallory", 3, 0.88),
    ]
    for sid, at, text, spk, conf in rows:
        conn.execute(
            "INSERT INTO transcript_segments "
            "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, "
            " start_at, text, speaker_id, speaker_confidence) "
            "VALUES (?, 1, 1, 0, 1, ?, ?, ?, ?)",
            (sid, at, text, spk, conf),
        )
    service.create_goal(conn, title="Ship pricing", description="revamp", priority=1)


def test_backup_database_produces_openable_copy(conn, settings, tmp_path):
    _seed(conn)
    dest = tmp_path / "snap.db"
    out = service.backup_database(settings=settings, dest=dest)
    assert out == dest and dest.exists()
    # the snapshot is a valid SQLite DB with the same data
    snap = sqlite3.connect(str(dest))
    try:
        n = snap.execute("SELECT COUNT(*) FROM transcript_segments").fetchone()[0]
    finally:
        snap.close()
    assert n == 3


def test_export_json_excludes_opted_out(conn, settings, tmp_path):
    _seed(conn)
    path = backup.export_json(conn, tmp_path, settings)
    data = json.loads(path.read_text())
    texts = [s["text"] for s in data["segments"]]
    assert "hello from me" in texts
    assert "alice speaking" in texts
    assert "secret from mallory" not in texts  # opted-out speaker excluded
    speakers = [s["text"] for s in data["segments"]]
    assert any(s == "hello from me" for s in speakers)
    # owner label resolves to "Me"
    me = next(s for s in data["segments"] if s["text"] == "hello from me")
    assert me["speaker"] == "Me"
    assert any(g["title"] == "Ship pricing" for g in data["goals"])


def test_export_markdown_excludes_opted_out(conn, settings, tmp_path):
    _seed(conn)
    path = backup.export_markdown(conn, tmp_path, settings)
    md = path.read_text()
    assert "hello from me" in md
    assert "alice speaking" in md
    assert "secret from mallory" not in md
    assert "## 2026-06-16" in md
    assert "Ship pricing" in md  # goals section


def test_export_data_both_formats(conn, settings, tmp_path):
    _seed(conn)
    paths = service.export_data(conn, tmp_path, fmt="both", settings=settings)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".json", ".md"]


def test_prune_backups_keeps_newest(conn, settings):
    backups_dir = settings.data_path / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    # 5 snapshots with ascending timestamps in the filename
    names = [
        "secondbrain-20260101-000000.db",
        "secondbrain-20260102-000000.db",
        "secondbrain-20260103-000000.db",
        "secondbrain-20260104-000000.db",
        "secondbrain-20260105-000000-pre-restore.db",
    ]
    for n in names:
        (backups_dir / n).write_bytes(b"x")

    removed = service.prune_backups(settings=settings, keep=2)
    assert removed == 3
    remaining = sorted(p.name for p in backups_dir.glob("secondbrain-*.db"))
    assert remaining == [
        "secondbrain-20260104-000000.db",
        "secondbrain-20260105-000000-pre-restore.db",
    ]


def test_prune_backups_keep_zero_is_noop(conn, settings):
    backups_dir = settings.data_path / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    (backups_dir / "secondbrain-20260101-000000.db").write_bytes(b"x")
    assert service.prune_backups(settings=settings, keep=0) == 0
    assert list(backups_dir.glob("secondbrain-*.db"))  # nothing deleted


def test_prune_backups_no_dir(settings):
    assert service.prune_backups(settings=settings, keep=5) == 0


def test_restore_replaces_live_db_and_backs_up_current(conn, settings, tmp_path):
    _seed(conn)
    snap = service.backup_database(settings=settings, dest=tmp_path / "snap.db")
    # mutate the live DB after the snapshot
    conn.execute("DELETE FROM transcript_segments")
    conn.close()

    restored = service.restore_database(settings=settings, src=snap)
    assert restored == settings.db_path

    check = sqlite3.connect(str(settings.db_path))
    try:
        n = check.execute("SELECT COUNT(*) FROM transcript_segments").fetchone()[0]
    finally:
        check.close()
    assert n == 3  # snapshot's rows are back

    # current DB was snapshotted to a *-pre-restore.db before replacement
    pre = list((settings.data_path / "backups").glob("*-pre-restore.db"))
    assert pre


def test_restore_rejects_non_database(conn, settings, tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("just some text, not sqlite")
    with pytest.raises(RestoreError):
        service.restore_database(settings=settings, src=bogus)


def test_restore_rejects_missing_source(conn, settings, tmp_path):
    with pytest.raises(RestoreError):
        service.restore_database(settings=settings, src=tmp_path / "nope.db")


def test_restore_rejects_unrelated_sqlite_db(conn, settings, tmp_path):
    other = tmp_path / "other.db"
    c = sqlite3.connect(str(other))
    c.execute("CREATE TABLE foo (id INTEGER)")
    c.commit()
    c.close()
    with pytest.raises(RestoreError):
        service.restore_database(settings=settings, src=other)
