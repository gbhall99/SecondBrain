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


def test_list_backups_newest_first(conn, settings):
    backups_dir = settings.data_path / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    (backups_dir / "secondbrain-20260101-000000.db").write_bytes(b"aa")
    (backups_dir / "secondbrain-20260103-000000.db").write_bytes(b"cccc")
    (backups_dir / "secondbrain-20260102-000000-pre-restore.db").write_bytes(b"bbb")
    (backups_dir / "ignore-me.db").write_bytes(b"z")  # not a snapshot

    rows = service.list_backups(settings=settings)
    names = [r["name"] for r in rows]
    assert names == [
        "secondbrain-20260103-000000.db",
        "secondbrain-20260102-000000-pre-restore.db",
        "secondbrain-20260101-000000.db",
    ]
    assert rows[0]["size_bytes"] == 4
    assert "modified" in rows[0]


def test_list_backups_no_dir(settings):
    assert service.list_backups(settings=settings) == []


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


def test_export_json_date_range(conn, settings, tmp_path):
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)"
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (1, '/tmp/a.flac', '2026-06-10T09:00:00.000Z', 16000, 'transcribed')"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (1, 1, 'mock')")
    for sid, day, text in [
        (1, "2026-06-10", "early"),
        (2, "2026-06-15", "middle"),
        (3, "2026-06-20", "late"),
    ]:
        conn.execute(
            "INSERT INTO transcript_segments "
            "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
            " speaker_id) VALUES (?, 1, 1, 0, 1, ?, ?, 1)",
            (sid, f"{day}T09:00:00.000Z", text),
        )

    path = backup.export_json(conn, tmp_path, settings, since="2026-06-12", until="2026-06-18")
    texts = [s["text"] for s in json.loads(path.read_text())["segments"]]
    assert texts == ["middle"]


def test_export_markdown_since_only(conn, settings, tmp_path):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (1, '/tmp/a.flac', '2026-06-10T09:00:00.000Z', 16000, 'transcribed')"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (1, 1, 'mock')")
    for sid, day, text in [(1, "2026-06-10", "old"), (2, "2026-06-20", "new")]:
        conn.execute(
            "INSERT INTO transcript_segments "
            "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
            " speaker_id) VALUES (?, 1, 1, 0, 1, ?, ?, 1)",
            (sid, f"{day}T09:00:00.000Z", text),
        )

    md = backup.export_markdown(conn, tmp_path, settings, since="2026-06-15").read_text()
    assert "new" in md and "old" not in md


def test_export_date_range_retains_undated_segments(conn, settings, tmp_path):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (1, '/tmp/a.flac', '2026-06-10T09:00:00.000Z', 16000, 'transcribed')"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (1, 1, 'mock')")
    # one dated (in range), one dated (out of range), one undated (NULL start_at)
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, speaker_id)"
        " VALUES (1, 1, 1, 0, 1, '2026-06-15T09:00:00.000Z', 'dated in range', 1)"
    )
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, speaker_id)"
        " VALUES (2, 1, 1, 0, 1, '2026-06-01T09:00:00.000Z', 'dated out of range', 1)"
    )
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, speaker_id)"
        " VALUES (3, 1, 1, 0, 1, NULL, 'undated', 1)"
    )
    texts = [
        s["text"]
        for s in json.loads(
            backup.export_json(conn, tmp_path, settings, since="2026-06-12", until="2026-06-18")
            .read_text()
        )["segments"]
    ]
    assert "dated in range" in texts
    assert "dated out of range" not in texts
    assert "undated" in texts  # NULL start_at retained, not silently dropped


def test_encrypted_backup_restore_round_trip(tmp_path):
    from secondbrain.storage import db as dbmod

    if not dbmod.sqlcipher_available():
        pytest.skip("SQLCipher driver not installed (the [secure] extra)")
    from secondbrain.config import Settings
    from secondbrain.storage.db import init_db

    s = Settings(
        paths={"data_dir": str(tmp_path / "data")},
        security={"encrypt_db": True, "db_passphrase": "correct horse"},
        transcription={"backend": "mock"}, vad={"enabled": False},
        search={"semantic_enabled": False},
    )
    s.ensure_dirs()
    c = init_db(settings=s)
    c.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    c.commit()
    c.close()
    snap = backup.backup_database(settings=s, dest=tmp_path / "enc-snap.db")
    # the snapshot must itself be encrypted (unreadable as plaintext sqlite)
    plain = sqlite3.connect(str(snap))
    with pytest.raises(sqlite3.DatabaseError):
        plain.execute("SELECT name FROM sqlite_master").fetchall()
    plain.close()
    # and it restores cleanly
    backup.restore_database(settings=s, src=snap)
    c2 = init_db(settings=s)
    assert c2.execute("SELECT COUNT(*) FROM speakers").fetchone()[0] == 1
    c2.close()


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
