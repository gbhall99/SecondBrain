"""db.transaction — atomic multi-step writes (commit / rollback / reentrant)."""

from __future__ import annotations

import pytest

from secondbrain.speaker import registry
from secondbrain.storage import state
from secondbrain.storage.db import transaction


def test_transaction_commits(conn):
    with transaction(conn):
        state.set_state(conn, "k", "v")
    assert state.get_state(conn, "k") == "v"


def test_transaction_rolls_back_on_error(conn):
    state.set_state(conn, "k", "orig")
    with pytest.raises(RuntimeError), transaction(conn):
        state.set_state(conn, "k", "changed")
        raise RuntimeError("boom")
    assert state.get_state(conn, "k") == "orig"  # change rolled back


def test_transaction_reentrant(conn):
    # An inner transaction must not start a nested BEGIN (SQLite has none).
    with transaction(conn), transaction(conn):
        state.set_state(conn, "k", "v")
    assert state.get_state(conn, "k") == "v"


def test_merge_speakers_atomic_on_failure(conn, settings, monkeypatch):
    """If a later step fails, the whole merge rolls back (no half-merged speaker)."""
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'A', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'B', 'known', 0)")
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (1, '/a.flac', '2026-06-16T09:00:00.000Z', 16000, 'transcribed')"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (1, 1, 'mock')")
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, text, speaker_id) "
        "VALUES (1, 1, 1, 0, 1, 'hi', 1)"
    )

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(registry, "recompute_centroid", boom)
    with pytest.raises(RuntimeError):
        registry.merge_speakers(conn, 1, 2, settings)

    # Nothing committed: src not merged, segment still attributed to src.
    assert conn.execute("SELECT merged_into FROM speakers WHERE id=1").fetchone()["merged_into"] is None
    assert conn.execute("SELECT speaker_id FROM transcript_segments WHERE id=1").fetchone()[0] == 1
