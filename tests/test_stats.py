"""corpus_stats — high-level overview counts and date span."""

from __future__ import annotations

from secondbrain.query import service


def test_empty_corpus_stats(conn):
    s = service.corpus_stats(conn)
    assert s["segments"] == 0
    assert s["kg_nodes"] == 0
    assert s["goals"] == 0
    assert s["first_day"] is None
    assert s["last_day"] is None


def test_corpus_stats_counts_and_span(conn):
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (1, '/tmp/a.flac', '2026-06-15T09:00:00.000Z', 16000, 'transcribed')"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (1, 1, 'mock')")
    for sid, day in [(1, "2026-06-15"), (2, "2026-06-17")]:
        conn.execute(
            "INSERT INTO transcript_segments "
            "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text) "
            "VALUES (?, 1, 1, 0, 1, ?, 'hi')",
            (sid, f"{day}T09:00:00.000Z"),
        )
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (1, 'project', 'Atlas')")
    gid = service.create_goal(conn, title="Ship", priority=1)
    service.set_goal_status(conn, gid, "active")
    service.create_task(conn, title="do thing")

    s = service.corpus_stats(conn)
    assert s["segments"] == 2
    assert s["speakers"] == 1
    assert s["kg_nodes"] == 1
    assert s["goals"] == 1
    assert s["goals_active"] == 1
    assert s["tasks"] == 1
    assert s["tasks_open"] == 1
    # Local calendar days (same bucketing as segments_today, search day groups,
    # and the /day view), derived from the stored UTC timestamps.
    assert s["first_day"] == service._local_day_of("2026-06-15T09:00:00.000Z")
    assert s["last_day"] == service._local_day_of("2026-06-17T09:00:00.000Z")
