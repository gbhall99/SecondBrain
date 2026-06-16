from secondbrain.storage import models, state
from secondbrain.storage.models import AudioFile, Segment


def test_schema_created(conn):
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"audio_files", "transcripts", "transcript_segments", "jobs", "app_state"} <= tables


def test_fts_trigger_indexes_segments(conn):
    af_id = models.insert_audio_file(
        conn, AudioFile(path="/x.flac", started_at="2026-06-16T10:00:00.000Z", sample_rate=16000)
    )
    t_id = models.insert_transcript(conn, af_id, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [Segment(t_id, af_id, 0.0, 2.0, "we decided to ship the pricing change")],
    )
    rows = conn.execute(
        "SELECT rowid FROM transcript_segments_fts WHERE transcript_segments_fts MATCH 'pricing'"
    ).fetchall()
    assert len(rows) == 1


def test_fts_trigger_handles_delete(conn):
    af_id = models.insert_audio_file(
        conn, AudioFile(path="/y.flac", started_at="2026-06-16T10:00:00.000Z", sample_rate=16000)
    )
    t_id = models.insert_transcript(conn, af_id, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(t_id, af_id, 0.0, 1.0, "ephemeral words here")])
    conn.execute("DELETE FROM transcript_segments")
    rows = conn.execute(
        "SELECT rowid FROM transcript_segments_fts WHERE transcript_segments_fts MATCH 'ephemeral'"
    ).fetchall()
    assert rows == []


def test_pause_state_roundtrip(conn):
    assert state.is_paused(conn, default=False) is False
    state.set_paused(conn, True)
    assert state.is_paused(conn) is True
    state.set_paused(conn, False)
    assert state.is_paused(conn) is False
