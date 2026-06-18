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


def test_phase2_schema_present(conn):
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"conversations", "speaker_observations"} <= tables
    speaker_cols = {r["name"] for r in conn.execute("PRAGMA table_info(speakers)").fetchall()}
    assert {"kind", "centroid", "merged_into", "segment_count"} <= speaker_cols
    seg_cols = {r["name"] for r in conn.execute("PRAGMA table_info(transcript_segments)").fetchall()}
    assert "speaker_confidence" in seg_cols
    af_cols = {r["name"] for r in conn.execute("PRAGMA table_info(audio_files)").fetchall()}
    assert "conversation_id" in af_cols


def test_phase3_schema_present(conn):
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"kg_nodes", "kg_aliases", "kg_edges", "knowledge_extractions"} <= tables
    conv_cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    assert "knowledge_status" in conv_cols


def test_phase4_schema_present(conn):
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"goals", "goal_links", "suggestions", "suggestion_feedback", "digests"} <= tables


def test_apply_base_schema_is_idempotent(conn):
    # second application must not raise on the non-idempotent ADD COLUMNs
    from secondbrain.storage.schema import apply_base_schema

    apply_base_schema(conn)
    ver = conn.execute("SELECT version_num FROM alembic_version").fetchone()["version_num"]
    assert ver == "0004_proactive"


def test_pause_state_roundtrip(conn):
    assert state.is_paused(conn, default=False) is False
    state.set_paused(conn, True)
    assert state.is_paused(conn) is True
    state.set_paused(conn, False)
    assert state.is_paused(conn) is False
