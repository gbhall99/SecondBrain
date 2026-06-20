from secondbrain.speaker import correct, registry
from secondbrain.speaker.attribution import _overlap_count
from secondbrain.speaker.reattribute import run_reattribution
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def _known(conn, name, centroid):
    sid = conn.execute(
        "INSERT INTO speakers (name, kind, display_label) VALUES (?, 'known', ?)", (name, name)
    ).lastrowid
    registry.update_centroid(conn, sid, centroid)
    return int(sid)


def _segment_with_obs(conn, speaker_id, embedding, *, confidence, locked=0):
    af = models.insert_audio_file(
        conn, AudioFile(path="/tmp/x.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 2.0, "hello", start_at="2026-06-16T09:00:00.000Z")])
    seg = conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"]
    obs = registry.record_observation(
        conn, speaker_id=speaker_id, audio_file_id=af, conversation_id=None,
        start_offset_s=0.0, end_offset_s=2.0, start_at="2026-06-16T09:00:00.000Z",
        confidence=confidence, embedding=embedding,
    )
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=?, observation_id=?, "
        "speaker_locked=? WHERE id=?",
        (speaker_id, confidence, obs, locked, seg),
    )
    return seg, obs


# --- exemplar-aware matching -------------------------------------------------


def test_exemplar_match_beats_centroid_only(conn, settings):
    alice = _known(conn, "Alice", [1.0, 0.0, 0.0, 0.0])
    probe = [0.0, 1.0, 0.0, 0.0]
    # centroid-only: orthogonal → no match
    assert registry.match_embedding(conn, probe, settings).speaker_id is None
    # add an exemplar near the probe → now matches
    registry.record_observation(
        conn, speaker_id=alice, audio_file_id=None, conversation_id=None,
        start_offset_s=0.0, end_offset_s=1.0, start_at=None, confidence=0.9, embedding=probe,
    )
    m = registry.match_embedding(conn, probe, settings)
    assert m.speaker_id == alice and m.similarity > 0.99


# --- pruning -----------------------------------------------------------------


def test_prune_drops_low_quality_and_caps(conn, settings):
    settings.diarization.prune_min_confidence = 0.5
    settings.diarization.max_exemplars_per_speaker = 2
    sid = registry.create_unknown_speaker(conn)
    for q in (0.9, 0.8, 0.7, 0.2):  # 0.2 is low quality; cap keeps top 2
        registry.record_observation(
            conn, speaker_id=sid, audio_file_id=None, conversation_id=None,
            start_offset_s=0, end_offset_s=1, start_at=None, confidence=q, embedding=[q, 0, 0, 0],
        )
    pruned = registry.prune_exemplars(conn, sid, settings)
    kept = conn.execute(
        "SELECT COUNT(*) AS n FROM speaker_observations WHERE speaker_id=? AND pruned=0", (sid,)
    ).fetchone()["n"]
    assert pruned == 2 and kept == 2


# --- re-attribution ----------------------------------------------------------


def test_reattribution_relabels_low_confidence(conn, settings):
    alice = _known(conn, "Alice", [1.0, 0.0, 0.0, 0.0])
    unknown = registry.create_unknown_speaker(conn)
    seg, _ = _segment_with_obs(conn, unknown, [1.0, 0.0, 0.0, 0.0], confidence=0.1)
    n = run_reattribution(conn, settings)
    assert n == 1
    row = conn.execute("SELECT speaker_id, speaker_source FROM transcript_segments WHERE id=?", (seg,)).fetchone()
    assert row["speaker_id"] == alice and row["speaker_source"] == "reattributed"


def test_reattribution_skips_locked(conn, settings):
    _known(conn, "Alice", [1.0, 0.0, 0.0, 0.0])
    unknown = registry.create_unknown_speaker(conn)
    seg, _ = _segment_with_obs(conn, unknown, [1.0, 0.0, 0.0, 0.0], confidence=0.1, locked=1)
    assert run_reattribution(conn, settings) == 0
    assert conn.execute("SELECT speaker_id FROM transcript_segments WHERE id=?", (seg,)).fetchone()["speaker_id"] == unknown


# --- correction loop ---------------------------------------------------------


def test_correction_locks_and_teaches(conn, settings):
    alice = _known(conn, "Alice", [1.0, 0.0, 0.0, 0.0])
    unknown = registry.create_unknown_speaker(conn)
    seg, _ = _segment_with_obs(conn, unknown, [0.0, 1.0, 0.0, 0.0], confidence=0.2)
    assert correct.reassign_segment(conn, seg, alice, settings)
    row = conn.execute(
        "SELECT speaker_id, speaker_locked, speaker_source FROM transcript_segments WHERE id=?", (seg,)
    ).fetchone()
    assert row["speaker_id"] == alice and row["speaker_locked"] == 1 and row["speaker_source"] == "user"
    # the confirmed exemplar now lets Alice match that voice
    assert registry.match_embedding(conn, [0.0, 1.0, 0.0, 0.0], settings).speaker_id == alice


# --- overlap helper ----------------------------------------------------------


def test_overlap_count(conn):
    from secondbrain.pipeline.diarize import SpeakerTurn

    turns = [SpeakerTurn(0.0, 2.0, "S0"), SpeakerTurn(1.0, 3.0, "S1")]
    assert _overlap_count(turns, 1.0, 1.5) == 2   # both speakers overlap
    assert _overlap_count(turns, 2.5, 3.0) == 1   # only S1
