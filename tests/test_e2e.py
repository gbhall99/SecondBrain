"""Cross-phase end-to-end: capture-registration → transcribe → conversation →
diarize → extract → knowledge graph. Uses mock backends only (no audio/LLM)."""

import json
from pathlib import Path

from secondbrain.capture import recorder
from secondbrain.knowledge.extract import run_extraction
from secondbrain.llm.client import MockLLM
from secondbrain.pipeline import conversation, worker
from secondbrain.pipeline.diarize import MockDiarizer, deterministic_embedding
from secondbrain.pipeline.transcribe import MockTranscriber, TranscribedSegment
from secondbrain.pipeline.vad import MockVad
from secondbrain.speaker import attribution


def _fake_builder(conn, chunks, settings):
    return Path("/tmp/concat.wav"), attribution.concat_offsets_from_db(conn, chunks)


def test_full_pipeline_capture_to_graph(conn, settings):
    settings.diarization.enabled = True
    settings.extraction.enabled = True

    # 1. a recorded chunk → transcription job
    af = recorder.register_chunk(
        conn, path=settings.audio_raw_dir / "c.flac",
        started_at="2026-06-16T09:00:00.000Z", ended_at="2026-06-16T09:00:02.000Z",
        duration_s=2.0, settings=settings,
    )
    # 2. transcribe → segment + grouped into a conversation
    worker.drain(
        conn, transcriber=MockTranscriber([TranscribedSegment(0.0, 2.0, "Dana owns Atlas", 0.9)]),
        vad=MockVad(), settings=settings, max_jobs=1,
    )
    seg = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    conv = conn.execute("SELECT conversation_id FROM audio_files WHERE id=?", (af,)).fetchone()[0]
    assert conv is not None

    # 3. close + diarize the conversation (inject fake audio builder)
    conversation.close_conversation(conn, conv)
    diar = MockDiarizer(
        turns=[(0.0, 2.0, "S0")],
        embeddings={"S0": deterministic_embedding("dana", settings.diarization.embedding_dim)},
    )
    attribution.attribute_conversation(conn, conv, diarizer=diar, settings=settings,
                                       audio_builder=_fake_builder)
    assert conn.execute("SELECT speaker_id FROM transcript_segments WHERE id=?", (seg,)).fetchone()["speaker_id"]

    # 4. extract knowledge (MockLLM returns one entity + fact citing the segment)
    payload = {
        "entities": [{"type": "person", "name": "Dana", "source_segment_ids": [seg], "confidence": 0.9}],
        "facts": [{"subject_ref": 0, "predicate": "owns", "object_text": "Atlas",
                   "source_segment_ids": [seg], "confidence": 0.8}],
        "action_items": [], "decisions": [], "ideas": [],
    }
    run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]), settings=settings)

    # 5. the knowledge graph reflects the whole chain
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_nodes WHERE name='Dana'").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_edges WHERE predicate='owns'").fetchone()["n"] == 1
    assert conn.execute(
        "SELECT knowledge_status FROM conversations WHERE id=?", (conv,)
    ).fetchone()[0] == "extracted"
