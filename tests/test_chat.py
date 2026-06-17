from secondbrain.knowledge import chat, graph
from secondbrain.llm.client import MockLLM
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def _seg(conn, text):
    af = models.insert_audio_file(
        conn, AudioFile(path="/tmp/a.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 2.0, text, start_at="2026-06-16T09:00:00.000Z")])
    return conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"]


def test_answer_resolves_citations_and_flags_general(conn, settings):
    seg = _seg(conn, "we agreed to raise pricing next quarter")
    node = graph.create_node(conn, type="topic", name="pricing", embedding=None,
                             confidence=0.9, extraction_id=None)
    graph.upsert_edge(conn, src_node_id=node, dst_node_id=None, predicate="decided",
                      kind="decision", object_text="raise pricing", source_segment_ids=[seg])
    answer_text = (
        f"You decided to raise pricing next quarter [{seg}]. "
        "(general knowledge — not from your data) Pricing strategy varies by market."
    )
    result = chat.answer(conn, "what about pricing?", llm=MockLLM(responses=[answer_text]), settings=settings)
    assert result["grounded"] is True
    assert result["general_used"] is True
    assert any(c["segment_id"] == seg for c in result["citations"])


def test_answer_without_citations_is_not_grounded(conn, settings):
    _seg(conn, "unrelated chatter about lunch")
    result = chat.answer(conn, "quarterly revenue?", llm=MockLLM(responses=["I don't have that."]),
                         settings=settings)
    assert result["grounded"] is False
    assert result["citations"] == []
