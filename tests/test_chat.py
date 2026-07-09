from datetime import UTC, datetime

from secondbrain.knowledge import chat, graph
from secondbrain.llm.client import MockLLM
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def _seg(conn, text, start_at="2026-06-16T09:00:00.000Z"):
    af = models.insert_audio_file(
        conn, AudioFile(path="/tmp/a.flac", started_at=start_at, sample_rate=16000)
    )
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 2.0, text, start_at=start_at)])
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


def test_answer_threads_history_into_prompt(conn, settings):
    _seg(conn, "we agreed to raise pricing next quarter")
    llm = MockLLM(by_substring={"Previous conversation": "follow-up seen"})
    result = chat.answer(
        conn,
        "and when?",
        llm=llm,
        settings=settings,
        history=[{"question": "what about pricing?", "answer": "You raise pricing."}],
    )
    assert result["answer"] == "follow-up seen"  # prompt contained the prior turn


def test_answer_resolves_citations_from_history(conn, settings):
    # A follow-up ("when is that due?") retrieves nothing by itself, but the
    # previous answer cited a segment — re-citing it must still resolve.
    seg = _seg(conn, "we agreed to raise pricing next quarter")
    llm = MockLLM(responses=[f"Next quarter [{seg}]."])
    result = chat.answer(
        conn,
        "zzz nothing matches this zzz",
        llm=llm,
        settings=settings,
        history=[{"question": "pricing?", "answer": f"Raise pricing [{seg}]."}],
    )
    assert result["grounded"] is True
    assert [c["segment_id"] for c in result["citations"]] == [seg]


def test_answer_ignores_malformed_history(conn, settings):
    _seg(conn, "we agreed to raise pricing next quarter")
    result = chat.answer(
        conn,
        "what about pricing?",
        llm=MockLLM(responses=["ok"]),
        settings=settings,
        history=[{"question": "", "answer": ""}, {"nope": 1}],
    )
    assert result["answer"] == "ok"


def test_temporal_window_parsing():
    now = datetime(2026, 7, 2, 15, 0).astimezone()  # a Thursday
    w = chat._temporal_window("What did I talk about today?", now)
    assert w == {"label": "today", "start_day": "2026-07-02", "end_day": "2026-07-02"}
    w = chat._temporal_window("what happened YESTERDAY", now)
    assert (w["start_day"], w["end_day"]) == ("2026-07-01", "2026-07-01")
    w = chat._temporal_window("summarise this week", now)
    assert (w["start_day"], w["end_day"]) == ("2026-06-29", "2026-07-02")
    w = chat._temporal_window("plans from the last 3 days", now)
    assert (w["start_day"], w["end_day"]) == ("2026-06-30", "2026-07-02")
    w = chat._temporal_window("summarise my recent conversations", now)
    assert w["label"] == "the last 7 days"
    assert chat._temporal_window("when is the pricing review?", now) is None


def test_temporal_window_absolute_dates():
    now = datetime(2026, 7, 9, 12, 0).astimezone()  # today = Thu 2026-07-09
    single = ("2026-07-02", "2026-07-02")

    # ISO, month-name (both orders, with/without ordinal + year), and slashed M/D/Y
    for q in (
        "What did I talk about on 2026-07-02?",
        "on July 2nd",
        "notes from July 2",  # no year → most recent past-or-today occurrence
        "anything from Jul 2 2026",
        "what about on 2 July 2026",
        "recap of 07/02/2026",
    ):
        w = chat._temporal_window(q, now)
        assert w is not None, q
        assert (w["start_day"], w["end_day"]) == single, q  # a single day
    # an explicit past date carries an absolute label, not a relative one
    assert chat._temporal_window("on July 2nd", now)["label"] == "Jul 2, 2026"
    # a bare month/day with no year that hasn't happened yet resolves to last year
    assert chat._temporal_window("summary for December 25", now)["start_day"] == "2025-12-25"
    # explicit dates equal to today/yesterday read the way the user would say them
    assert chat._temporal_window("what did I say on July 9", now)["label"] == "today"
    assert chat._temporal_window("on July 8", now)["label"] == "yesterday"
    # explicit dates beat a stray relative word ("recently") and pin the day
    assert chat._temporal_window("recently, on 2026-07-02", now)["start_day"] == "2026-07-02"
    # impossible or incomplete dates don't produce a bogus window
    assert chat._temporal_window("Feb 30 plans", now) is None
    assert chat._temporal_window("June 31st", now) is None
    assert chat._temporal_window("what about July", now) is None  # bare month, no day
    assert chat._temporal_window("call me at 555-1234", now) is None  # not a date


def test_absolute_date_question_finds_the_day(conn, settings):
    # A question that names a date but shares no searchable tokens with the
    # segment must still surface that day's lines via the date-window merge —
    # the regression the 'no data on July 2nd' bug came from.
    from datetime import datetime as _dt

    from secondbrain.llm.client import MockLLM

    # Anchor the segment to local noon today, then ask about that local day by
    # its explicit ISO date — timezone-robust (no UTC/local off-by-one).
    now_local = _dt.now().astimezone()
    day = now_local.strftime("%Y-%m-%d")
    noon_utc = now_local.replace(hour=12, minute=0, second=0, microsecond=0).astimezone(
        UTC
    ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    seg = _seg(conn, "we agreed to raise pricing next quarter", start_at=noon_utc)
    llm = MockLLM(by_substring={"raise pricing next quarter": f"On that day [{seg}]."},
                  default="missed")
    result = chat.answer(conn, f"What did I talk about on {day}?", llm=llm, settings=settings)
    assert result["answer"] == f"On that day [{seg}]."
    assert result["grounded"] is True
    assert result["time_window"]["start_day"] == result["time_window"]["end_day"] == day
    assert result["time_window"]["segment_count"] == 1


def test_temporal_question_pulls_days_segments_into_context(conn, settings):
    # The question shares no searchable tokens with the segment, so only the
    # date-window merge can bring it into the prompt.
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    _seg(conn, "we agreed to raise pricing next quarter", start_at=now_iso)
    llm = MockLLM(by_substring={"raise pricing next quarter": "windowed"}, default="missed")
    result = chat.answer(conn, "What did I talk about today?", llm=llm, settings=settings)
    assert result["answer"] == "windowed"
    assert result["time_window"]["label"] == "today"
    assert result["time_window"]["segment_count"] == 1


def test_temporal_question_with_empty_window_tells_model(conn, settings):
    _seg(conn, "old chatter about lunch")  # 2026-06-16: far outside "today"
    llm = MockLLM(by_substring={"nothing was captured in that period": "empty-day"},
                  default="missed")
    result = chat.answer(conn, "What did I talk about today?", llm=llm, settings=settings)
    assert result["answer"] == "empty-day"
    assert result["time_window"]["segment_count"] == 0


def test_non_temporal_answer_has_no_time_window(conn, settings):
    _seg(conn, "we agreed to raise pricing next quarter")
    result = chat.answer(conn, "what about pricing?", llm=MockLLM(responses=["ok"]),
                         settings=settings)
    assert result["time_window"] is None


def test_prepare_finalize_roundtrip_matches_answer(conn, settings):
    seg = _seg(conn, "we agreed to raise pricing next quarter")
    prep = chat.prepare(conn, "what about pricing?", settings=settings)
    assert seg in prep.info  # retrieval surfaced the segment for citation
    result = chat.finalize(prep, f"Raise pricing [{seg}].")
    assert result["grounded"] is True
    assert result["citations"][0]["segment_id"] == seg


def test_seed_nodes_matches_edges_and_names(conn, settings):
    seg = _seg(conn, "atlas kickoff meeting")
    node = graph.create_node(conn, type="project", name="Atlas", embedding=None,
                             confidence=0.9, extraction_id=None)
    other = graph.create_node(conn, type="topic", name="Pricing", embedding=None,
                              confidence=0.9, extraction_id=None)
    graph.upsert_edge(conn, src_node_id=node, dst_node_id=None, predicate="kicked off",
                      kind="fact", object_text="kickoff", source_segment_ids=[seg])
    # via cited segment (json_each path) and via name-in-question (instr path)
    assert set(chat._seed_nodes(conn, [seg], "when is the pricing review?")) == {node, other}
    # no inputs -> no seeds (and no full scans)
    assert chat._seed_nodes(conn, [], "") == []
