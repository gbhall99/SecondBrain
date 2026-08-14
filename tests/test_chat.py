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


def _conv_seg(conn, texts, conv_id=7, start="2026-06-16T09:00:0{i}.000Z", speaker_id=None):
    """A conversation of consecutive segments; returns their ids in order."""
    conn.execute(
        "INSERT OR IGNORE INTO conversations (id, started_at, status) "
        "VALUES (?, '2026-06-16T09:00:00.000Z', 'diarized')",
        (conv_id,),
    )
    af = models.insert_audio_file(
        conn, AudioFile(path=f"/tmp/conv{conv_id}.flac",
                        started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    conn.execute("UPDATE audio_files SET conversation_id=? WHERE id=?", (conv_id, af))
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    ids = []
    for i, text in enumerate(texts):
        models.insert_segments(
            conn,
            [Segment(tid, af, float(i), i + 1.0, text,
                     start_at=f"2026-06-16T09:00:{i:02d}.000Z", speaker_id=speaker_id)],
        )
        ids.append(conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"])
    return ids


def test_hits_expand_with_neighboring_turns(conn, settings):
    ids = _conv_seg(conn, [
        "how was the offsite",
        "pretty good overall",
        "we agreed to raise pricing next quarter",
        "makes sense to me",
        "let's tell the team on friday",
        "sounds good",
    ])
    prep = chat.prepare(conn, "what about pricing?", settings=settings)
    # The matched line AND ±2 neighbors of the same conversation are citable.
    for sid in ids[0:5]:
        assert sid in prep.info, sid
    # ...in chronological order within the excerpt block.
    body = prep.prompt
    assert body.index("pretty good overall") < body.index("raise pricing")
    assert body.index("raise pricing") < body.index("tell the team on friday")


def test_excerpt_count_is_configurable(conn, settings, monkeypatch):
    _seg(conn, "we agreed to raise pricing next quarter")
    calls = {}
    from secondbrain.search import combined as comb

    real = comb.search

    def spy(conn_, q, limit=20, **kw):
        calls["limit"] = limit
        return real(conn_, q, limit, **kw)

    monkeypatch.setattr(chat.combined, "search", spy)
    settings.extraction.chat_max_excerpts = 7
    chat.prepare(conn, "pricing?", settings=settings)
    assert calls["limit"] == 7


def test_followup_merges_previous_question_into_retrieval(conn, settings, monkeypatch):
    _seg(conn, "we agreed to raise pricing next quarter")
    seen = {}

    def spy(conn_, q, limit=20, **kw):
        seen["q"] = q
        return []

    monkeypatch.setattr(chat.combined, "search", spy)
    chat.prepare(conn, "and when?", settings=settings,
                 history=[{"question": "what about pricing?", "answer": "Raise it."}])
    assert "and when?" in seen["q"] and "what about pricing?" in seen["q"]
    # No history → the question is used verbatim.
    chat.prepare(conn, "and when?", settings=settings)
    assert seen["q"] == "and when?"


def test_seed_nodes_requires_word_boundary_and_min_length(conn, settings):
    hr = graph.create_node(conn, type="topic", name="hr", embedding=None,
                           confidence=0.9, extraction_id=None)
    price = graph.create_node(conn, type="topic", name="pricing", embedding=None,
                              confidence=0.9, extraction_id=None)
    # "hr" (2 chars) must not attach to a question containing "three"; "pricing"
    # matches only on a word boundary, not inside "repricing".
    assert chat._seed_nodes(conn, [], "what are the three hr topics") == []
    assert hr not in chat._seed_nodes(conn, [], "hr said so")  # too short, ever
    assert chat._seed_nodes(conn, [], "what about repricing strategy") == []
    assert chat._seed_nodes(conn, [], "what about pricing strategy") == [price]


def test_grounding_states_context_empty_vs_uncited(conn, settings):
    # Nothing in the corpus matches → context_empty, not uncited.
    result = chat.answer(conn, "zzz nothing zzz", llm=MockLLM(responses=["No idea."]),
                         settings=settings)
    assert result["context_empty"] is True and result["uncited"] is False
    assert result["grounded"] is False
    # Context retrieved but the model cites none of it → uncited.
    _seg(conn, "we agreed to raise pricing next quarter")
    result = chat.answer(conn, "what about pricing?",
                         llm=MockLLM(responses=["You raised pricing."]), settings=settings)
    assert result["context_empty"] is False and result["uncited"] is True
    assert result["grounded"] is False
    # Cited answers are grounded, with both flags off.
    seg = conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"]
    result = chat.answer(conn, "what about pricing?",
                         llm=MockLLM(responses=[f"Raised [{seg}]."]), settings=settings)
    assert result["grounded"] is True
    assert result["context_empty"] is False and result["uncited"] is False


def test_dangling_citation_markers_are_stripped(conn, settings):
    seg = _seg(conn, "we agreed to raise pricing next quarter")
    result = chat.answer(
        conn, "what about pricing?",
        llm=MockLLM(responses=[f"Raise pricing [{seg}] and hire [99999] people."]),
        settings=settings,
    )
    assert "[99999]" not in result["answer"]
    assert f"[{seg}]" in result["answer"]
    assert result["dangling_citations"] == 1
    assert [c["segment_id"] for c in result["citations"]] == [seg]


def test_citations_carry_speaker_low_confidence(conn, settings):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    seg = _seg(conn, "we agreed to raise pricing next quarter")
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=5, speaker_confidence=0.2 WHERE id=?",
        (seg,),
    )
    result = chat.answer(conn, "what about pricing?",
                         llm=MockLLM(responses=[f"Raised [{seg}]."]), settings=settings)
    c = result["citations"][0]
    assert c["speaker"] == "Dana" and c["speaker_low_confidence"] is True


def test_fact_block_dates_decisions_and_orders_newest_first(conn, settings):
    node = graph.create_node(conn, type="project", name="Atlas", embedding=None,
                             confidence=0.9, extraction_id=None)
    old = graph.upsert_edge(conn, src_node_id=node, dst_node_id=None, predicate="decision",
                            kind="decision", object_text="ship the beta in June", confidence=0.9,
                            when="2026-06-01T09:00:00.000Z")
    graph.upsert_edge(conn, src_node_id=node, dst_node_id=None, predicate="decision",
                      kind="decision", object_text="hire a designer", confidence=0.5,
                      when="2026-07-12T09:00:00.000Z")
    facts = chat._subgraph_facts(conn, [node], settings)
    # Newest decision first despite lower confidence.
    assert [f["object_text"] for f in facts] == ["hire a designer", "ship the beta in June"]
    line = chat._fact_line(facts[0])
    assert line.startswith("- [2026-07-12]")
    # A decision that superseded an earlier one says so — and the superseded
    # one leaves the fact block entirely.
    new = graph.upsert_edge(conn, src_node_id=node, dst_node_id=None, predicate="decision",
                            kind="decision", object_text="ship the beta in July", confidence=0.9,
                            when="2026-08-01T09:00:00.000Z")
    facts = chat._subgraph_facts(conn, [node], settings)
    texts = [f["object_text"] for f in facts]
    assert "ship the beta in June" not in texts and "ship the beta in July" in texts
    marked = next(f for f in facts if f["object_text"] == "ship the beta in July")
    assert "(supersedes an earlier decision)" in chat._fact_line(marked)
    assert old != new


def test_scoped_ask_filters_by_speaker_and_dates(conn, settings):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    dana_ids = _conv_seg(conn, ["dana talks pricing today"], conv_id=8, speaker_id=5)
    other_ids = _conv_seg(conn, ["someone else talks pricing"], conv_id=9)
    prep = chat.prepare(conn, "pricing?", settings=settings, speaker_id=5)
    assert dana_ids[0] in prep.info
    assert other_ids[0] not in prep.info
    # A date window that excludes everything retrieves nothing.
    prep = chat.prepare(conn, "pricing?", settings=settings, until="2020-01-01")
    assert prep.has_context is False
