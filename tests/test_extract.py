import json

from secondbrain.knowledge import extract
from secondbrain.llm.client import MockLLM
from secondbrain.speaker import registry
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def _named_speaker(conn, name):
    cur = conn.execute("INSERT INTO speakers (name, kind, display_label) VALUES (?, 'known', ?)", (name, name))
    return int(cur.lastrowid)


def _diarized_conversation(conn, segs):
    """segs: list of (text, speaker_id, speaker_conf). Returns conversation id."""
    conv = conn.execute(
        "INSERT INTO conversations (started_at, status, knowledge_status) "
        "VALUES ('2026-06-16T09:00:00.000Z','diarized','pending')"
    ).lastrowid
    af = models.insert_audio_file(
        conn,
        AudioFile(path="/tmp/c.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000,
                  duration_s=10.0, status="transcribed"),
    )
    conn.execute("UPDATE audio_files SET conversation_id=? WHERE id=?", (conv, af))
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    seg_ids = []
    for i, (text, spk, conf) in enumerate(segs):
        models.insert_segments(
            conn,
            [Segment(tid, af, float(i), float(i) + 1, text,
                     start_at=f"2026-06-16T09:00:0{i}.000Z", speaker_id=spk, confidence=0.9)],
        )
        sid = conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"]
        conn.execute("UPDATE transcript_segments SET speaker_confidence=? WHERE id=?", (conf, sid))
        seg_ids.append(sid)
    return conv, seg_ids


def test_extraction_writes_nodes_edges_with_provenance(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    dana = _named_speaker(conn, "Dana")
    conv, seg_ids = _diarized_conversation(
        conn, [("I'll loop in Dana.", owner, 0.95), ("I'll send the report Friday.", dana, 0.95)]
    )
    payload = {
        "entities": [{"type": "person", "name": "Dana", "source_segment_ids": [seg_ids[1]], "confidence": 0.9}],
        "facts": [{"subject_ref": 0, "predicate": "works_on", "object_text": "Atlas",
                   "source_segment_ids": [seg_ids[1]], "confidence": 0.8}],
        "action_items": [{"owed_by_ref": 0, "description": "send the report", "due_date": "2026-06-20",
                          "source_segment_ids": [seg_ids[1]], "confidence": 0.8}],
        "decisions": [], "ideas": [],
    }
    llm = MockLLM(responses=[json.dumps(payload)])
    n = extract.run_extraction(conn, conv, llm=llm, settings=settings)
    assert n == 2  # one fact + one action item

    node = conn.execute("SELECT * FROM kg_nodes WHERE type='person' AND name='Dana'").fetchone()
    assert node is not None and node["speaker_id"] == dana  # Person bound to the voice
    fact = conn.execute("SELECT * FROM kg_edges WHERE kind='fact'").fetchone()
    assert fact["predicate"] == "works_on" and json.loads(fact["source_segment_ids"]) == [seg_ids[1]]
    assert conn.execute("SELECT COUNT(*) AS n FROM knowledge_extractions").fetchone()["n"] == 1
    assert conn.execute("SELECT knowledge_status FROM conversations WHERE id=?", (conv,)).fetchone()[0] == "extracted"


def test_low_confidence_attribution_downgrades_to_mention(conn, settings):
    settings.diarization.low_confidence_threshold = 0.5
    dana = _named_speaker(conn, "Dana")
    conv, seg_ids = _diarized_conversation(conn, [("mumbled something", dana, 0.2)])  # low conf
    payload = {
        "entities": [{"type": "person", "name": "Dana", "source_segment_ids": [seg_ids[0]]}],
        "facts": [{"subject_ref": 0, "predicate": "promised", "object_text": "a raise",
                   "source_segment_ids": [seg_ids[0]], "confidence": 0.9}],
        "action_items": [], "decisions": [], "ideas": [],
    }
    extract.run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]), settings=settings)
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM kg_edges").fetchall()]
    assert kinds == ["mention"]  # not asserted as a hard fact


def test_hallucinated_citations_dropped(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    conv, seg_ids = _diarized_conversation(conn, [("real line", owner, 0.95)])
    payload = {
        "entities": [],
        "facts": [{"subject_ref": -1, "predicate": "likes", "object_text": "coffee",
                   "source_segment_ids": [seg_ids[0], 99999], "confidence": 0.9}],
        "action_items": [], "decisions": [], "ideas": [],
    }
    extract.run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]), settings=settings)
    fact = conn.execute("SELECT source_segment_ids FROM kg_edges WHERE kind='fact'").fetchone()
    assert json.loads(fact["source_segment_ids"]) == [seg_ids[0]]  # fake id 99999 dropped


def test_chunk_write_is_atomic_on_failure(conn, settings, monkeypatch):
    """If an edge write fails mid-chunk, that chunk's nodes/edges roll back."""
    owner = registry.get_or_create_owner(conn, "Me")
    conv, seg_ids = _diarized_conversation(conn, [("I'll loop in Dana.", owner, 0.95)])
    payload = {
        "entities": [{"type": "person", "name": "Dana", "source_segment_ids": [seg_ids[0]]}],
        "facts": [{"subject_ref": 0, "predicate": "works_on", "object_text": "Atlas",
                   "source_segment_ids": [seg_ids[0]], "confidence": 0.8}],
        "action_items": [], "decisions": [], "ideas": [],
    }
    # Fail on the edge write, after the extraction record + entity node were written.
    monkeypatch.setattr(
        extract.graph, "upsert_edge",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    import pytest
    with pytest.raises(RuntimeError):
        extract.run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]), settings=settings)
    # The whole chunk rolled back: no extraction record, no node, no edge.
    assert conn.execute("SELECT COUNT(*) AS n FROM knowledge_extractions").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_nodes WHERE name='Dana'").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_edges").fetchone()["n"] == 0


def test_redacted_and_optout_segments_excluded(conn, settings):
    settings.consent.speaker_opt_out = ["Private"]
    private = _named_speaker(conn, "Private")
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=?", (private,))
    conv, seg_ids = _diarized_conversation(conn, [("secret stuff", private, 0.95)])
    # MockLLM default returns "{}" for schema → no entities; main point: no crash, no rows
    n = extract.run_extraction(conn, conv, llm=MockLLM(), settings=settings)
    assert n == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_nodes").fetchone()["n"] == 0


def test_decision_subject_prefers_non_person_entity(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    conv, seg_ids = _diarized_conversation(
        conn, [("We decided to ship Atlas in Q3.", owner, 0.95)]
    )
    payload = {
        "entities": [
            {"type": "person", "name": "Dana", "source_segment_ids": seg_ids},
            {"type": "project", "name": "Atlas", "source_segment_ids": seg_ids},
        ],
        "facts": [], "action_items": [],
        "decisions": [{"summary": "ship Atlas in Q3", "participant_refs": [0, 1],
                       "source_segment_ids": seg_ids, "confidence": 0.9}],
        "ideas": [],
    }
    extract.run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]),
                           settings=settings)
    edge = conn.execute("SELECT * FROM kg_edges WHERE kind='decision'").fetchone()
    subject = conn.execute(
        "SELECT name, type FROM kg_nodes WHERE id=?", (edge["src_node_id"],)
    ).fetchone()
    # The project (what the decision is about) wins over the person listed first.
    assert subject["name"] == "Atlas" and subject["type"] == "project"


def test_decision_falls_back_to_first_participant_then_owner(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    conv, seg_ids = _diarized_conversation(conn, [("we decided things", owner, 0.95)])
    payload = {
        "entities": [{"type": "person", "name": "Dana", "source_segment_ids": seg_ids}],
        "facts": [], "action_items": [],
        "decisions": [
            {"summary": "people-only decision", "participant_refs": [0],
             "source_segment_ids": seg_ids, "confidence": 0.9},
            {"summary": "no-participant decision", "participant_refs": [],
             "source_segment_ids": seg_ids, "confidence": 0.9},
        ],
        "ideas": [],
    }
    extract.run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]),
                           settings=settings)
    rows = conn.execute(
        "SELECT e.object_text, n.name FROM kg_edges e JOIN kg_nodes n "
        "ON n.id=e.src_node_id WHERE e.kind='decision' ORDER BY e.id"
    ).fetchall()
    by_text = {r["object_text"]: r["name"] for r in rows}
    assert by_text["people-only decision"] == "Dana"   # first participant
    assert by_text["no-participant decision"] == "Me"  # owner fallback


def test_failed_chunk_is_isolated_and_rest_continue(conn, settings, monkeypatch):
    owner = registry.get_or_create_owner(conn, "Me")
    # Force two chunks: tiny context budget with two segments.
    settings.extraction.max_context_chars = 60
    settings.extraction.overlap_segments = 0
    conv, seg_ids = _diarized_conversation(
        conn, [("first chunk line here", owner, 0.95), ("second chunk line here", owner, 0.95)]
    )
    good = {
        "entities": [], "facts": [{"subject_ref": -1, "predicate": "said",
                                   "object_text": "ok", "source_segment_ids": [seg_ids[1]],
                                   "confidence": 0.9}],
        "action_items": [], "decisions": [], "ideas": [],
    }
    # Chunk 1: garbage twice (initial + reprompt) → skipped with a warning.
    # Chunk 2: valid → still lands.
    llm = MockLLM(responses=["not json at all", "still not json", json.dumps(good)])
    n = extract.run_extraction(conn, conv, llm=llm, settings=settings)
    assert n == 1
    assert conn.execute(
        "SELECT knowledge_status FROM conversations WHERE id=?", (conv,)
    ).fetchone()[0] == "extracted"


def test_all_chunks_failing_fails_the_job(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    conv, _ = _diarized_conversation(conn, [("only line", owner, 0.95)])
    llm = MockLLM(responses=["garbage", "more garbage"])  # initial + reprompt
    import pytest
    with pytest.raises(RuntimeError, match="all 1 chunk"):
        extract.run_extraction(conn, conv, llm=llm, settings=settings)


def test_bad_json_chunk_recovers_via_reprompt(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    conv, seg_ids = _diarized_conversation(conn, [("a line", owner, 0.95)])
    good = {
        "entities": [], "facts": [{"subject_ref": -1, "predicate": "said",
                                   "object_text": "ok", "source_segment_ids": seg_ids,
                                   "confidence": 0.9}],
        "action_items": [], "decisions": [], "ideas": [],
    }
    llm = MockLLM(responses=["not json", json.dumps(good)])  # retry succeeds
    assert extract.run_extraction(conn, conv, llm=llm, settings=settings) == 1


def test_speaker_hint_matches_unique_first_name_prefix(conn, settings):
    dana = _named_speaker(conn, "Dana Whitfield")
    assert extract._speaker_hint(conn, "Dana") == dana
    assert extract._speaker_hint(conn, "dana whitfield") == dana  # exact still wins
    # A second Dana makes the bare first name ambiguous → no link.
    _named_speaker(conn, "Dana Smith")
    assert extract._speaker_hint(conn, "Dana") is None
    assert extract._speaker_hint(conn, "Nobody") is None
    assert extract._speaker_hint(conn, "") is None


def test_unresolvable_owed_by_marks_needs_review_not_owner_default(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    conv, seg_ids = _diarized_conversation(conn, [("someone will send it", owner, 0.95)])
    payload = {
        "entities": [],
        "facts": [],
        "action_items": [
            # owed_by_ref points at a non-existent entity index → unresolvable
            {"owed_by_ref": 5, "description": "send the report",
             "source_segment_ids": seg_ids, "confidence": 0.8},
            # owed_by_ref omitted entirely → owner by design, no review flag
            {"description": "book the room", "source_segment_ids": seg_ids,
             "confidence": 0.8},
        ],
        "decisions": [], "ideas": [],
    }
    extract.run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]),
                           settings=settings)
    rows = {r["object_text"]: r for r in conn.execute(
        "SELECT object_text, predicate, confidence FROM kg_edges WHERE kind='action_item'"
    ).fetchall()}
    flagged = rows["send the report"]
    assert flagged["predicate"] == extract.NEEDS_REVIEW_PREDICATE
    assert flagged["confidence"] == 0.4  # halved
    plain = rows["book the room"]
    assert plain["predicate"] == "action_item" and plain["confidence"] == 0.8


def test_normalize_due_date_parses_common_forms():
    from datetime import date

    ref = date(2026, 6, 16)
    f = extract.normalize_due_date
    assert f("2026-06-20", ref) == "2026-06-20"
    assert f("March 3", ref) == "2027-03-03"        # already past → next year
    assert f("July 3rd", ref) == "2026-07-03"
    assert f("Mar 3, 2026", ref) == "2026-03-03"
    assert f("3/14", ref) == "2027-03-14"
    assert f("7/14/2026", ref) == "2026-07-14"
    assert f("14 July", ref) == "2026-07-14"
    # unparseable / invalid stays None — conservative
    assert f("next week", ref) is None
    assert f("2026-13-40", ref) is None
    assert f("", ref) is None
    assert f(None, ref) is None


def test_due_date_norm_written_alongside_raw(conn, settings):
    owner = registry.get_or_create_owner(conn, "Me")
    conv, seg_ids = _diarized_conversation(
        conn, [("I'll send it by July 3rd.", owner, 0.95)]
    )
    payload = {
        "entities": [], "facts": [],
        "action_items": [{"owed_by_ref": -1, "description": "send it",
                          "due_date": "July 3rd", "source_segment_ids": seg_ids,
                          "confidence": 0.8}],
        "decisions": [], "ideas": [],
    }
    extract.run_extraction(conn, conv, llm=MockLLM(responses=[json.dumps(payload)]),
                           settings=settings)
    row = conn.execute(
        "SELECT due_date, due_date_norm FROM kg_edges WHERE kind='action_item'"
    ).fetchone()
    assert row["due_date"] == "July 3rd"          # raw string kept
    assert row["due_date_norm"] == "2026-07-03"   # conversation was 2026-06-16
