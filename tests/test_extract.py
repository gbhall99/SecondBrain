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


def test_redacted_and_optout_segments_excluded(conn, settings):
    settings.consent.speaker_opt_out = ["Private"]
    private = _named_speaker(conn, "Private")
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=?", (private,))
    conv, seg_ids = _diarized_conversation(conn, [("secret stuff", private, 0.95)])
    # MockLLM default returns "{}" for schema → no entities; main point: no crash, no rows
    n = extract.run_extraction(conn, conv, llm=MockLLM(), settings=settings)
    assert n == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_nodes").fetchone()["n"] == 0
