"""Person dossier — aggregates identity, interactions, facts, commitments, quotes."""

from __future__ import annotations

from secondbrain.query import service


def _audio(conn, aid, conv, day, path="/tmp/a.flac"):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (?, ?, 'diarized')",
        (conv, f"{day}T09:00:00.000Z"),
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (?, ?, ?, 16000, 'transcribed', ?)",
        (aid, f"{path}{aid}", f"{day}T09:00:00.000Z", conv),
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')", (aid, aid)
    )


def _seg(conn, sid, aid, day, sec, text, speaker_id, conf=0.95):
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
        " speaker_id, speaker_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, aid, aid, sec, sec + 2.0, f"{day}T09:00:{sec:02d}.000Z", text, speaker_id, conf),
    )


def _seed(conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Dana', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (3, 'Sam', 'known', 0)")
    _audio(conn, 1, 1, "2026-06-16")
    _seg(conn, 1, 1, "2026-06-16", 0, "me talking", 1)
    _seg(conn, 2, 1, "2026-06-16", 2, "dana talking here", 2)
    _seg(conn, 3, 1, "2026-06-16", 4, "sam talking too", 3)
    # person nodes for Dana + a project; facts + action items
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (1, 'person', 'Dana', 2)"
    )
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (2, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_aliases (node_id, alias) VALUES (1, 'Dana Smith')"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        " confidence, valid, conversation_id, source_segment_ids) "
        "VALUES (1, 1, 2, 'works_on', 'fact', 'Atlas', 0.9, 1, 1, '[2]')"
    )
    # Dana owes something; something owed to Dana
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, due_date, valid, "
        " conversation_id) "
        "VALUES (2, 1, 'action_item', 'send the deck', '2026-06-20', 1, 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, valid) "
        "VALUES (3, 2, 1, 'action_item', 'review by Dana', 1)"
    )


def test_dossier_aggregates(conn, settings):
    _seed(conn)
    d = service.person_dossier(conn, 2, settings)
    assert d["label"] == "Dana"
    assert d["node_id"] == 1
    assert "Dana Smith" in d["aliases"]
    assert d["interactions"]["segments"] == 1
    assert d["interactions"]["conversations"] == 1
    assert d["interactions"]["talk_minutes"] == round(2.0 / 60, 1)
    assert any(f["object_text"] == "Atlas" for f in d["facts"])
    assert any(c["object_text"] == "send the deck" for c in d["commitments"]["owed_by"])
    assert any(c["object_text"] == "review by Dana" for c in d["commitments"]["owed_to"])
    quotes = [q["text"] for q in d["recent_quotes"]]
    assert "dana talking here" in quotes


def test_dossier_source_links(conn, settings):
    """Facts/commitments carry provenance: parsed segment ids + local day + anchor."""
    _seed(conn)
    d = service.person_dossier(conn, 2, settings)
    fact = next(f for f in d["facts"] if f["object_text"] == "Atlas")
    assert fact["conversation_id"] == 1
    assert fact["source_segment_ids"] == [2]  # JSON string parsed to ints
    assert fact["source_seg"] == 2
    assert fact["source_day"] == service._local_day_of("2026-06-16T09:00:02.000Z")
    # no cited segments -> falls back to the conversation's local day, no anchor
    owed = next(c for c in d["commitments"]["owed_by"] if c["object_text"] == "send the deck")
    assert owed["source_seg"] is None
    assert owed["source_day"] == service._local_day_of("2026-06-16T09:00:00.000Z")
    # no provenance at all -> both stay None (template hides the source link)
    owed_to = next(c for c in d["commitments"]["owed_to"] if c["object_text"] == "review by Dana")
    assert owed_to["source_seg"] is None and owed_to["source_day"] is None


def test_dossier_malformed_source_segment_ids(conn, settings):
    _seed(conn)
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, valid, conversation_id, "
        " source_segment_ids) VALUES (9, 1, 'fact', 'likes tea', 1, 1, 'not-json')"
    )
    d = service.person_dossier(conn, 2, settings)
    bad = next(f for f in d["facts"] if f["object_text"] == "likes tea")
    assert bad["source_segment_ids"] == []  # malformed JSON degrades, never raises
    assert bad["source_day"] == service._local_day_of("2026-06-16T09:00:00.000Z")


def test_dossier_recent_conversations_and_quote_days(conn, settings):
    _seed(conn)
    _audio(conn, 2, 2, "2026-06-17", path="/tmp/b.flac")
    _seg(conn, 4, 2, "2026-06-17", 0, "dana again", 2)
    d = service.person_dossier(conn, 2, settings)
    convs = d["recent_conversations"]
    assert [c["conversation_id"] for c in convs] == [2, 1]  # newest first
    assert convs[0]["segments"] == 1
    assert convs[0]["anchor_segment_id"] == 4  # her earliest line there
    assert convs[0]["day"] == service._local_day_of("2026-06-17T09:00:00.000Z")
    # quotes carry the same local-day bucketing plus their conversation
    q = d["recent_quotes"][0]
    assert q["text"] == "dana again"
    assert q["day"] == service._local_day_of("2026-06-17T09:00:00.000Z")
    assert q["conversation_id"] == 2


def test_dossier_connections(conn, settings):
    _seed(conn)
    d = service.person_dossier(conn, 2, settings)
    labels = {c["label"] for c in d["connections"]}
    assert "Me" in labels and "Sam" in labels
    assert 2 not in {c["speaker_id"] for c in d["connections"]}  # not self


def test_dossier_owner(conn, settings):
    _seed(conn)
    d = service.person_dossier(conn, 1, settings)
    assert d["is_owner"] is True
    assert d["label"] == "Me"


def test_dossier_owner_stored_name_wins(conn, settings):
    """A renamed owner shows the stored name; 'Me' is only the default."""
    _seed(conn)
    conn.execute("UPDATE speakers SET name='George' WHERE id=1")
    d = service.person_dossier(conn, 1, settings)
    assert d["label"] == "George" and d["is_owner"] is True
    assert d["name"] == "George"  # raw stored name (rename-form prefill source)
    # connection pills on other dossiers agree with the owner's page
    dana = service.person_dossier(conn, 2, settings)
    assert "George" in {c["label"] for c in dana["connections"]}
    # no stored name still reads "Me"
    conn.execute("UPDATE speakers SET name=NULL WHERE id=1")
    d2 = service.person_dossier(conn, 1, settings)
    assert d2["label"] == "Me" and d2["name"] is None


def test_dossier_talk_label(conn, settings):
    """Human talk-time label; None when nothing was heard (UI shows 0)."""
    _seed(conn)
    d = service.person_dossier(conn, 2, settings)  # Dana spoke for 2 s
    assert d["interactions"]["talk_label"] == "under 1 min"
    conn.execute("INSERT INTO speakers (id, name, kind) VALUES (20, 'Silent', 'known')")
    silent = service.person_dossier(conn, 20, settings)
    assert silent["interactions"]["talk_label"] is None
    assert silent["interactions"]["talk_minutes"] == 0.0  # original field intact


def test_dossier_commitments_dated_before_undated(conn, settings):
    """Dated commitments come first (soonest on top); undated trail behind."""
    _seed(conn)  # edge 2: 'send the deck' due 2026-06-20
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, valid) "
        "VALUES (20, 1, 'action_item', 'undated errand', 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, due_date, valid) "
        "VALUES (21, 1, 'action_item', 'urgent thing', '2026-06-18', 1)"
    )
    d = service.person_dossier(conn, 2, settings)
    texts = [c["object_text"] for c in d["commitments"]["owed_by"]]
    assert texts == ["urgent thing", "send the deck", "undated errand"]


def test_dossier_facts_capped_with_total(conn, settings):
    """facts_total reports the full count while the list keeps the top-25."""
    _seed(conn)  # one fact at 0.9 confidence
    for i in range(30):
        conn.execute(
            "INSERT INTO kg_edges (id, src_node_id, kind, object_text, confidence, valid) "
            "VALUES (?, 1, 'fact', ?, ?, 1)",
            (100 + i, f"fact {i}", 0.5 + i / 100),
        )
    d = service.person_dossier(conn, 2, settings)
    assert d["facts_total"] == 31
    assert len(d["facts"]) == service._DOSSIER_FACTS_LIMIT == 25
    confs = [f["confidence"] for f in d["facts"]]
    assert confs == sorted(confs, reverse=True) and confs[0] == 0.9  # keep the most confident


def test_dossier_mentions_capped_with_total(conn, settings):
    _seed(conn)
    for i in range(15):
        conn.execute(
            "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, valid, "
            "conversation_id) VALUES (?, 2, 1, 'mention', ?, 1, 1)",
            (200 + i, f"mention {i}"),
        )
    d = service.person_dossier(conn, 2, settings)
    assert d["mentions_total"] == 15
    assert len(d["mentions"]) == 12  # list itself stays capped
    # the privacy gate zeroes the totals along with the lists
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    d2 = service.person_dossier(conn, 2, settings)
    assert d2["mentions_total"] == 0 and d2["facts_total"] == 0


def test_dossier_opted_out_hides_content(conn, settings):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    d = service.person_dossier(conn, 2, settings)
    assert d["opted_out"] is True
    assert d["facts"] == []
    assert d["recent_quotes"] == []
    assert d["recent_conversations"] == []
    assert d["commitments"] == {"owed_by": [], "owed_to": []}
    assert d["mentions"] == []
    # identity/interaction shape still present
    assert d["interactions"]["segments"] == 1


def test_dossier_dst_side_facts(conn, settings):
    """Facts naming the person as the object surface too, flagged 'referenced'."""
    _seed(conn)
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        " confidence, valid, conversation_id, source_segment_ids) "
        "VALUES (10, 2, 1, 'led_by', 'fact', 'Dana', 0.95, 1, 1, '[2]')"
    )
    d = service.person_dossier(conn, 2, settings)
    about = next(f for f in d["facts"] if f["object_text"] == "Atlas")
    assert about["direction"] == "about" and about["other_label"] is None
    ref = next(f for f in d["facts"] if f["direction"] == "referenced")
    assert ref["predicate"] == "led_by"
    assert ref["other_label"] == "Atlas"  # the node the fact belongs to
    assert ref["source_seg"] == 2  # provenance parsed exactly like src-side facts
    # both directions rank together by confidence
    confs = [f["confidence"] for f in d["facts"]]
    assert confs == sorted(confs, reverse=True) and confs[0] == 0.95


def test_dossier_mentions(conn, settings):
    """mention/decision/idea edges touching the person land in 'mentions' with quotes."""
    _seed(conn)
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, confidence, "
        " valid, conversation_id, source_segment_ids) "
        "VALUES (11, 2, 1, 'decision', 'Dana leads QA for Atlas', 0.9, 1, 1, '[2]')"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, predicate, valid, "
        " conversation_id) VALUES (12, 1, 2, 'mention', 'mentions', 1, 1)"
    )
    d = service.person_dossier(conn, 2, settings)
    assert {m["kind"] for m in d["mentions"]} == {"decision", "mention"}
    dec = next(m for m in d["mentions"] if m["kind"] == "decision")
    assert dec["other_label"] == "Atlas"
    assert dec["source_seg"] == 2  # source link provenance
    assert dec["quotes"][0]["segment_id"] == 2  # the cited line, ready to quote
    assert dec["quotes"][0]["day"] == service._local_day_of("2026-06-16T09:00:02.000Z")
    men = next(m for m in d["mentions"] if m["kind"] == "mention")
    assert men["quotes"] == []  # no citations -> no quotes
    assert men["source_day"] is not None  # falls back to the conversation's day


def test_dossier_mention_quotes_respect_optout(conn, settings):
    _seed(conn)
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, valid, "
        " conversation_id, source_segment_ids) "
        "VALUES (11, 2, 1, 'decision', 'ship it', 1, 1, '[3]')"  # cites Sam's line
    )
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=3")
    d = service.person_dossier(conn, 2, settings)
    dec = next(m for m in d["mentions"] if m["kind"] == "decision")
    assert dec["quotes"] == []  # Sam opted out; his words stay private


def test_dossier_commitments_carry_edge_and_task_ids(conn, settings):
    _seed(conn)
    d = service.person_dossier(conn, 2, settings)
    owed = next(c for c in d["commitments"]["owed_by"] if c["object_text"] == "send the deck")
    assert owed["id"] == 2 and owed["task_id"] is None
    tid = service.promote_action_item(conn, 2)
    d2 = service.person_dossier(conn, 2, settings)
    owed2 = next(c for c in d2["commitments"]["owed_by"] if c["object_text"] == "send the deck")
    assert owed2["task_id"] == tid  # already tracked -> the UI shows its done state
    owed_to = next(c for c in d2["commitments"]["owed_to"] if c["object_text"] == "review by Dana")
    assert owed_to["id"] == 3 and owed_to["task_id"] is None


def test_dossier_name_duplicates(conn, settings):
    _seed(conn)
    conn.execute("INSERT INTO speakers (id, name, kind) VALUES (12, 'dana', 'known')")
    conn.execute("INSERT INTO speakers (id, name, kind, merged_into) VALUES (13, 'Dana', 'known', 2)")
    d = service.person_dossier(conn, 2, settings)
    assert {x["speaker_id"] for x in d["name_duplicates"]} == {12}  # case-insensitive, unmerged only
    # unknowns (no name yet) never report duplicates
    conn.execute("INSERT INTO speakers (id, kind, display_label) VALUES (14, 'unknown', 'U1')")
    assert service.person_dossier(conn, 14, settings)["name_duplicates"] == []


def test_dossier_resolves_merged_speaker(conn, settings):
    _seed(conn)
    conn.execute("INSERT INTO speakers (id, name, kind, merged_into) VALUES (9, 'Dana?', 'unknown', 2)")
    d = service.person_dossier(conn, 9, settings)
    assert d["speaker_id"] == 2
    assert d["label"] == "Dana"


def test_dossier_missing_speaker_returns_none(conn, settings):
    assert service.person_dossier(conn, 999, settings) is None
