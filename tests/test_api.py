import pytest
from fastapi.testclient import TestClient

from secondbrain.query.api import create_app
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


@pytest.fixture
def client(conn, settings):
    # conn fixture has created the DB file at settings.db_path; seed a segment.
    af = models.insert_audio_file(
        conn, AudioFile(path="/a.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [Segment(t, af, 0.0, 2.0, "decided to adopt the new onboarding flow",
                 start_at="2026-06-16T09:00:00.000Z")],
    )
    return TestClient(create_app(settings))


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["segments_total"] == 1
    assert "disk_free_gb" in body


def test_stats_endpoint(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["segments"] == 1
    assert "kg_nodes" in body
    assert "goals" in body


def test_person_endpoints(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    r = client.get("/api/person/5")
    assert r.status_code == 200
    assert r.json()["label"] == "Dana"
    assert client.get("/person/5").status_code == 200  # HTML page renders
    assert client.get("/api/person/999").status_code == 404


def test_person_page_sources_and_conversations(client, conn):
    from secondbrain.query.service import _local_day_of

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (7, '2026-06-16T09:00:00.000Z', 'diarized')"
    )
    conn.execute("UPDATE audio_files SET conversation_id=7")
    conn.execute("UPDATE transcript_segments SET speaker_id=5")
    day = _local_day_of("2026-06-16T09:00:00.000Z")

    d = client.get("/api/person/5").json()
    # quotes carry provenance (additive fields; id/start_at/text unchanged)
    q = d["recent_quotes"][0]
    assert q["conversation_id"] == 7 and q["day"] == day and q["id"] == 1
    # distinct conversations ride along with anchors for deep-linking
    c = d["recent_conversations"][0]
    assert c["conversation_id"] == 7 and c["segments"] == 1
    assert c["anchor_segment_id"] == 1 and c["day"] == day

    html = client.get("/person/5").text
    # quotes and the conversation list deep-link into the day view anchor
    assert f"/day?date={day}#seg-1" in html
    assert "Recent conversations" in html
    # empty sections stay visible with helpful microcopy instead of vanishing
    assert "Known facts" in html and "No facts extracted about Dana yet" in html
    assert "Commitments" in html and "No open commitments involving Dana" in html
    # timestamps are rendered readable (localdt filter), not raw ISO-with-ms
    assert "Jun 2026," in html


def test_person_page_fact_source_links(client, conn):
    from secondbrain.query.service import _local_day_of

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (7, '2026-06-16T09:00:00.000Z', 'diarized')"
    )
    conn.execute("UPDATE audio_files SET conversation_id=7")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (40, 'person', 'Dana', 5)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, predicate, kind, object_text, confidence, "
        "valid, conversation_id, source_segment_ids) "
        "VALUES (41, 40, 'works_on', 'fact', 'Atlas', 0.9, 1, 7, '[1]')"
    )
    day = _local_day_of("2026-06-16T09:00:00.000Z")

    d = client.get("/api/person/5").json()
    f = d["facts"][0]
    assert f["source_segment_ids"] == [1]
    assert f["source_seg"] == 1 and f["source_day"] == day

    html = client.get("/person/5").text
    assert f'/day?date={day}#seg-1"' in html and ">source</a>" in html


def test_person_page_unknown_identify_form(client, conn):
    conn.execute(
        "INSERT INTO speakers (id, kind, display_label, is_owner) "
        "VALUES (6, 'unknown', 'Unknown #1', 0)"
    )
    html = client.get("/person/6").text
    assert "Who is this voice?" in html
    assert "/api/speakers/6/name" in html  # inline rename posts to the existing API
    assert 'href="/speakers#spk-6"' in html  # escape hatch to voice clips
    # a known, named person gets no identify prompt
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (7, 'Sam', 'known', 0)")
    assert "Who is this voice?" not in client.get("/person/7").text


def test_person_merged_speaker_redirects_to_canonical(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, merged_into) "
        "VALUES (6, 'Dana?', 'unknown', 0, 5)"
    )
    # old bookmarks to the merged voice land on the one canonical profile
    r = client.get("/person/6", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/person/5"
    assert client.get("/person/6").status_code == 200  # redirect resolves cleanly
    # the JSON API keeps its shape (no redirect) and reports the canonical id
    d = client.get("/api/person/6").json()
    assert d["speaker_id"] == 5 and d["label"] == "Dana"


def test_person_page_commitment_track_as_task(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (40, 'person', 'Dana', 5)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, valid) "
        "VALUES (41, 40, 'action_item', 'send the deck', 1)"
    )
    d = client.get("/api/person/5").json()
    owed = d["commitments"]["owed_by"][0]
    assert owed["id"] == 41 and owed["task_id"] is None
    html = client.get("/person/5").text
    assert 'data-edge="41"' in html and "Track as task" in html
    # promoting via the existing endpoint flips the card to its tracked state
    r = client.post("/api/actions/41/promote")
    assert r.status_code == 200 and r.json()["ok"] is True
    d2 = client.get("/api/person/5").json()
    assert d2["commitments"]["owed_by"][0]["task_id"] == r.json()["task_id"]
    html2 = client.get("/person/5").text
    assert "In your tasks" in html2 and 'data-edge="41"' not in html2


def test_person_page_mentions_and_referenced_facts(client, conn):
    from secondbrain.query.service import _local_day_of

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (7, '2026-06-16T09:00:00.000Z', 'diarized')"
    )
    conn.execute("UPDATE audio_files SET conversation_id=7")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (40, 'person', 'Dana', 5)"
    )
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (41, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        "confidence, valid, conversation_id, source_segment_ids) "
        "VALUES (50, 41, 40, 'led_by', 'fact', 'Dana', 0.8, 1, 7, '[1]')"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, confidence, "
        "valid, conversation_id, source_segment_ids) "
        "VALUES (51, 41, 40, 'decision', 'adopt the onboarding flow', 0.9, 1, 7, '[1]')"
    )
    day = _local_day_of("2026-06-16T09:00:00.000Z")

    d = client.get("/api/person/5").json()
    ref = next(f for f in d["facts"] if f["direction"] == "referenced")
    assert ref["other_label"] == "Atlas" and ref["source_seg"] == 1
    m = d["mentions"][0]
    assert m["kind"] == "decision" and m["other_label"] == "Atlas"
    assert m["quotes"][0]["segment_id"] == 1 and m["quotes"][0]["day"] == day

    html = client.get("/person/5").text
    assert "Mentions" in html and ">referenced</span>" in html
    assert "adopt the onboarding flow" in html
    assert f"/day?date={day}#seg-1" in html  # quote deep-links into the day view


def test_person_page_manage_and_rename_affordances(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    html = client.get("/person/5").text
    assert 'href="/speakers#known-5"' in html  # named people link to their People card
    assert 'id="rename-toggle"' in html and 'id="rename-form"' in html
    # unknown voices keep the identify flow (and the spk- anchor) instead
    conn.execute(
        "INSERT INTO speakers (id, kind, display_label, is_owner) "
        "VALUES (6, 'unknown', 'Unknown #1', 0)"
    )
    html_u = client.get("/person/6").text
    assert 'href="/speakers#spk-6"' in html_u
    assert 'id="rename-form"' not in html_u


def test_person_owner_rename_shows_stored_name(client, conn):
    """Renaming the owner is visible on /person and the form prefills the stored name."""
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (4, 'Me', 'owner', 1)")
    r = client.post("/api/speakers/4/name", json={"name": "George"})
    assert r.status_code == 200 and r.json()["ok"] is True
    d = client.get("/api/person/4").json()
    assert d["label"] == "George" and d["name"] == "George" and d["is_owner"] is True
    html = client.get("/person/4").text
    assert "George" in html and "(you)" in html  # page reflects the rename
    assert 'value="George"' in html  # prefill = stored name, never a hardcoded "Me"
    # an owner who never renamed still reads "Me" (and prefills "Me", a no-op save)
    conn.execute("UPDATE speakers SET name='Me', display_label='Me' WHERE id=4")
    d2 = client.get("/api/person/4").json()
    assert d2["label"] == "Me" and d2["name"] == "Me"


def test_person_commitment_due_annotations(client, conn):
    """Commitments carry Tasks-style due labels; dated ones sort above undated."""
    from datetime import date, timedelta

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (40, 'person', 'Dana', 5)"
    )
    overdue = (date.today() - timedelta(days=3)).isoformat()
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, valid) "
        "VALUES (41, 40, 'action_item', 'undated errand', 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, due_date, valid) "
        "VALUES (42, 40, 'action_item', 'send the deck', ?, 1)",
        (overdue,),
    )
    d = client.get("/api/person/5").json()
    owed = d["commitments"]["owed_by"]
    assert [c["object_text"] for c in owed] == ["send the deck", "undated errand"]
    assert owed[0]["due_label"] == "3 days overdue" and owed[0]["overdue"] is True
    assert owed[0]["due_date"] == overdue  # raw field untouched
    assert owed[1]["due_label"] is None and owed[1]["overdue"] is False
    html = client.get("/person/5").text
    assert 'class="overdue"' in html and "3 days overdue" in html


def test_person_page_truncation_notes_and_graph_link(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (40, 'person', 'Dana', 5)"
    )
    for i in range(12):  # 12 conversations, list shows the latest 10
        cid = 100 + i
        day = f"2026-06-{i + 1:02d}"
        conn.execute(
            "INSERT INTO conversations (id, started_at, status) VALUES (?, ?, 'diarized')",
            (cid, f"{day}T09:00:00.000Z"),
        )
        conn.execute(
            "INSERT INTO audio_files (id, path, started_at, sample_rate, status, "
            "conversation_id) VALUES (?, ?, ?, 16000, 'transcribed', ?)",
            (cid, f"/tmp/x{cid}.flac", f"{day}T09:00:00.000Z", cid),
        )
        conn.execute(
            "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')",
            (cid, cid),
        )
        conn.execute(
            "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, "
            "start_offset_s, end_offset_s, start_at, text, speaker_id) "
            "VALUES (?, ?, ?, 0, 2, ?, 'hi there', 5)",
            (1000 + i, cid, cid, f"{day}T09:00:00.000Z"),
        )
    d = client.get("/api/person/5").json()
    assert len(d["recent_conversations"]) == 10
    assert d["interactions"]["conversations"] == 12
    html = client.get("/person/5").text
    assert "Showing the latest 10 of" in html and "12 conversations" in html
    assert 'href="/timeline"' in html
    assert 'href="/graph#node=40"' in html  # dossier links into the knowledge graph


def test_person_page_optedout_unknown_keeps_manage_path(client, conn):
    conn.execute(
        "INSERT INTO speakers (id, kind, display_label, is_owner, opted_out) "
        "VALUES (6, 'unknown', 'Unknown #1', 0, 1)"
    )
    html = client.get("/person/6").text
    assert "Who is this voice?" not in html  # no identify card while opted out
    assert 'id="rename-form"' not in html
    # …but merge/opt-in controls stay reachable from the voice's own page
    assert "Manage on the People page" in html and 'href="/speakers#spk-6"' in html


def test_person_page_valid_time_markup_and_talk_stat(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute("UPDATE transcript_segments SET speaker_id=5")
    # a line with no timestamp must not render <time datetime="">
    conn.execute(
        "INSERT INTO transcript_segments (transcript_id, audio_file_id, start_offset_s, "
        "end_offset_s, text, speaker_id) VALUES (1, 1, 4.0, 6.0, 'undated line', 5)"
    )
    html = client.get("/person/5").text
    assert "undated line" in html
    assert 'datetime=""' not in html
    # 4 s of talk reads 'under 1 min talking', never '0.1 min'
    assert "under 1 min" in html
    d = client.get("/api/person/5").json()
    assert d["interactions"]["talk_label"] == "under 1 min"
    # a voice with no audio shows an honest zero stat
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (7, 'Sam', 'known', 0)")
    assert "min talking" in client.get("/person/7").text


def test_name_endpoint_duplicate_hint(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO speakers (id, kind, display_label, is_owner) "
        "VALUES (6, 'unknown', 'Unknown #1', 0)"
    )
    r = client.post("/api/speakers/6/name", json={"name": "dana"})
    body = r.json()
    assert body["ok"] is True and body["redacted_segments"] == 0
    assert body["duplicate_of"] == {"id": 5, "name": "Dana"}  # case-insensitive match
    # the dossier surfaces the duplicate persistently, with a merge link
    html = client.get("/person/6").text
    assert "Two voices share this name" in html and "/speakers#known-5" in html
    # a unique name carries no hint (additive field is always present, null)
    r2 = client.post("/api/speakers/6/name", json={"name": "Unique Nadia"})
    assert r2.json()["duplicate_of"] is None


def test_relationships_endpoints(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (6, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO speakers (id, display_label, kind, is_owner) VALUES (7, 'Unknown #1', 'unknown', 0)"
    )
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (3, '2026-06-16T09:00:00.000Z', 'diarized')"
    )
    conn.execute("UPDATE audio_files SET conversation_id=3")
    conn.execute("UPDATE transcript_segments SET speaker_id=6")
    conn.execute(
        "INSERT INTO transcript_segments (transcript_id, audio_file_id, start_offset_s, "
        "end_offset_s, start_at, text, speaker_id) "
        "VALUES (1, 1, 2.0, 4.0, '2026-06-16T09:00:02.000Z', 'hi', 7)"
    )
    r = client.get("/api/relationships")
    assert r.status_code == 200
    rel = r.json()["relationships"]
    dana = next(x for x in rel if x["label"] == "Dana")
    # original response contract is untouched…
    assert dana["speaker_id"] == 6 and dana["conversations"] == 1
    assert dana["kind"] == "known" and "talk_minutes" in dana
    # …and the additive fields the page renders are present
    assert "conversations_30d" in dana
    assert dana["last_seen_label"]  # friendly local wording, e.g. "Jun 16"
    assert dana["often_with"][0]["label"] == "Unknown #1"  # who talks with whom
    html = client.get("/relationships").text  # page renders with the new affordances
    assert "sortbtn" in html  # sortable column headers
    assert 'data-named="0"' in html and 'data-named="1"' in html  # unnamed filter hook
    assert "Often with" in html and "Unknown #1" in html


def test_relationships_page_empty_state(client):
    # fixture segments carry no speaker → nobody to rank yet
    html = client.get("/relationships").text
    assert "No interactions recorded yet" in html
    assert "/speakers" in html  # guidance links to the People page


def test_relationships_page_marks_stale(client, conn):
    from datetime import UTC, datetime, timedelta

    old = (datetime.now(UTC) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (6, 'Old Pal', 'known', 0)")
    conn.execute("UPDATE transcript_segments SET speaker_id=6, start_at=?", (old,))
    html = client.get("/relationships").text
    assert "stale-pill" in html  # visible text badge, not color alone
    assert "No interaction in over 30 days" in html  # explains the threshold


def test_project_endpoints(client, conn):
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (10, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, valid) "
        "VALUES (20, 10, 'fact', 'on track', 1)"
    )
    r = client.get("/api/projects")
    assert r.status_code == 200
    assert any(p["label"] == "Atlas" for p in r.json()["projects"])
    assert client.get("/projects").status_code == 200  # list page renders
    d = client.get("/api/project/10")
    assert d.status_code == 200 and d.json()["label"] == "Atlas"
    assert client.get("/project/10").status_code == 200  # dossier page renders
    assert client.get("/api/project/999").status_code == 404


def test_project_error_pages_and_merge_redirect(client, conn):
    # API JSON error contract unchanged (CLI / menu bar rely on it)
    r = client.get("/api/project/999")
    assert r.status_code == 404 and r.json()["detail"] == "project not found"
    r = client.get("/project/999")  # non-browser client on the page route: JSON too
    assert r.status_code == 404 and r.json()["detail"] == "project not found"
    # browsers get a specific, navigable page with the shared nav intact
    r = client.get("/project/999", headers=HTML)
    assert r.status_code == 404
    assert "Project not found" in r.text and 'class="nav"' in r.text
    assert 'href="/projects"' in r.text
    # /project/abc → styled 422 for browsers, default JSON blob otherwise
    r = client.get("/project/abc", headers=HTML)
    assert r.status_code == 422 and 'class="nav"' in r.text
    assert "detail" in client.get("/project/abc").json()
    # a merged node's stale URL redirects to the canonical project page
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (10, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, merged_into) "
        "VALUES (11, 'project', 'Atlas v1', 10)"
    )
    r = client.get("/project/11", headers=HTML, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/project/10"
    # the JSON route keeps returning the dossier directly (canonical id inside)
    assert client.get("/api/project/11").json()["node_id"] == 10


def test_project_page_empty_and_populated_states(client, conn):
    # list page: helpful empty state, not a bare sentence
    html = client.get("/projects", headers=HTML).text
    assert "No projects yet" in html and 'href="/timeline"' in html
    # a just-extracted project with no edges explains itself instead of a blank page
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (10, 'project', 'Atlas')")
    html = client.get("/project/10", headers=HTML).text
    assert "Nothing extracted for this project yet" in html

    # populated dossier: sources deep-link into /day, action items are actionable
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (7, '2026-06-16T09:00:00.000Z', 'diarized')"
    )
    conn.execute("UPDATE audio_files SET conversation_id=7")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, due_date, "
        "conversation_id, source_segment_ids, valid) "
        "VALUES (20, 10, 'action_item', 'send onboarding doc', '2026-06-20', 7, '[1]', 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, conversation_id, "
        "source_segment_ids, valid) VALUES (21, 10, 'decision', 'adopt new flow', 7, '[1]', 1)"
    )
    html = client.get("/project/10", headers=HTML).text
    assert "Track as task" in html  # promotable action item
    assert "#seg-1" in html and "/day?date=" in html  # navigable provenance
    assert "overdue" in html  # due 2026-06-20 is long past
    assert "adopt new flow" in html
    # list page shows the open count and semantic table headers
    html = client.get("/projects", headers=HTML).text
    assert "<thead>" in html and 'scope="col"' in html
    assert ">1<" in html  # one open item

    # promote via the same endpoint the page button calls → tracked state
    r = client.post("/api/actions/20/promote")
    assert r.status_code == 200 and r.json()["ok"] is True
    html = client.get("/project/10", headers=HTML).text
    assert "In your tasks" in html
    # finish the task → item marked done and it leaves the open count
    from secondbrain.query import service as svc

    svc.task_set_status(conn, r.json()["task_id"], "done")
    html = client.get("/project/10", headers=HTML).text
    assert "✓ Done" in html
    assert "1 open" not in html  # heading now says 0 open, 1 done
    assert "0 open, 1 done" in html


def test_project_fact_cards_name_their_subject(client, conn):
    """Person→project facts render as 'Dana — works on this project', never a
    bare echo of the page title; the JSON gains additive subject fields."""
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (12, 'person', 'Dana', 5)"
    )
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (10, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        "confidence, valid) VALUES (20, 12, 10, 'works_on', 'fact', 'Atlas', 0.9, 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, predicate, kind, object_text, confidence, "
        "valid) VALUES (21, 10, 'status', 'fact', 'on track', 0.8, 1)"
    )
    body = client.get("/api/project/10").json()
    works = next(f for f in body["facts"] if f["predicate"] == "works_on")
    assert works["src_label"] == "Dana" and works["object_redundant"] is True
    assert "quotes_total" in body  # additive cap-awareness field
    html = client.get("/project/10", headers=HTML).text
    assert "Dana" in html and "works on this project" in html
    assert 'href="/person/5"' in html  # the subject links to the person page
    assert "works on: Atlas" not in html  # no subject-less echo of the title
    assert "status: on track" in html  # project-as-subject facts stay implicit


def test_projects_list_filter_and_sort_affordances(client, conn):
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (10, 'project', 'Atlas')")
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (11, 'project', 'Beacon')")
    html = client.get("/projects", headers=HTML).text
    assert 'id="proj-filter"' in html and "sortbtn" in html  # filter + sortable headers
    assert 'aria-live="polite"' in html  # filter count announced to screen readers
    assert 'data-search="atlas"' in html and 'data-search="beacon"' in html
    assert "Extracted facts, decisions and action items" in html  # Mentions tooltip
    assert 'title="Action items not yet done"' in html  # Open-items tooltip kept


def test_opted_out_speaker_audio_blocked(client, conn):
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (8, 'X', 'known', 0, 1)"
    )
    assert client.get("/api/speakers/8/samples").status_code == 403
    assert client.get("/api/speakers/8/clip/1").status_code == 403


def test_timeline_endpoints(client):
    r = client.get("/api/timeline/2026-06-16")
    assert r.status_code == 200
    convs = r.json()["conversations"]
    assert len(convs) == 1
    block = convs[0]
    # duration/density metadata rides along for API consumers
    assert block["segment_count"] == 1
    assert "duration_minutes" in block and "duration_label" in block
    assert "ended_at" in block and "start_time" in block
    assert block["segments"][0]["time"]  # local wall-clock display time
    # malformed day -> 422 for API clients (the page falls back gracefully)
    assert client.get("/api/timeline/not-a-date").status_code == 422
    assert client.get("/api/timeline/0001-01-01").status_code == 422
    # unpadded-but-parseable input echoes the canonical day key (same
    # normalization as /api/day and the HTML routes)
    r = client.get("/api/timeline/2026-6-16")
    assert r.status_code == 200 and r.json()["day"] == "2026-06-16"
    assert len(r.json()["conversations"]) == 1
    assert client.get("/timeline/2026-06-16").status_code == 200  # page renders
    assert client.get("/timeline").status_code == 200  # today


def test_timeline_page_overview_and_day_links(client):
    r = client.get("/timeline/2026-06-16")
    assert r.status_code == 200
    # day-to-day navigation
    assert 'href="/timeline/2026-06-15"' in r.text  # prev day
    assert 'href="/timeline/2026-06-17"' in r.text  # next day
    assert 'type="date"' in r.text  # jump-to-date picker
    assert 'href="/timeline"' in r.text  # Today shortcut / shared nav
    # density strip with a jump to the conversation below
    assert 'class="strip"' in r.text
    assert 'href="#conv-1"' in r.text
    # each conversation links into the day view at its first line
    assert 'href="/day?date=2026-06-16#seg-1"' in r.text
    assert "onboarding flow" in r.text  # transcript preview renders
    assert "1 line" in r.text  # density in the header
    # past days are settled — no live-refresh poll target
    assert 'id="fresh-note"' not in r.text


def test_timeline_today_resolves_date_and_empty_state(client):
    r = client.get("/timeline")
    assert r.status_code == 200
    assert "<title>Timeline 2" in r.text  # resolved date, not the literal 'today'
    assert "Nothing recorded yet today" in r.text
    assert "Last recorded day · 2026-06-16" in r.text  # jump back toward the data
    # today keeps growing: the stale-tab poll target renders (today only)
    assert 'id="fresh-note"' in r.text


def test_timeline_daynav_skips_to_recorded_days_and_zooms(client, conn):
    # a second recorded day, weeks before the fixture's 2026-06-16
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (70, '2026-05-05T09:00:00.000Z', 'diarized')"
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (70, '/sk.flac', '2026-05-05T09:00:00.000Z', 16000, 'transcribed', 70)"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (70, 70, 'mock')")
    conn.execute(
        "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, start_offset_s,"
        " end_offset_s, start_at, text) VALUES (700, 70, 70, 0, 2,"
        " '2026-05-05T09:00:00.000Z', 'skip target')"
    )
    r = client.get("/timeline/2026-06-16")
    assert r.status_code == 200
    # adjacent day is empty -> the daynav offers a hop across the gap
    assert "Jump back to the last recorded day, 2026-05-05" in r.text
    assert 'href="/timeline/2026-05-05"' in r.text
    # a one-conversation morning is a narrow window -> the magnified hour
    # axis renders alongside the 24 h strip (with per-span zoom geometry)
    assert "Zoomed in ·" in r.text and "zoom" in r.text
    r2 = client.get("/timeline/2026-05-05")
    assert "Jump forward to the next recorded day, 2026-06-16" in r2.text
    assert 'href="/timeline/2026-06-16"' in r2.text


def test_timeline_page_invalid_date_falls_back_with_notice(client):
    r = client.get("/timeline/not-a-date")
    assert r.status_code == 200
    assert "look like a date" in r.text
    # extreme-but-parseable years can't overflow local-time math
    assert client.get("/timeline/0001-01-01").status_code == 200
    assert client.get("/timeline/9999-12-31").status_code == 200


def test_timeline_caps_long_transcript_previews(client, conn):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (50, '2026-05-05T09:00:00.000Z', 'diarized')"
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (50, '/tl.flac', '2026-05-05T09:00:00.000Z', 16000, 'transcribed', 50)"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (50, 50, 'mock')")
    for i in range(12):
        conn.execute(
            "INSERT INTO transcript_segments (transcript_id, audio_file_id, start_offset_s,"
            " end_offset_s, start_at, text) VALUES (50, 50, ?, ?, ?, ?)",
            (i * 2.0, i * 2.0 + 1.5, f"2026-05-05T09:00:{i * 2:02d}.000Z", f"tlline {i}"),
        )
    r = client.get("/timeline/2026-05-05")
    assert r.status_code == 200
    assert "tlline 7" in r.text  # the preview shows the first lines
    assert "tlline 9" not in r.text  # capped — the rest lives in the day view
    assert "Read all 12 lines in the day view" in r.text


def test_timeline_extractions_link_to_source_segment(client, conn):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (60, '2026-04-04T09:00:00.000Z', 'diarized')"
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (60, '/ex.flac', '2026-04-04T09:00:00.000Z', 16000, 'transcribed', 60)"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (60, 60, 'mock')")
    conn.execute(
        "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, start_offset_s,"
        " end_offset_s, start_at, text) VALUES (600, 60, 60, 0, 2,"
        " '2026-04-04T09:00:00.000Z', 'we should ship on friday')"
    )
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (30, 'person', 'Dana')")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, predicate, object_text, conversation_id,"
        " valid, source_segment_ids) VALUES (99, 30, 'fact', 'works_on', NULL, 60, 1, '[600]')"
    )
    r = client.get("/timeline/2026-04-04")
    assert r.status_code == 200
    assert 'href="/day?date=2026-04-04#seg-600"' in r.text  # grounded, not dead text
    assert "works on" in r.text  # humanised predicate-only edge


def test_search_endpoint(client):
    r = client.get("/api/search", params={"q": "onboarding"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert "onboarding" in results[0]["text"]


def test_search_response_metadata_and_highlight_sentinels(client):
    from secondbrain.query import service
    from secondbrain.search import fulltext

    body = client.get("/api/search", params={"q": "onboarding"}).json()
    # Additive metadata the dashboard UI relies on.
    assert body["count"] == 1
    assert body["limit"] == 20
    assert body["semantic_available"] is False  # disabled in test settings
    hit = body["results"][0]
    # Match markers are invisible sentinels (safe to escape then swap for
    # <mark>), not CLI-style [brackets] leaking into the web UI.
    assert f"{fulltext.MARK_START}onboarding{fulltext.MARK_END}" in hit["snippet"]
    assert "[" not in hit["snippet"]
    # Every hit carries its *local* calendar day for grouping + /day links.
    assert hit["day"] == service._local_day_of(hit["start_at"])
    assert hit["day"] and len(hit["day"]) == 10


def test_search_date_range_uses_local_days(client):
    from secondbrain.query import service

    day = service._local_day_of("2026-06-16T09:00:00.000Z")
    hits = client.get(
        "/api/search", params={"q": "onboarding", "since": day, "until": day}
    ).json()["results"]
    assert len(hits) == 1
    # A window that ends the day before the hit's local day excludes it.
    before = f"{int(day[:4]) - 1}-01-01"
    hits = client.get(
        "/api/search", params={"q": "onboarding", "until": before}
    ).json()["results"]
    assert hits == []


def test_search_rejects_malformed_date_filters(client):
    # A typo'd date must not silently match nothing — that reads as "no results".
    assert client.get("/api/search", params={"q": "x", "since": "notadate"}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "until": "2026-13-45"}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "since": "0001-01-01"}).status_code == 422
    r = client.get("/api/search", params={"q": "x", "until": "nope"})
    assert "until" in r.json()["detail"]
    # Empty values mean "no filter" (a cleared form field), same as omitting them.
    r = client.get("/api/search", params={"q": "onboarding", "since": "", "until": ""})
    assert r.status_code == 200 and r.json()["count"] == 1


def test_status_segments_today_matches_day_view_bucketing(client, conn):
    from secondbrain.query import service

    # A conversation just after local midnight belongs to the local "today"
    # even where the UTC date prefix differs (e.g. BST): the pill must agree
    # with the /day view it links to.
    today = service.local_today()
    ts = _utc_at_local(today, 0, 15)
    af = models.insert_audio_file(
        conn, AudioFile(path="/mid.flac", started_at=ts, sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(t, af, 0.0, 1.0, "night owl", start_at=ts)])
    st = client.get("/api/status").json()
    day = client.get(f"/api/day/{today}").json()
    assert st["segments_today"] == len(day["segments"]) == 1


def test_pause_resume_toggle(client):
    assert client.post("/api/pause").json()["paused"] is True
    assert client.get("/api/status").json()["paused"] is True
    assert client.post("/api/resume").json()["paused"] is False


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SecondBrain" in r.text
    # unified dashboard: shared nav links present
    for href in ('href="/timeline"', 'href="/relationships"', 'href="/projects"', 'href="/speakers"'):
        assert href in r.text


def test_index_page_dashboard_and_search_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    # live recorder status + pause control (recording enabled by default)
    assert 'id="rec-pill"' in r.text
    assert 'id="rec-btn"' in r.text
    # search UI: landmarked form, labeled search input, filters, live regions
    assert 'role="search"' in r.text
    assert 'type="search"' in r.text
    for el in ('id="mode"', 'id="since"', 'id="until"', 'id="res-status"', 'id="results"'):
        assert el in r.text
    # failed-jobs pill exists but stays hidden while nothing has failed; it
    # deep-links to the failure details ON the health page, inside the shell
    assert 'id="pill-jobs"' in r.text
    assert 'href="/health#jobs"' in r.text
    # the health page stays reachable even when nothing has failed
    assert 'href="/health"' in r.text
    # retention pill humanizes hours into days (default 168h)
    assert "audio kept 7 days" in r.text


def test_health_json_for_probes_html_page_for_browsers(client, conn):
    # Probes / CLI / menu bar (accept */*) keep the exact JSON contract.
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["status"] in ("ok", "degraded") and "checks" in body
    # Browsers get an in-shell page (the dashboard's failed-jobs pill lands
    # here), never a raw JSON dump outside the app.
    r = client.get("/health", headers=HTML)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'class="nav"' in r.text            # shared shell, nav intact
    assert "System health" in r.text
    assert "checks passing" in r.text         # overall verdict line
    assert 'id="jobs"' in r.text              # anchor the failed-jobs pill targets
    assert "No failed jobs" in r.text         # empty failure state has copy
    # A dead-lettered job renders with a human job label, error, and attempts.
    conn.execute(
        "INSERT INTO jobs (type, state, attempts, max_attempts, error, finished_at) "
        "VALUES ('extract_knowledge', 'failed', 3, 3, "
        "'HTTPStatusError(\"404 for url http://127.0.0.1:11434/api/chat\")', "
        "'2026-06-16T09:00:00.000Z')"
    )
    r = client.get("/health", headers=HTML)
    assert "Failed jobs" in r.text
    assert "Knowledge extraction" in r.text   # humanised, not the raw job type
    assert "11434/api/chat" in r.text         # the actual error is on the page
    assert "3 of 3 attempts" in r.text
    assert "1 failed" in r.text               # queue counts pills


def test_index_page_recording_disabled_replaces_button(conn, settings):
    settings.consent.recording_enabled = False
    page = TestClient(create_app(settings)).get("/")
    assert page.status_code == 200
    assert 'id="rec-btn"' not in page.text  # no dead Resume button
    assert "switched off in config" in page.text
    assert "Recording off" in page.text


def test_index_page_empty_corpus_state(conn, settings):
    page = TestClient(create_app(settings)).get("/")
    assert page.status_code == 200
    assert "Nothing captured yet" in page.text


def test_shared_nav_on_new_pages(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (7, 'Dana', 'known', 0)")
    for path in ("/person/7", "/relationships", "/timeline"):
        r = client.get(path)
        assert r.status_code == 200
        assert 'class="nav"' in r.text  # extends base.html


def test_speakers_endpoints(client, conn):
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    # list + unknown
    assert client.get("/api/speakers").status_code == 200
    unknown = client.get("/api/speakers/unknown").json()["unknown"]
    assert any(s["id"] == sid for s in unknown)
    # name it
    r = client.post(f"/api/speakers/{sid}/name", json={"name": "Dana"})
    assert r.status_code == 200 and r.json()["ok"]
    names = [s["name"] for s in client.get("/api/speakers").json()["speakers"]]
    assert "Dana" in names


def test_speakers_page_renders(client):
    r = client.get("/speakers")
    assert r.status_code == 200
    assert "Who is this?" in r.text


def test_speakers_page_unknown_card_affordances(client, conn):
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    page = client.get("/speakers").text
    assert f'href="/person/{sid}"' in page  # read-their-lines escape hatch
    assert "Not a person" in page  # dismiss affordance


def test_speaker_samples_guard_and_no_path_leak(client, conn):
    assert client.get("/api/speakers/999/samples").status_code == 404
    conn.execute(
        "INSERT INTO speakers (id, kind, display_label) VALUES (21, 'unknown', 'Unknown #9')"
    )
    conn.execute(
        "INSERT INTO speaker_observations (speaker_id, audio_file_id, start_offset_s, "
        "end_offset_s, start_at, confidence) "
        "VALUES (21, 1, 0.0, 1.5, '2026-06-16T09:00:00.000Z', 0.9)"
    )
    r = client.get("/api/speakers/21/samples")
    assert r.status_code == 200
    samples = r.json()["samples"]
    assert samples
    assert all("path" not in s for s in samples)  # fs paths stay server-side
    # UI contract intact (duration_s is additive: clips say how long they are)
    assert {"id", "start_at", "audio_status", "duration_s"} <= set(samples[0])


def test_speaker_clip_served_inline(client, conn, tmp_path, monkeypatch):
    from secondbrain.query import api as api_module

    conn.execute(
        "INSERT INTO speakers (id, kind, display_label) VALUES (22, 'unknown', 'Unknown #8')"
    )
    cur = conn.execute(
        "INSERT INTO speaker_observations (speaker_id, audio_file_id, start_offset_s, "
        "end_offset_s, start_at, confidence) "
        "VALUES (22, 1, 0.0, 1.0, '2026-06-16T09:00:00.000Z', 0.9)"
    )
    obs = cur.lastrowid
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    monkeypatch.setattr(
        api_module, "_extract_clip", lambda sample, settings, prefix="sample": wav
    )
    r = client.get(f"/api/speakers/22/clip/{obs}")
    assert r.status_code == 200
    # Firefox refuses to play attachment-disposition media in <audio>.
    assert r.headers["content-disposition"].startswith("inline")


def _seed_voice_obs(conn, sid, af, start, end, conf, start_at="2026-06-16T09:00:00.000Z"):
    return conn.execute(
        "INSERT INTO speaker_observations (speaker_id, audio_file_id, start_offset_s, "
        "end_offset_s, start_at, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (sid, af, start, end, start_at, conf),
    ).lastrowid


def test_speaker_samples_prefer_attributed_segment_windows(client, conn):
    """Sub-second exemplar blips are upgraded to the overlapping spoken line.

    Diarization exemplars are often 0.4-0.8s — useless for 'who is this?' by
    ear — while the same voice has 8-10s attributed transcript segments in the
    same audio file. The clip window must come from the segment (clamped to
    ~10s), not the raw blip.
    """
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    af = models.insert_audio_file(
        conn, AudioFile(path="/voice.flac", started_at="2026-06-16T09:00:00.000Z",
                        sample_rate=16000, status="transcribed"),
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [
        # 30s attributed line that contains both blips below
        Segment(t, af, 20.0, 50.0, "long attributed line", speaker_id=sid,
                start_at="2026-06-16T09:00:20.000Z"),
        # another speaker's overlapping line must never be picked
        Segment(t, af, 20.0, 59.0, "someone else talking", speaker_id=999),
    ])
    blip1 = _seed_voice_obs(conn, sid, af, 30.0, 30.4, 0.99)   # highest confidence
    _seed_voice_obs(conn, sid, af, 41.0, 41.6, 0.98)           # same segment → deduped
    lone = _seed_voice_obs(conn, sid, af, 55.0, 58.0, 0.10)    # no overlap → raw window

    samples = client.get(f"/api/speakers/{sid}/samples").json()["samples"]
    # the two blips inside one segment collapse into a single clip
    assert [s["id"] for s in samples] == [blip1, lone]
    # blip window upgraded to the segment's, clamped to 10s from segment start
    assert samples[0]["start_offset_s"] == 20.0 and samples[0]["end_offset_s"] == 30.0
    assert samples[0]["duration_s"] == 10.0
    assert samples[0]["start_at"] == "2026-06-16T09:00:20.000Z"  # honest "heard at"
    # no attributed overlap → the observation's own window is served as-is
    assert samples[1]["start_offset_s"] == 55.0 and samples[1]["end_offset_s"] == 58.0
    assert samples[1]["duration_s"] == 3.0


def test_speaker_samples_blip_falls_back_to_longest_line_in_same_file(client, conn):
    """A sub-usable blip with no overlapping line borrows the voice's longest
    attributed line elsewhere in the same recording (real-data shape: 0.8s
    exemplar at 40s, 6.6s attributed line at 12s of the same file)."""
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    af = models.insert_audio_file(
        conn, AudioFile(path="/voice3.flac", started_at="2026-06-16T09:00:00.000Z",
                        sample_rate=16000, status="transcribed"),
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [
        Segment(t, af, 12.0, 18.6, "their longest line", speaker_id=sid,
                start_at="2026-06-16T09:00:12.000Z"),
        Segment(t, af, 2.0, 3.0, "short line", speaker_id=sid),
    ])
    obs = _seed_voice_obs(conn, sid, af, 40.0, 40.8, 0.9)  # 0.8s, overlaps nothing

    samples = client.get(f"/api/speakers/{sid}/samples").json()["samples"]
    assert samples[0]["id"] == obs
    assert samples[0]["start_offset_s"] == 12.0 and samples[0]["end_offset_s"] == 18.6
    assert samples[0]["duration_s"] == 6.6


def test_speaker_samples_rank_usable_length_over_confidence(client, conn):
    """With no attributed segments, longer windows beat confident blips."""
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    af = models.insert_audio_file(
        conn, AudioFile(path="/voice2.flac", started_at="2026-06-16T09:00:00.000Z",
                        sample_rate=16000, status="transcribed"),
    )
    blip = _seed_voice_obs(conn, sid, af, 1.0, 1.4, 0.99)    # 0.4s, top confidence
    mid = _seed_voice_obs(conn, sid, af, 5.0, 7.0, 0.20)     # 2.0s
    long = _seed_voice_obs(conn, sid, af, 10.0, 18.0, 0.50)  # 8.0s

    samples = client.get(f"/api/speakers/{sid}/samples").json()["samples"]
    assert [s["id"] for s in samples] == [long, mid, blip]  # sub-1.5s blip last


def test_clip_503_when_soundfile_missing(client, conn, tmp_path, monkeypatch):
    """A missing audio backend is a fixable install gap (503), never a 410."""
    from secondbrain.query import api as api_module
    from secondbrain.speaker import registry

    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # exists on disk, not swept
    sid = registry.create_unknown_speaker(conn)
    af = models.insert_audio_file(
        conn, AudioFile(path=str(src), started_at="2026-06-16T09:00:00.000Z",
                        sample_rate=16000, status="transcribed"),
    )
    obs = _seed_voice_obs(conn, sid, af, 0.0, 1.0, 0.9)
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(t, af, 0.0, 2.0, "hi", speaker_id=sid)])
    seg = conn.execute("SELECT id FROM transcript_segments WHERE audio_file_id=?",
                       (af,)).fetchone()["id"]
    monkeypatch.setattr(api_module, "_load_soundfile", lambda: None)

    r = client.get(f"/api/speakers/{sid}/clip/{obs}")
    assert r.status_code == 503
    assert "soundfile" in r.json()["detail"]
    r2 = client.get(f"/api/segments/{seg}/clip")
    assert r2.status_code == 503


def test_clip_served_from_cache_without_soundfile(client, conn, settings, tmp_path, monkeypatch):
    """A previously sliced clip keeps playing even if the backend disappears."""
    from secondbrain.query import api as api_module
    from secondbrain.speaker import registry

    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    sid = registry.create_unknown_speaker(conn)
    af = models.insert_audio_file(
        conn, AudioFile(path=str(src), started_at="2026-06-16T09:00:00.000Z",
                        sample_rate=16000, status="transcribed"),
    )
    obs = _seed_voice_obs(conn, sid, af, 0.0, 1.0, 0.9)
    settings.audio_processed_dir.mkdir(parents=True, exist_ok=True)
    cached = settings.audio_processed_dir / f"sample_{obs}_0-100.wav"
    cached.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    monkeypatch.setattr(api_module, "_load_soundfile", lambda: None)
    assert client.get(f"/api/speakers/{sid}/clip/{obs}").status_code == 200


def test_dismiss_and_restore_unknown_voice(client, conn):
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    conn.execute("UPDATE transcript_segments SET speaker_id=?", (sid,))
    r = client.post(f"/api/speakers/{sid}/dismiss")
    assert r.status_code == 200
    assert r.json()["ok"] and r.json()["already_ignored"] is False
    # gone from the queue, the roster, and the merge targets…
    assert all(s["id"] != sid for s in client.get("/api/speakers/unknown").json()["unknown"])
    assert all(s["id"] != sid for s in client.get("/api/speakers").json()["speakers"])
    # …but listed as ignored, and restorable from the page
    assert any(s["id"] == sid for s in client.get("/api/speakers/ignored").json()["ignored"])
    page = client.get("/speakers").text
    assert "Ignored voices" in page and f'id="ign-{sid}"' in page
    # its transcript lines keep their label (nothing is unattributed)
    assert conn.execute(
        "SELECT speaker_id FROM transcript_segments LIMIT 1"
    ).fetchone()[0] == sid
    # dismissing twice is a harmless no-op
    assert client.post(f"/api/speakers/{sid}/dismiss").json()["already_ignored"] is True
    # restore puts it back in the queue
    r2 = client.post(f"/api/speakers/{sid}/restore")
    assert r2.status_code == 200 and r2.json()["ok"]
    assert any(s["id"] == sid for s in client.get("/api/speakers/unknown").json()["unknown"])


def test_dismiss_and_restore_guards(client, conn):
    from secondbrain.speaker import registry

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (61, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (62, 'Dana', 'known', 0)")
    assert client.post("/api/speakers/61/dismiss").status_code == 400  # owner
    assert client.post("/api/speakers/62/dismiss").status_code == 400  # named person
    assert client.post("/api/speakers/999/dismiss").status_code == 404
    assert client.post("/api/speakers/999/restore").status_code == 404
    assert client.post("/api/speakers/62/restore").status_code == 400  # not ignored
    # restoring a voice that's already in the queue is a no-op, not an error
    sid = registry.create_unknown_speaker(conn)
    r = client.post(f"/api/speakers/{sid}/restore")
    assert r.status_code == 200 and r.json()["already_active"] is True


def test_unknown_label_numbering_skips_ignored(client, conn):
    from secondbrain.speaker import registry

    a = registry.create_unknown_speaker(conn)  # Unknown #1
    assert client.post(f"/api/speakers/{a}/dismiss").status_code == 200
    b = registry.create_unknown_speaker(conn)  # must not mint a second "Unknown #1"
    la = conn.execute("SELECT display_label FROM speakers WHERE id=?", (a,)).fetchone()[0]
    lb = conn.execute("SELECT display_label FROM speakers WHERE id=?", (b,)).fetchone()[0]
    assert la != lb


def test_owner_endpoint_optional_name_atomic(client, conn):
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    r = client.post(f"/api/speakers/{sid}/owner", json={"name": "  Zaza  "})
    assert r.status_code == 200 and r.json()["ok"] and r.json()["name"] == "Zaza"
    row = conn.execute(
        "SELECT name, is_owner, kind FROM speakers WHERE id=?", (sid,)
    ).fetchone()
    assert row["name"] == "Zaza" and row["is_owner"] == 1 and row["kind"] == "owner"
    # the original body-less contract keeps working
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (71, 'Pal', 'known', 0)")
    assert client.post("/api/speakers/71/owner").status_code == 200
    assert conn.execute("SELECT is_owner FROM speakers WHERE id=71").fetchone()[0] == 1
    assert conn.execute(
        "SELECT is_owner FROM speakers WHERE id=?", (sid,)
    ).fetchone()[0] == 0
    # blank name → clean 400
    assert client.post(f"/api/speakers/{sid}/owner", json={"name": "   "}).status_code == 400


def test_merge_undo_roundtrip(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (31, 'Ana', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (32, 'Bo', 'known', 0)")
    conn.execute("UPDATE transcript_segments SET speaker_id=31")
    conn.execute(
        "INSERT INTO speaker_observations (speaker_id, audio_file_id, start_offset_s, "
        "end_offset_s, start_at, confidence) "
        "VALUES (31, 1, 0.0, 1.0, '2026-06-16T09:00:00.000Z', 0.9)"
    )
    r = client.post("/api/speakers/merge", json={"src": 31, "dst": 32})
    assert r.status_code == 200
    body = r.json()
    assert body["relabeled_segments"] == 1 and body["undo_available"] is True
    assert body["kept_name"] is None  # both voices named: no name adoption
    assert conn.execute("SELECT merged_into FROM speakers WHERE id=31").fetchone()[0] == 32
    assert conn.execute("SELECT speaker_id FROM transcript_segments LIMIT 1").fetchone()[0] == 32
    # the page renders the undo strip while the window is open
    page = client.get("/speakers").text
    assert "Undo merge" in page and "Ana" in page
    # undo restores the rows, the merge pointer, and the per-speaker stats
    r2 = client.post("/api/speakers/merge/undo")
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["ok"] and b2["restored_segments"] == 1 and b2["src"]["label"] == "Ana"
    assert conn.execute("SELECT merged_into FROM speakers WHERE id=31").fetchone()[0] is None
    assert conn.execute("SELECT speaker_id FROM transcript_segments LIMIT 1").fetchone()[0] == 31
    assert conn.execute("SELECT speaker_id FROM speaker_observations LIMIT 1").fetchone()[0] == 31
    assert conn.execute("SELECT segment_count FROM speakers WHERE id=31").fetchone()[0] == 1
    assert conn.execute("SELECT segment_count FROM speakers WHERE id=32").fetchone()[0] == 0
    # one-shot: nothing left to undo
    assert client.post("/api/speakers/merge/undo").status_code == 404


def test_merge_undo_expiry_and_empty_state(client, conn):
    import json as _json

    from secondbrain.query import service
    from secondbrain.storage import state as st

    assert client.post("/api/speakers/merge/undo").status_code == 404  # nothing recorded
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (41, 'Ana', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (42, 'Bo', 'known', 0)")
    assert client.post("/api/speakers/merge", json={"src": 41, "dst": 42}).status_code == 200
    snap = _json.loads(st.get_state(conn, service.MERGE_UNDO_KEY))
    snap["at"] = "2020-01-01T00:00:00.000Z"
    st.set_state(conn, service.MERGE_UNDO_KEY, _json.dumps(snap))
    assert "Undo merge" not in client.get("/speakers").text  # stale strip never renders
    assert client.post("/api/speakers/merge/undo").status_code == 410  # window passed
    assert client.post("/api/speakers/merge/undo").status_code == 404  # snapshot cleared


def test_merge_into_opted_out_is_not_undoable(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (51, 'Ana', 'known', 0)")
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) "
        "VALUES (52, 'X', 'known', 0, 1)"
    )
    conn.execute("UPDATE transcript_segments SET speaker_id=51")
    r = client.post("/api/speakers/merge", json={"src": 51, "dst": 52})
    assert r.status_code == 200 and r.json()["undo_available"] is False
    # redaction is irreversible, so there is deliberately nothing to undo
    assert client.post("/api/speakers/merge/undo").status_code == 404


def test_merge_named_into_unknown_keeps_name_and_undo_restores(client, conn):
    from secondbrain.speaker import registry

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (61, 'Ana', 'known', 0)")
    unk = registry.create_unknown_speaker(conn)
    prior_label = conn.execute(
        "SELECT display_label FROM speakers WHERE id=?", (unk,)
    ).fetchone()["display_label"]
    r = client.post("/api/speakers/merge", json={"src": 61, "dst": unk})
    assert r.status_code == 200
    assert r.json()["kept_name"] == "Ana"  # additive field: the surviving name
    row = conn.execute(
        "SELECT name, display_label, kind FROM speakers WHERE id=?", (unk,)
    ).fetchone()
    # the unnamed target adopted the name instead of silently dropping it
    assert row["name"] == "Ana" and row["display_label"] == "Ana" and row["kind"] == "known"
    # undo restores dst's anonymous identity — no phantom second "Ana" left over
    assert client.post("/api/speakers/merge/undo").status_code == 200
    row = conn.execute(
        "SELECT name, display_label, kind FROM speakers WHERE id=?", (unk,)
    ).fetchone()
    assert row["name"] is None and row["display_label"] == prior_label
    assert row["kind"] == "unknown"
    assert conn.execute("SELECT name FROM speakers WHERE id=61").fetchone()["name"] == "Ana"


def test_merge_adopted_name_yields_to_manual_rename_on_undo(client, conn):
    from secondbrain.speaker import registry

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (62, 'Cara', 'known', 0)")
    unk = registry.create_unknown_speaker(conn)
    assert client.post("/api/speakers/merge", json={"src": 62, "dst": unk}).status_code == 200
    # the user renames the merged voice before undoing — their newer name wins
    assert client.post(f"/api/speakers/{unk}/name", json={"name": "Carina"}).status_code == 200
    assert client.post("/api/speakers/merge/undo").status_code == 200
    assert conn.execute(
        "SELECT name FROM speakers WHERE id=?", (unk,)
    ).fetchone()["name"] == "Carina"
    assert conn.execute("SELECT name FROM speakers WHERE id=62").fetchone()["name"] == "Cara"


def test_speaker_routes_reject_out_of_range_ids(client):
    # SQLite ids are 64-bit: a crafted id beyond that must be a clean 422 from
    # validation, never an OverflowError 500 out of the sqlite3 driver.
    huge = "99999999999999999999"
    assert client.get(f"/api/speakers/{huge}/samples").status_code == 422
    assert client.get(f"/api/speakers/{huge}/clip/1").status_code == 422
    assert client.get(f"/api/speakers/1/clip/{huge}").status_code == 422
    assert client.post(f"/api/speakers/{huge}/name", json={"name": "Zed"}).status_code == 422
    assert client.post(f"/api/speakers/{huge}/owner").status_code == 422
    assert client.post(f"/api/speakers/{huge}/dismiss").status_code == 422
    assert client.post(f"/api/speakers/{huge}/restore").status_code == 422
    assert client.post(
        "/api/speakers/merge", json={"src": int(huge), "dst": 1}
    ).status_code == 422
    assert client.post(
        "/api/speakers/merge", json={"src": 1, "dst": int(huge)}
    ).status_code == 422
    # zero / negative ids are equally impossible — rejected up front too
    assert client.post("/api/speakers/merge", json={"src": 0, "dst": 1}).status_code == 422
    assert client.get("/api/speakers/-4/samples").status_code == 422
    # the search speaker filter funnels into the same sqlite lookup
    assert client.get(f"/api/search?q=x&speaker={huge}").status_code == 422
    # segment endpoints take a bare id too — bound both so an overflow id is a
    # clean 422 from validation, not an OverflowError 500 inside sqlite3.
    assert client.get(f"/api/segments/{huge}/clip").status_code == 422
    assert client.get("/api/segments/-1/clip").status_code == 422
    assert client.post(
        f"/api/segments/{huge}/speaker", json={"speaker_id": 1}
    ).status_code == 422
    assert client.post(
        "/api/segments/0/speaker", json={"speaker_id": 1}
    ).status_code == 422


def test_ask_endpoint(client):
    r = client.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "citations" in body
    # grounding flags ride along for the UI's badges
    assert "grounded" in body and "general_used" in body


def test_ask_rejects_empty_question(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422  # min_length
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_ask_accepts_history(client):
    r = client.post(
        "/api/ask",
        json={
            "question": "and what else?",
            "history": [{"question": "what was decided?", "answer": "Onboarding flow [1]."}],
        },
    )
    assert r.status_code == 200
    assert "answer" in r.json()
    # malformed history entries -> validation error, not a 500
    assert (
        client.post("/api/ask", json={"question": "x", "history": ["bogus"]}).status_code == 422
    )


def test_ask_stream_endpoint(client):
    import json as _json

    r = client.post("/api/ask/stream", json={"question": "anything"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = [_json.loads(line) for line in r.text.strip().splitlines()]
    assert any(e["event"] == "delta" and e["text"] for e in events)
    final = events[-1]
    assert final["event"] == "done"
    res = final["result"]
    # The done payload mirrors the /api/ask contract (plus time_window).
    for key in ("answer", "citations", "grounded", "general_used", "time_window"):
        assert key in res
    # Streamed deltas concatenate to the final answer.
    assert "".join(e["text"] for e in events if e["event"] == "delta") == res["answer"]


def test_ask_stream_validates_like_ask(client):
    assert client.post("/api/ask/stream", json={"question": ""}).status_code == 422
    assert client.post("/api/ask/stream", json={"question": "   "}).status_code == 400
    assert (
        client.post("/api/ask/stream", json={"question": "x", "history": ["bogus"]}).status_code
        == 422
    )


def test_ask_stream_reports_llm_failure_in_band(client, monkeypatch):
    import json as _json

    import httpx

    from secondbrain.llm import client as llm_client

    class BoomLLM(llm_client.MockLLM):
        async def astream(self, **kwargs):
            raise httpx.ConnectError("nope")
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(llm_client, "get_llm", lambda s=None: BoomLLM())
    r = client.post("/api/ask/stream", json={"question": "anything"})
    assert r.status_code == 200  # stream already started: error travels in-band
    events = [_json.loads(line) for line in r.text.strip().splitlines()]
    assert events[-1]["event"] == "error"
    assert "Ollama" in events[-1]["detail"]


def test_ask_response_includes_time_window_for_temporal_questions(client):
    r = client.post("/api/ask", json={"question": "What did I talk about today?"})
    assert r.status_code == 200
    body = r.json()
    assert body["time_window"]["label"] == "today"


def test_ask_maps_llm_failures_to_503(client, monkeypatch):
    import httpx

    from secondbrain.query import service

    def _raise_connect(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(service, "ask", _raise_connect)
    r = client.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]

    def _raise_timeout(*a, **k):
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(service, "ask", _raise_timeout)
    r = client.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 503
    assert "didn't answer" in r.json()["detail"]

    def _raise_status(*a, **k):
        req = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
        raise httpx.HTTPStatusError(
            "not found", request=req, response=httpx.Response(404, request=req)
        )

    monkeypatch.setattr(service, "ask", _raise_status)
    r = client.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 503
    assert "ollama pull" in r.json()["detail"]


def test_graph_endpoints(client, conn):
    from secondbrain.knowledge import graph

    nid = graph.create_node(conn, type="project", name="Atlas", embedding=None,
                            confidence=0.9, extraction_id=None)
    found = client.get("/api/graph/search", params={"q": "atlas"}).json()["nodes"]
    assert any(n["id"] == nid for n in found)
    detail = client.get(f"/api/graph/node/{nid}").json()
    assert detail["node"]["name"] == "Atlas"


def test_graph_search_browse_alias_and_multiterm(client, conn):
    from secondbrain.knowledge import graph

    # Empty graph: browse mode reports zero totals instead of erroring.
    r = client.get("/api/graph/search").json()
    assert r == {"nodes": [], "total": 0, "node_total": 0, "offset": 0}

    atlas = graph.create_node(conn, type="project", name="Project Atlas", embedding=None,
                              confidence=0.9, extraction_id=None)
    greg = graph.create_node(conn, type="person", name="Greg", embedding=None,
                             confidence=0.9, extraction_id=None)
    full = graph.create_node(conn, type="person", name="Hallman", embedding=None,
                             confidence=0.9, extraction_id=None)
    graph.upsert_edge(conn, src_node_id=full, dst_node_id=atlas, predicate="works_on",
                      kind="fact", confidence=0.9)
    graph.add_alias(conn, full, "The Duke")
    graph.merge_nodes(conn, greg, full)

    # Browse (no q): most-connected non-merged nodes, with totals and labels.
    r = client.get("/api/graph/search").json()
    ids = [n["id"] for n in r["nodes"]]
    assert full in ids and atlas in ids and greg not in ids
    assert r["total"] == 2 and r["node_total"] == 2
    assert all(n["label"] for n in r["nodes"])
    assert r["nodes"][0]["edge_count"] == 1

    # Aliases match: the merged-away name and an explicit alias both find the
    # canonical node ('hallman' contains neither 'greg' nor 'duke'). Each hit
    # reports WHICH alias matched so the UI can explain the row ('via "Greg"').
    for q, alias in (("greg", "Greg"), ("duke", "The Duke")):
        found = client.get("/api/graph/search", params={"q": q}).json()
        assert [n["id"] for n in found["nodes"]] == [full], q
        assert found["nodes"][0]["matched_alias"] == alias, q
    # ...but a name match needs no explanation (and browse mode never does).
    found = client.get("/api/graph/search", params={"q": "hallman"}).json()
    assert found["nodes"][0]["matched_alias"] is None
    browse = client.get("/api/graph/search").json()
    assert all(n["matched_alias"] is None for n in browse["nodes"])

    # Multi-term queries AND their terms regardless of word order.
    found = client.get("/api/graph/search", params={"q": "atlas project"}).json()
    assert [n["id"] for n in found["nodes"]] == [atlas]
    # ...and a term that matches nothing yields no rows but keeps node_total.
    found = client.get("/api/graph/search", params={"q": "atlas zebra"}).json()
    assert found["nodes"] == [] and found["total"] == 0 and found["node_total"] == 2


def test_graph_search_pagination(client, conn):
    from secondbrain.knowledge import graph

    for i in range(25):
        graph.create_node(conn, type="topic", name=f"Topic {i:02d}", embedding=None,
                          confidence=0.5, extraction_id=None)

    # Pages are disjoint and echo their offset (the UI's "Show more").
    first = client.get("/api/graph/search", params={"limit": 10}).json()
    assert len(first["nodes"]) == 10 and first["total"] == 25 and first["offset"] == 0
    second = client.get("/api/graph/search", params={"limit": 10, "offset": 10}).json()
    assert len(second["nodes"]) == 10 and second["offset"] == 10
    third = client.get("/api/graph/search", params={"limit": 10, "offset": 20}).json()
    assert len(third["nodes"]) == 5
    seen = [n["id"] for n in first["nodes"] + second["nodes"] + third["nodes"]]
    assert len(seen) == len(set(seen)) == 25  # every node reachable exactly once
    # Past the end: an empty page with totals intact (the "no more" signal).
    past = client.get("/api/graph/search", params={"offset": 100}).json()
    assert past["nodes"] == [] and past["total"] == 25
    # Offsets page through filtered matches too.
    m = client.get("/api/graph/search", params={"q": "topic", "limit": 20, "offset": 20}).json()
    assert len(m["nodes"]) == 5 and m["total"] == 25
    # Malformed paging params are rejected, not clamped.
    assert client.get("/api/graph/search", params={"offset": -1}).status_code == 422


def test_graph_search_type_filter(client, conn):
    from secondbrain.knowledge import graph

    dana = graph.create_node(conn, type="person", name="Dana", embedding=None,
                             confidence=0.9, extraction_id=None)
    atlas = graph.create_node(conn, type="project", name="Atlas", embedding=None,
                              confidence=0.9, extraction_id=None)
    graph.create_node(conn, type="topic", name="Budget", embedding=None,
                      confidence=0.5, extraction_id=None)

    # type narrows browse mode; node_total still counts the whole graph.
    r = client.get("/api/graph/search", params={"type": "person"}).json()
    assert [n["id"] for n in r["nodes"]] == [dana]
    assert r["total"] == 1 and r["node_total"] == 3

    # ...and combines with q (both must hold).
    r = client.get("/api/graph/search", params={"q": "atlas", "type": "project"}).json()
    assert [n["id"] for n in r["nodes"]] == [atlas] and r["total"] == 1
    r = client.get("/api/graph/search", params={"q": "atlas", "type": "person"}).json()
    assert r["nodes"] == [] and r["total"] == 0 and r["node_total"] == 3

    # An empty type means "all" (what the UI's All pill sends is no param at
    # all, but a hand-built URL with type= must not 422)...
    r = client.get("/api/graph/search", params={"type": ""}).json()
    assert r["total"] == 3
    # ...while an unknown type is rejected, not silently ignored.
    assert client.get("/api/graph/search", params={"type": "banana"}).status_code == 422


def test_int_params_beyond_sqlite_range_are_rejected(client):
    """Ids/offsets past SQLite's signed 64-bit range used to raise
    OverflowError inside the driver (a 500 + traceback); they must fail
    validation like any other bad input instead."""
    too_big = str(10**20)  # > 2**63 - 1
    assert client.get(f"/api/graph/node/{too_big}").status_code == 422
    assert client.get("/api/graph/node/0").status_code == 422  # ids start at 1
    r = client.get("/api/graph/search", params={"offset": str(10**26)})
    assert r.status_code == 422
    # The bound is generous: a billion-row offset is still a clean 200.
    r = client.get("/api/graph/search", params={"offset": 1_000_000_000})
    assert r.status_code == 200 and r.json()["nodes"] == []
    # The person routes share the same guard (JSON and HTML alike).
    assert client.get(f"/api/person/{too_big}").status_code == 422
    html = client.get(f"/person/{too_big}", headers={"accept": "text/html"})
    assert html.status_code == 422
    assert "look right" in html.text  # friendly error page, not a traceback
    # Projects, goals, tasks, suggestions and action edges take bare ids too;
    # each must 422 on an overflowing id rather than 500 out of the driver.
    # (Valid bodies are sent where required so the id guard is what trips.)
    assert client.get(f"/api/project/{too_big}").status_code == 422
    assert client.get(f"/project/{too_big}").status_code == 422
    assert client.get(f"/api/goals/{too_big}").status_code == 422
    assert client.delete(f"/api/goals/{too_big}").status_code == 422
    assert client.get("/api/tasks", params={"goal_id": too_big}).status_code == 422
    assert client.post(
        f"/api/tasks/{too_big}/status", json={"status": "done"}
    ).status_code == 422
    assert client.get(f"/api/tasks/{too_big}/research").status_code == 422
    assert client.post(
        f"/api/suggestions/{too_big}/action", json={"action": "done"}
    ).status_code == 422
    assert client.post(f"/api/actions/{too_big}/promote").status_code == 422
    assert client.post(f"/api/goals/{too_big}/decompose").status_code == 422


def test_graph_search_escapes_like_wildcards(client, conn):
    from secondbrain.knowledge import graph

    pct = graph.create_node(conn, type="topic", name="100% Coverage", embedding=None,
                            confidence=0.5, extraction_id=None)
    sara = graph.create_node(conn, type="person", name="Sara", embedding=None,
                             confidence=0.5, extraction_id=None)
    graph.add_alias(conn, sara, "The 50% Partner")

    # '%' means a literal percent sign — in names and in aliases — not "match
    # everything" (edge_count ties break by name: '100% coverage' < 'sara').
    r = client.get("/api/graph/search", params={"q": "%"}).json()
    assert [n["id"] for n in r["nodes"]] == [pct, sara] and r["total"] == 2
    r = client.get("/api/graph/search", params={"q": "100%"}).json()
    assert [n["id"] for n in r["nodes"]] == [pct]
    r = client.get("/api/graph/search", params={"q": "50%"}).json()
    assert [n["id"] for n in r["nodes"]] == [sara]
    # '_' is a literal underscore, not "any one character" ('a_a' ≠ 'ara').
    r = client.get("/api/graph/search", params={"q": "a_a"}).json()
    assert r["nodes"] == [] and r["total"] == 0
    # A lone backslash is literal too (no ESCAPE-sequence breakage → no 500).
    resp = client.get("/api/graph/search", params={"q": "\\"})
    assert resp.status_code == 200 and resp.json()["nodes"] == []


def _utc_at_local(day: str, hh: int, mm: int) -> str:
    """UTC storage timestamp for a *local* wall-clock time (tz-robust tests)."""
    from datetime import UTC, datetime

    naive = datetime.strptime(f"{day} {hh:02d}:{mm:02d}", "%Y-%m-%d %H:%M")
    return naive.astimezone().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_graph_node_provenance_merge_resolution_and_404(client, conn):
    from secondbrain.knowledge import graph

    started = _utc_at_local("2026-06-16", 9, 5)
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (7, ?, 'diarized')",
        (started,),
    )
    dana = graph.create_node(conn, type="person", name="Dana", embedding=None,
                             confidence=0.9, extraction_id=None)
    atlas = graph.create_node(conn, type="project", name="Atlas", embedding=None,
                              confidence=0.8, extraction_id=None)
    # The client fixture seeded transcript segment id 1 — cite it.
    graph.upsert_edge(conn, src_node_id=dana, dst_node_id=atlas, predicate="leads",
                      kind="fact", confidence=0.9, conversation_id=7,
                      source_segment_ids=[1])
    graph.upsert_edge(conn, src_node_id=dana, dst_node_id=None, predicate=None,
                      kind="action_item", object_text="send the report",
                      confidence=0.4, due_date="2026-06-20")

    d = client.get(f"/api/graph/node/{dana}").json()
    assert d["node"]["label"] == "Dana"
    assert d["edge_count"] == 2 and len(d["edges"]) == 2
    fact = next(e for e in d["edges"] if e["kind"] == "fact")
    # Both endpoints are addressable so the UI can link node-to-node.
    assert fact["src_id"] == dana and fact["dst_id"] == atlas
    assert fact["src_label"] == "Dana" and fact["dst_label"] == "Atlas"
    # Provenance: conversation resolved to the owner's local day/time + citation.
    assert fact["conversation_id"] == 7
    assert fact["conversation_day"] == "2026-06-16"
    assert fact["conversation_time"] == "09:05"
    assert fact["segment_ids"] == [1]
    assert fact["quote"]["segment_id"] == 1
    assert "onboarding flow" in fact["quote"]["text"]
    assert [q["segment_id"] for q in fact["quotes"]] == [1]  # full list mirrors it
    item = next(e for e in d["edges"] if e["kind"] == "action_item")
    assert item["conversation_day"] is None and item["quote"] is None
    assert item["quotes"] == []

    # A node with a binary embedding must still serialize (BLOB is excluded).
    emb = graph.create_node(conn, type="topic", name="Embeddings", embedding=[0.1] * 8,
                            confidence=0.5, extraction_id=None)
    r = client.get(f"/api/graph/node/{emb}")
    assert r.status_code == 200
    assert "embedding" not in r.json()["node"]

    # Merged ids resolve to the canonical node; aliases are reported.
    dupe = graph.create_node(conn, type="person", name="D. Smith", embedding=None,
                             confidence=0.9, extraction_id=None)
    graph.merge_nodes(conn, dupe, dana)
    d = client.get(f"/api/graph/node/{dupe}").json()
    assert d["node"]["id"] == dana
    assert "D. Smith" in d["aliases"]

    # Unknown ids get a helpful 404 the UI shows inline.
    r = client.get("/api/graph/node/999999")
    assert r.status_code == 404
    assert "merged" in r.json()["detail"]


def test_graph_node_lists_every_citable_quote(client, conn):
    """An edge cited by several transcript lines surfaces all of them, in
    citation order, each carrying its own segment id for deep-linking."""
    from secondbrain.knowledge import graph

    af = models.insert_audio_file(
        conn, AudioFile(path="/b.flac", started_at="2026-06-16T09:03:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [Segment(t, af, 0.0, 2.0, "we should double the budget",
                 start_at="2026-06-16T09:03:00.000Z"),
         Segment(t, af, 2.0, 4.0, "and hire two more people",
                 start_at="2026-06-16T09:03:10.000Z")],
    )  # ids 2 and 3 (the client fixture seeded id 1)
    node = graph.create_node(conn, type="topic", name="Budget", embedding=None,
                             confidence=0.9, extraction_id=None)
    graph.upsert_edge(conn, src_node_id=node, dst_node_id=None, predicate="doubled",
                      kind="decision", object_text="the budget", confidence=0.9,
                      source_segment_ids=[3, 999, 1, 2])  # 999: dangling citation
    # One line has an identified voice: its quote must say WHO said it.
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Zaza', 'known', 1)")
    conn.execute("UPDATE transcript_segments SET speaker_id=5 WHERE id=2")

    e = client.get(f"/api/graph/node/{node}").json()["edges"][0]
    # upsert_edge stores citations sorted, so quotes come back in segment-id
    # (= chronological) order; the dangling 999 is skipped, not erred on.
    assert e["segment_ids"] == [1, 2, 3, 999]
    assert [q["segment_id"] for q in e["quotes"]] == [1, 2, 3]
    assert e["quotes"][1]["text"] == "we should double the budget"
    assert all(q["start_at"] for q in e["quotes"])
    assert e["quote"] == e["quotes"][0]  # legacy field still = first quote
    # Speaker attribution: named where known, None for unidentified voices.
    assert e["quotes"][1]["speaker"] == "Zaza"
    assert e["quotes"][0]["speaker"] is None and e["quotes"][2]["speaker"] is None


def test_graph_node_quote_respects_opt_out(client, conn):
    from secondbrain.knowledge import graph

    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) "
        "VALUES (9, 'Private Pat', 'known', 0, 1)"
    )
    conn.execute("UPDATE transcript_segments SET speaker_id=9 WHERE id=1")
    node = graph.create_node(conn, type="topic", name="Budget", embedding=None,
                             confidence=0.9, extraction_id=None)
    graph.upsert_edge(conn, src_node_id=node, dst_node_id=None, predicate="costs",
                      kind="fact", object_text="$100", confidence=0.9,
                      source_segment_ids=[1])
    e = client.get(f"/api/graph/node/{node}").json()["edges"][0]
    assert e["segment_ids"] == [1]
    assert e["quote"] is None  # opted-out speaker's words never surface
    assert e["quotes"] == []   # ...in the full list either


def test_chat_and_graph_pages_render(client):
    assert "Ask your second brain" in client.get("/chat").text
    assert "Knowledge graph" in client.get("/graph").text


def test_tasks_api_and_plan(client):
    tid = client.post("/api/tasks", json={"title": "Write spec", "value": 5,
                                          "estimate_minutes": 30}).json()["id"]
    assert any(t["id"] == tid for t in client.get("/api/tasks").json()["tasks"])
    plan = client.post("/api/plan/today", json={"action": "propose"}).json()
    assert tid in plan["task_ids"]
    assert plan["plan"]["task_ids"] == plan["task_ids"]  # consistent envelope
    accepted = client.post("/api/plan/today", json={"action": "accept"}).json()
    assert accepted["status"] == "accepted"              # legacy top-level mirror
    assert accepted["plan"]["status"] == "accepted"
    assert client.post(f"/api/tasks/{tid}/status", json={"status": "done"}).json()["ok"]


def test_tasks_page_renders(client):
    assert "Tasks" in client.get("/tasks").text


def test_task_create_validation(client):
    assert client.post("/api/tasks", json={"title": "   "}).status_code == 422
    assert client.post("/api/tasks", json={"title": "x", "value": 9}).status_code == 422
    assert client.post("/api/tasks", json={"title": "x", "effort": 0}).status_code == 422
    assert client.post("/api/tasks", json={"title": "x", "estimate_minutes": -5}).status_code == 422
    assert client.post("/api/tasks", json={"title": "x", "due_date": "not-a-date"}).status_code == 422
    assert client.post("/api/tasks", json={"title": "x", "goal_id": 4242}).status_code == 404
    tid = client.post("/api/tasks", json={"title": "  padded  "}).json()["id"]
    t = next(t for t in client.get("/api/tasks").json()["tasks"] if t["id"] == tid)
    assert t["title"] == "padded"  # stored stripped


def test_task_status_validation_and_undo(client):
    tid = client.post("/api/tasks", json={"title": "flip me"}).json()["id"]
    assert client.post(f"/api/tasks/{tid}/status", json={"status": "banana"}).status_code == 422
    assert client.post("/api/tasks/99999/status", json={"status": "done"}).status_code == 404
    assert client.post(f"/api/tasks/{tid}/status", json={"status": "done"}).json()["ok"]
    done = next(t for t in client.get("/api/tasks").json()["tasks"] if t["id"] == tid)
    assert done["completed_at"]
    # Undo: done → backlog clears completed_at
    assert client.post(f"/api/tasks/{tid}/status", json={"status": "backlog"}).json()["ok"]
    undone = next(t for t in client.get("/api/tasks").json()["tasks"] if t["id"] == tid)
    assert undone["status"] == "backlog" and not undone["completed_at"]


def test_task_patch_edits_and_validates(client):
    tid = client.post("/api/tasks", json={"title": "tpyo"}).json()["id"]
    r = client.patch(f"/api/tasks/{tid}", json={
        "title": "typo fixed", "due_date": "2026-08-01", "estimate_minutes": 45, "value": 5})
    task = r.json()["task"]
    assert task["title"] == "typo fixed" and task["due_date"] == "2026-08-01"
    assert task["estimate_minutes"] == 45 and task["value"] == 5
    assert "quadrant" in task and "priority_score" in task  # annotated like GET
    # clearing: '' due date and 0 estimate
    task = client.patch(f"/api/tasks/{tid}", json={"due_date": "", "estimate_minutes": 0}).json()["task"]
    assert task["due_date"] is None and task["estimate_minutes"] is None
    assert client.patch(f"/api/tasks/{tid}", json={}).status_code == 422
    assert client.patch(f"/api/tasks/{tid}", json={"title": " "}).status_code == 422
    assert client.patch(f"/api/tasks/{tid}", json={"value": 7}).status_code == 422
    assert client.patch(f"/api/tasks/{tid}", json={"due_date": "2026-99-99"}).status_code == 422
    assert client.patch("/api/tasks/99999", json={"title": "x"}).status_code == 404


def test_tasks_list_annotates_priority_and_validates_status(client):
    from secondbrain.query import service

    hot = client.post("/api/tasks", json={
        "title": "urgent important", "value": 5, "due_date": service.local_today()}).json()["id"]
    dull = client.post("/api/tasks", json={"title": "someday", "value": 2}).json()["id"]
    tasks = {t["id"]: t for t in client.get("/api/tasks").json()["tasks"]}
    assert tasks[hot]["quadrant"] == "do" and tasks[dull]["quadrant"] == "eliminate"
    assert tasks[hot]["priority_score"] > tasks[dull]["priority_score"]
    assert client.get("/api/tasks?status=banana").status_code == 422
    assert client.get("/api/tasks?status=done").status_code == 200


def test_task_research_get_post_and_404s(client):
    assert client.get("/api/tasks/99999/research").status_code == 404
    # missing task 404s BEFORE any LLM work happens
    assert client.post("/api/tasks/99999/research", json={"web": False}).status_code == 404
    tid = client.post("/api/tasks", json={"title": "look into pricing"}).json()["id"]
    assert client.get(f"/api/tasks/{tid}/research").json()["notes"] == []
    # web research is disabled in test settings → clean 400, not a 500
    assert client.post(f"/api/tasks/{tid}/research", json={"web": True}).status_code == 400
    r = client.post(f"/api/tasks/{tid}/research", json={"web": False})  # mock LLM
    assert r.status_code == 200 and r.json()["notes"]
    notes = client.get(f"/api/tasks/{tid}/research").json()["notes"]
    assert len(notes) == 1 and notes[0]["summary_md"]
    assert isinstance(notes[0]["sources"], list)  # parsed for the UI
    assert notes[0]["backend"] == "local"


def test_plan_get_is_pure_and_post_validates(client, conn):
    # A bare GET must never create a plan (pollers/prefetches are safe).
    assert client.get("/api/plan/today").json() == {"plan": None}
    assert conn.execute("SELECT COUNT(*) AS n FROM day_plans").fetchone()["n"] == 0
    # accept before any proposal → clear conflict
    assert client.post("/api/plan/today", json={"action": "accept"}).status_code == 409
    assert client.post("/api/plan/today", json={"action": "banana"}).status_code == 422
    assert client.post(
        "/api/plan/today", json={"action": "propose", "capacity_minutes": 5}
    ).status_code == 422
    # proposing with an empty backlog stores an empty plan; accepting it is a 409
    plan = client.post("/api/plan/today", json={"action": "propose"}).json()
    assert plan["task_ids"] == []
    assert client.post("/api/plan/today", json={"action": "accept"}).status_code == 409
    # GET always carries the 'plan' envelope, plus legacy top-level mirrors.
    body = client.get("/api/plan/today").json()
    assert body["status"] == "proposed"
    assert body["plan"]["status"] == "proposed" and body["plan"]["task_ids"] == []


def test_task_detail_create_edit_render_and_research_query(client, conn):
    tid = client.post("/api/tasks", json={
        "title": "Fix the fence", "detail": "  Left post is rotten — get quotes first  ",
    }).json()["id"]
    t = next(t for t in client.get("/api/tasks").json()["tasks"] if t["id"] == tid)
    assert t["detail"] == "Left post is rotten — get quotes first"  # stored stripped
    assert "Left post is rotten" in client.get("/tasks").text      # rendered on the row
    # auto-built research queries include the detail, not just the title
    client.post(f"/api/tasks/{tid}/research", json={"web": False})
    q = conn.execute(
        "SELECT query FROM task_research WHERE task_id=?", (tid,)
    ).fetchone()["query"]
    assert "Fix the fence" in q and "Left post is rotten" in q
    # PATCH with '' clears it
    t = client.patch(f"/api/tasks/{tid}", json={"detail": ""}).json()["task"]
    assert t["detail"] is None


def test_research_note_sources_carry_day_for_links(client, conn):
    import json as _json
    from datetime import datetime

    # A note stored by an older build: seg ref without a day recorded.
    tid = client.post("/api/tasks", json={"title": "old note"}).json()["id"]
    conn.execute(
        "INSERT INTO task_research (task_id, query, backend, summary_md, sources) "
        "VALUES (?, 'q', 'local', 'summary', ?)",
        (tid, _json.dumps([{"title": "Me · then", "ref": "seg:1"},
                           {"title": "gone", "ref": "seg:999999"}])),
    )
    src = client.get(f"/api/tasks/{tid}/research").json()["notes"][0]["sources"]
    # seg 1 was seeded at 2026-06-16T09:00:00Z; day = that instant's LOCAL date.
    expected = (datetime.fromisoformat("2026-06-16T09:00:00+00:00")
                .astimezone().strftime("%Y-%m-%d"))
    assert src[0]["day"] == expected
    assert "day" not in src[1]  # unknown segment: no misleading link target


def test_plan_capacity_override(client):
    for i in range(3):
        client.post("/api/tasks", json={"title": f"chunk {i}", "estimate_minutes": 30})
    plan = client.post(
        "/api/plan/today", json={"action": "propose", "capacity_minutes": 30}
    ).json()
    assert plan["capacity_minutes"] == 30 and len(plan["task_ids"]) == 1
    plan = client.post(
        "/api/plan/today", json={"action": "propose", "capacity_minutes": 90}
    ).json()
    assert len(plan["task_ids"]) == 3


def test_action_item_promotion_flow(client, conn):
    from secondbrain.knowledge import graph

    me = graph.create_node(conn, type="person", name="Me", embedding=None,
                           confidence=1.0, extraction_id=None)
    edge = graph.upsert_edge(conn, src_node_id=me, dst_node_id=None, predicate="action_item",
                             kind="action_item", object_text="send the quarterly report",
                             due_date="2026-06-17", source_segment_ids=[1])
    page = client.get("/tasks").text
    assert "send the quarterly report" in page and "Add to tasks" in page
    r = client.post(f"/api/actions/{edge}/promote")
    tid = r.json()["task_id"]
    assert r.json()["title"] == "send the quarterly report"
    # idempotent: promoting again returns the same task
    assert client.post(f"/api/actions/{edge}/promote").json()["task_id"] == tid
    assert client.post("/api/actions/99999/promote").status_code == 404
    task = next(t for t in client.get("/api/tasks").json()["tasks"] if t["id"] == tid)
    assert task["source"] == "conversation" and task["source_edge_id"] == edge
    # once promoted it leaves the detected list; the task itself is on the page
    page = client.get("/tasks").text
    assert "No action items detected" in page


def test_action_item_dismiss_flow(client, conn):
    from secondbrain.knowledge import graph

    me = graph.create_node(conn, type="person", name="Me", embedding=None,
                           confidence=1.0, extraction_id=None)
    edge = graph.upsert_edge(conn, src_node_id=me, dst_node_id=None, predicate="action_item",
                             kind="action_item", object_text="might repaint the shed",
                             source_segment_ids=[1])
    page = client.get("/tasks").text
    assert "might repaint the shed" in page and "Dismiss" in page

    assert client.post(f"/api/actions/{edge}/dismiss").json()["ok"]
    # gone from the page for good — but marked invalid, never deleted
    page = client.get("/tasks").text
    assert "might repaint the shed" not in page
    assert "No action items detected" in page
    row = conn.execute("SELECT valid FROM kg_edges WHERE id=?", (edge,)).fetchone()
    assert row["valid"] == 0
    # no task was created by dismissing
    assert all(t["source_edge_id"] != edge
               for t in client.get("/api/tasks").json()["tasks"])
    # idempotent for a known edge; honest 404 for an unknown one
    assert client.post(f"/api/actions/{edge}/dismiss").json()["ok"]
    assert client.post("/api/actions/99999/dismiss").status_code == 404


def test_plan_remove_task_endpoint(client):
    keep = client.post("/api/tasks", json={"title": "keep me",
                                           "estimate_minutes": 10}).json()["id"]
    out = client.post("/api/tasks", json={"title": "not today",
                                          "estimate_minutes": 10}).json()["id"]
    # no plan yet → clear conflict
    assert client.post("/api/plan/today",
                       json={"action": "remove_task", "task_id": keep}).status_code == 409
    client.post("/api/plan/today", json={"action": "propose"})
    client.post("/api/plan/today", json={"action": "accept"})
    # the affordance renders on plan rows
    assert "Not today" in client.get("/tasks").text
    # task_id is required and must be in the plan
    assert client.post("/api/plan/today", json={"action": "remove_task"}).status_code == 422
    assert client.post("/api/plan/today",
                       json={"action": "remove_task", "task_id": 99999}).status_code == 404

    body = client.post("/api/plan/today",
                       json={"action": "remove_task", "task_id": out}).json()
    assert out not in body["task_ids"] and keep in body["task_ids"]
    assert body["plan"]["status"] == "accepted"       # plan stays accepted
    t = next(t for t in client.get("/api/tasks").json()["tasks"] if t["id"] == out)
    assert t["status"] == "backlog" and t["scheduled_for"] is None
    # the removed task is back in the Backlog section of the page
    assert "not today" in client.get("/tasks").text


def test_tasks_page_releases_stale_scheduled(client, conn):
    tid = client.post("/api/tasks", json={"title": "stuck in the past"}).json()["id"]
    conn.execute(
        "UPDATE tasks SET status='scheduled', scheduled_for='2001-01-01' WHERE id=?", (tid,)
    )
    page = client.get("/tasks").text
    assert "No plan for today yet" in page      # no contradictory 'scheduled' pill…
    assert "ts-scheduled" not in page
    t = next(t for t in client.get("/api/tasks").json()["tasks"] if t["id"] == tid)
    assert t["status"] == "backlog" and t["scheduled_for"] is None


def test_web_research_button_gated_by_config(client, settings):
    client.post("/api/tasks", json={"title": "compare fence quotes"})
    page = client.get("/tasks").text
    assert ">Research<" in page                 # local research is always there
    assert "Research (web)" not in page         # web is off by default
    settings.tasks.web_research_enabled = True
    on = TestClient(create_app(settings))
    assert "Research (web)" in on.get("/tasks").text


def test_goal_sourced_tasks_link_their_goal(client):
    gid = client.post("/api/goals", json={"title": "Learn woodworking"}).json()["id"]
    client.post("/api/tasks", json={"title": "Buy chisels", "goal_id": gid})
    page = client.get("/tasks").text
    assert f'href="/goals#goal-{gid}"' in page  # jump straight to the goal card
    assert "Learn woodworking" in page
    assert "from a goal plan" not in page       # named link replaces the vague label


def test_tasks_page_states_and_pills(client):
    page = client.get("/tasks").text
    assert "Nothing to plan yet" in page          # empty DB: no plan AND no tasks
    assert "No tasks yet" in page
    assert "No action items detected" in page
    from secondbrain.query import service

    client.post("/api/tasks", json={"title": "big rock", "value": 5,
                                    "due_date": service.local_today(),
                                    "estimate_minutes": 60})
    client.post("/api/tasks", json={"title": "meh chore", "value": 2})
    page = client.get("/tasks").text
    assert "No plan for today yet" in page        # tasks exist, plan doesn't
    assert ">Do<" in page and ">Eliminate<" in page   # Eisenhower pills
    assert "due today" in page
    client.post("/api/plan/today", json={"action": "propose"})
    page = client.get("/tasks").text
    assert "Proposed" in page and "Accept plan" in page
    assert "min planned" in page
    client.post("/api/plan/today", json={"action": "accept"})
    page = client.get("/tasks").text
    assert "Accepted" in page
    # completing a task lands it in the collapsed Completed section
    lone = client.post("/api/tasks", json={"title": "already finished"}).json()["id"]
    client.post(f"/api/tasks/{lone}/status", json={"status": "done"})
    assert "Completed (1)" in client.get("/tasks").text


def test_speaker_quality_and_reattribute(client):
    assert client.get("/api/speakers/quality").status_code == 200
    assert client.post("/api/speakers/reattribute").json()["relabeled"] == 0
    assert "Day ·" in client.get("/day").text


def test_speaker_quality_payload_context_fields(client):
    q = client.get("/api/speakers/quality").json()
    for key in ("speakers", "exemplars", "locked_segments", "low_confidence_segments",
                "attributed_segments", "unattributed_segments", "total_segments",
                "low_confidence_threshold"):
        assert key in q, key
    assert q["total_segments"] == q["attributed_segments"] + q["unattributed_segments"]


def test_name_speaker_validation(client, conn):
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    # empty / whitespace-only / oversized names rejected before any write
    assert client.post(f"/api/speakers/{sid}/name", json={"name": ""}).status_code == 400
    assert client.post(f"/api/speakers/{sid}/name", json={"name": "   "}).status_code == 400
    assert client.post(f"/api/speakers/{sid}/name", json={"name": "x" * 121}).status_code == 400
    assert conn.execute("SELECT name FROM speakers WHERE id=?", (sid,)).fetchone()["name"] is None
    # unknown speaker -> 404
    assert client.post("/api/speakers/99999/name", json={"name": "Zed"}).status_code == 404
    # a valid name is trimmed
    r = client.post(f"/api/speakers/{sid}/name", json={"name": "  Dana  "})
    assert r.status_code == 200 and r.json()["ok"] and r.json()["name"] == "Dana"
    assert conn.execute("SELECT name FROM speakers WHERE id=?", (sid,)).fetchone()["name"] == "Dana"


def test_merge_endpoint_validations_and_success(client, conn):
    from secondbrain.speaker import registry

    src = registry.create_unknown_speaker(conn)
    dst = registry.create_unknown_speaker(conn)
    af = models.insert_audio_file(
        conn, AudioFile(path="/b.flac", started_at="2026-06-16T10:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn, [Segment(t, af, 0.0, 1.0, "hello", start_at="2026-06-16T10:00:00.000Z")]
    )
    seg = conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"]
    registry.assign_segment_speaker(conn, seg, src, 0.9)
    # validation: unknown ids, self-merge, owner as source
    assert client.post("/api/speakers/merge", json={"src": 99999, "dst": dst}).status_code == 404
    assert client.post("/api/speakers/merge", json={"src": src, "dst": 99999}).status_code == 404
    assert client.post("/api/speakers/merge", json={"src": src, "dst": src}).status_code == 400
    owner = registry.get_or_create_owner(conn)
    assert client.post("/api/speakers/merge", json={"src": owner, "dst": dst}).status_code == 400
    # success relabels history and resolves src -> dst
    r = client.post("/api/speakers/merge", json={"src": src, "dst": dst})
    assert r.status_code == 200 and r.json()["relabeled_segments"] == 1
    assert registry.resolve_speaker_id(conn, src) == dst
    # merging the merged pair again is a no-op error, not a cycle
    assert client.post("/api/speakers/merge", json={"src": src, "dst": dst}).status_code == 400


def test_set_owner_endpoint_demotes_previous_owner(client, conn):
    from secondbrain.speaker import registry

    old = registry.get_or_create_owner(conn, "Me")
    sid = registry.create_unknown_speaker(conn)
    assert client.post("/api/speakers/99999/owner").status_code == 404
    assert client.post(f"/api/speakers/{sid}/owner").json()["ok"] is True
    rows = {r["id"]: r for r in conn.execute("SELECT id, is_owner, kind FROM speakers").fetchall()}
    assert rows[sid]["is_owner"] == 1 and rows[sid]["kind"] == "owner"
    # the demoted row keeps its history but is a regular known voice again
    assert rows[old]["is_owner"] == 0 and rows[old]["kind"] == "known"


def test_clip_caching_and_retention_410(client, conn, settings, tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    wav = tmp_path / "src.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    af = models.insert_audio_file(
        conn,
        AudioFile(path=str(wav), started_at="2026-06-16T09:00:00.000Z",
                  sample_rate=16000, status="transcribed"),
    )
    obs = registry.record_observation(
        conn, speaker_id=sid, audio_file_id=af, conversation_id=None,
        start_offset_s=0.0, end_offset_s=0.5, start_at="2026-06-16T09:00:00.000Z",
        confidence=0.9, embedding=[1.0, 0.0],
    )
    samples = client.get(f"/api/speakers/{sid}/samples").json()["samples"]
    assert samples and samples[0]["id"] == obs and samples[0]["audio_status"] == "transcribed"
    assert samples[0]["duration_s"] == 0.5  # additive field: set expectations pre-play
    # first fetch slices and caches; second serves the cache without re-slicing
    assert client.get(f"/api/speakers/{sid}/clip/{obs}").status_code == 200
    cached_all = list(settings.audio_processed_dir.glob(f"sample_{obs}_*.wav"))
    assert len(cached_all) == 1  # cache name is window-stamped (stale-proof)
    cached = cached_all[0]
    before = cached.stat().st_mtime_ns
    assert client.get(f"/api/speakers/{sid}/clip/{obs}").status_code == 200
    assert cached.stat().st_mtime_ns == before
    # a stale slice from an older window is replaced once the window changes
    conn.execute(
        "UPDATE speaker_observations SET end_offset_s=0.75 WHERE id=?", (obs,)
    )
    assert client.get(f"/api/speakers/{sid}/clip/{obs}").status_code == 200
    renamed = list(settings.audio_processed_dir.glob(f"sample_{obs}_*.wav"))
    assert len(renamed) == 1 and renamed[0] != cached  # old window's cache gone
    # retention sweeps the source -> 410, and the cached derived clip goes too
    conn.execute("UPDATE audio_files SET status='deleted' WHERE id=?", (af,))
    r = client.get(f"/api/speakers/{sid}/clip/{obs}")
    assert r.status_code == 410
    assert not list(settings.audio_processed_dir.glob(f"sample_{obs}_*.wav"))
    # the samples listing still reports the observation, flagged as deleted,
    # so the UI can explain instead of rendering a dead player
    samples = client.get(f"/api/speakers/{sid}/samples").json()["samples"]
    assert samples and samples[0]["audio_status"] == "deleted"


def test_speakers_page_full_surface(client, conn):
    from secondbrain.speaker import registry

    registry.get_or_create_owner(conn, "Me")
    uid = registry.create_unknown_speaker(conn)
    r = client.get("/speakers")
    assert r.status_code == 200
    # every core function of the capability is reachable from the page
    assert "Who is this?" in r.text
    assert "Known voices" in r.text
    assert "Profile quality" in r.text
    assert "Re-run attribution" in r.text
    assert 'id="merge-template"' in r.text  # merge affordance (duplicates)
    assert "Rename" in r.text               # known voices can be renamed
    assert "Hear this voice" in r.text      # clip playback
    # unknowns queue once and are NOT duplicated into Known voices
    assert f'id="spk-{uid}"' in r.text
    assert f'id="known-{uid}"' not in r.text


def test_speakers_page_fresh_empty_state(client):
    r = client.get("/speakers")
    assert r.status_code == 200
    assert "No voices yet" in r.text            # helpful fresh-DB empty state
    assert "Profile quality" not in r.text      # no dead controls without speakers
    assert 'id="merge-template"' not in r.text  # nobody to merge yet


def test_goals_api_crud(client):
    gid = client.post("/api/goals", json={"title": "Win Q3", "priority": 1}).json()["id"]
    titles = [g["title"] for g in client.get("/api/goals").json()["goals"]]
    assert "Win Q3" in titles
    assert client.post(f"/api/goals/{gid}/status", json={"status": "done"}).json()["ok"]


def test_brief_and_goals_pages_render(client):
    assert "Morning brief" in client.get("/brief").text
    assert "Goals" in client.get("/goals").text


def test_digest_generate_and_suggestions(client):
    r = client.post("/api/digest/generate", json={"kind": "daily", "force": True})
    assert r.status_code == 200
    assert client.get("/api/suggestions").status_code == 200


def _local_today():
    # digests/suggestions are filed under the machine-local calendar day
    from datetime import datetime

    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _seed_suggestion(conn, kind="commitment_owed", dedupe="h-test-1"):
    return conn.execute(
        "INSERT INTO suggestions (digest_date, kind, title, detail, importance, confidence,"
        " citations, dedupe_hash) VALUES (?, ?, 'You owe: send deck', 'Due soon', 0.8, 0.9,"
        " '[1]', ?)",
        (_local_today(), kind, dedupe),
    ).lastrowid


def test_digest_param_validation(client):
    assert client.get("/api/digest?kind=banana").status_code == 422
    assert client.get("/api/digest?date=not-a-date").status_code == 422
    assert client.get("/api/digest?date=2026-13-45").status_code == 422
    r = client.get("/api/digest?kind=weekly")  # legitimately absent → empty 200
    assert r.status_code == 200 and r.json() == {}
    assert client.post("/api/digest/generate", json={"kind": "banana"}).status_code == 422


def test_digest_dates_endpoint(client):
    r = client.get("/api/digest/dates")
    assert r.status_code == 200
    assert r.json() == {"daily": [], "weekly": []}
    client.post("/api/digest/generate", json={"kind": "weekly", "force": True})
    r = client.get("/api/digest/dates").json()
    assert r["weekly"] == [_local_today()] and r["daily"] == []


def test_digest_citations_resolved(client, conn):
    # summary cites the seeded segment (id 1) plus a dead id → only the real one resolves
    conn.execute(
        "INSERT INTO digests (digest_date, kind, summary_md) VALUES (?, 'daily', ?)",
        (_local_today(), "## Commitments\n- Send the deck [1] and [999]"),
    )
    d = client.get("/api/digest").json()
    assert [c["segment_id"] for c in d["citations"]] == [1]
    assert "onboarding" in d["citations"][0]["text"]


def test_suggestions_param_validation(client):
    assert client.get("/api/suggestions?status=banana").status_code == 422
    assert client.get("/api/suggestions?date=nope").status_code == 422
    r = client.get("/api/suggestions?status=done")
    assert r.status_code == 200 and r.json()["suggestions"] == []


def test_suggestion_action_validation(client, conn):
    sid = _seed_suggestion(conn)
    assert client.post(f"/api/suggestions/{sid}/action", json={"action": "explode"}).status_code == 422
    assert client.post("/api/suggestions/99999/action", json={"action": "done"}).status_code == 404


def test_suggestion_done_then_reopen(client, conn):
    sid = _seed_suggestion(conn)
    assert client.post(f"/api/suggestions/{sid}/action", json={"action": "done"}).json()["ok"]
    assert client.get("/api/suggestions").json()["suggestions"] == []
    done = client.get("/api/suggestions?status=done").json()["suggestions"]
    assert [s["id"] for s in done] == [sid]
    assert client.post(f"/api/suggestions/{sid}/action", json={"action": "reopen"}).json()["ok"]
    reopened = client.get("/api/suggestions").json()["suggestions"]
    assert [s["id"] for s in reopened] == [sid]
    # suggestion citations resolve to segment metadata for source links
    cites = client.get("/api/suggestions").json()["citations"]
    assert [c["segment_id"] for c in cites] == [1]


def test_suggestion_reopen_lifts_kind_snooze(client, conn):
    sid = _seed_suggestion(conn)
    client.post(f"/api/suggestions/{sid}/action", json={"action": "snooze"})
    row = conn.execute(
        "SELECT value FROM app_state WHERE key='proactive_snooze:commitment_owed'"
    ).fetchone()
    assert row is not None and row["value"]  # kind snoozed
    client.post(f"/api/suggestions/{sid}/action", json={"action": "reopen"})
    row = conn.execute(
        "SELECT value FROM app_state WHERE key='proactive_snooze:commitment_owed'"
    ).fetchone()
    assert row["value"] == ""  # snooze lifted
    assert client.get("/api/suggestions").json()["suggestions"][0]["status"] == "open"


def test_suggestion_vote_recorded_in_listing(client, conn):
    sid = _seed_suggestion(conn)
    assert client.post(f"/api/suggestions/{sid}/action", json={"action": "down"}).json()["ok"]
    s = client.get("/api/suggestions").json()["suggestions"][0]
    assert s["voted"] == "down"


def test_digest_regenerate_refreshes_created_at(client, conn):
    # On quiet days the regenerated summary can be byte-identical, so the
    # "Generated <when>" stamp is the only proof the regenerate did anything.
    conn.execute(
        "INSERT INTO digests (digest_date, kind, summary_md, created_at) "
        "VALUES (?, 'daily', 'old text', '2020-01-01T00:00:00.000Z')",
        (_local_today(),),
    )
    old_id = client.get("/api/digest").json()["id"]
    assert client.post("/api/digest/generate", json={"kind": "daily", "force": True}).status_code == 200
    d = client.get("/api/digest").json()
    assert d["created_at"] > "2020-01-02"  # stamp moved to the regeneration time
    assert d["id"] == old_id  # updated in place, not re-created


def test_snooze_hides_all_same_kind_items_and_undo_restores_them(client, conn):
    a = _seed_suggestion(conn, dedupe="h-a")
    b = _seed_suggestion(conn, dedupe="h-b")
    c = _seed_suggestion(conn, kind="connection", dedupe="h-c")
    client.post(f"/api/suggestions/{a}/action", json={"action": "snooze"})
    # the whole kind is hidden, as the button label promises; other kinds stay
    assert [s["id"] for s in client.get("/api/suggestions").json()["suggestions"]] == [c]
    snoozed = client.get("/api/suggestions?status=snoozed").json()["suggestions"]
    assert sorted(s["id"] for s in snoozed) == sorted([a, b])
    # undo (reopen) is symmetric: it lifts the snooze and restores the siblings
    client.post(f"/api/suggestions/{a}/action", json={"action": "reopen"})
    assert {s["id"] for s in client.get("/api/suggestions").json()["suggestions"]} == {a, b, c}


def test_reopen_of_done_item_does_not_cascade(client, conn):
    a = _seed_suggestion(conn, dedupe="h-a")
    b = _seed_suggestion(conn, dedupe="h-b")
    client.post(f"/api/suggestions/{a}/action", json={"action": "done"})
    client.post(f"/api/suggestions/{b}/action", json={"action": "done"})
    client.post(f"/api/suggestions/{a}/action", json={"action": "reopen"})
    assert [s["id"] for s in client.get("/api/suggestions").json()["suggestions"]] == [a]


def test_vote_can_be_switched_and_weight_neutralized(client, conn):
    import json as _json

    sid = _seed_suggestion(conn)

    def weight():
        row = conn.execute(
            "SELECT value FROM app_state WHERE key='proactive_feedback_weights'"
        ).fetchone()
        return _json.loads(row["value"])["commitment_owed"]

    client.post(f"/api/suggestions/{sid}/action", json={"action": "down"})
    assert weight() == pytest.approx(0.9)
    # a flip corrects the mis-click: old nudge neutralized, new one applied
    client.post(f"/api/suggestions/{sid}/action", json={"action": "up"})
    assert weight() == pytest.approx(1.05)
    assert client.get("/api/suggestions").json()["suggestions"][0]["voted"] == "up"
    # the stale 'down' no longer suppresses this item in future digests
    rows = conn.execute(
        "SELECT vote FROM suggestion_feedback WHERE suggestion_id=?", (sid,)
    ).fetchall()
    assert [r["vote"] for r in rows] == ["up"]
    # repeating the same vote is a no-op (no double bump)
    client.post(f"/api/suggestions/{sid}/action", json={"action": "up"})
    assert weight() == pytest.approx(1.05)


def test_generate_conflicts_while_run_in_flight(client, conn):
    from secondbrain.proactive import store as pstore

    pstore.mark_generating(conn, "daily")
    r = client.post("/api/digest/generate", json={"kind": "daily", "force": True})
    assert r.status_code == 409
    assert "already being written" in r.json()["detail"]
    st = client.get("/api/digest/status").json()
    assert st["generating"] is True and st["started_at"]
    # the marker is per-kind: a weekly run is not blocked by a daily one
    assert client.get("/api/digest/status?kind=weekly").json()["generating"] is False
    pstore.clear_generating(conn, "daily")
    assert client.post("/api/digest/generate", json={"kind": "daily", "force": True}).status_code == 200
    st = client.get("/api/digest/status").json()
    assert st["generating"] is False  # marker cleared after the run
    assert st["created_at"]  # today's digest stamp, for the resume poller


def test_stale_generation_marker_is_ignored(client, conn):
    # a marker left behind by a crashed run must not wedge generation forever
    from secondbrain.storage import state as app_state

    app_state.set_state(conn, "proactive_generating:daily", "2020-01-01T00:00:00.000Z")
    assert client.get("/api/digest/status").json()["generating"] is False
    assert client.post("/api/digest/generate", json={"kind": "daily"}).status_code == 200


def test_digest_status_param_validation(client):
    assert client.get("/api/digest/status?kind=banana").status_code == 422


def test_brief_page_embeds_generating_marker(client, conn):
    from secondbrain.proactive import store as pstore

    r = client.get("/brief")
    assert '"generating"' in r.text  # state payload always carries the key
    pstore.mark_generating(conn, "daily")
    started = conn.execute(
        "SELECT value FROM app_state WHERE key='proactive_generating:daily'"
    ).fetchone()["value"]
    assert started in client.get("/brief").text  # page can resume the progress line


def test_brief_page_kind_and_date_params(client):
    assert "Morning brief" in client.get("/brief").text
    assert "Weekly review" in client.get("/brief?kind=weekly").text
    # hand-mangled params fall back gracefully instead of erroring
    r = client.get("/brief?kind=banana&date=nope")
    assert r.status_code == 200 and "Morning brief" in r.text
    assert 'id="brief-state"' in r.text  # embedded state for the client renderer


def test_brief_page_renders_items_with_human_labels(client, conn):
    _seed_suggestion(conn)  # kind commitment_owed
    r = client.get("/brief").text
    assert "You owe: send deck" in r
    assert "You promised" in r  # human label, not the raw snake_case kind
    assert "Items (1)" in r
    assert "importance 0.8" not in r  # raw floats stay internal


# --- shared shell (base.html) -------------------------------------------------

HTML = {"accept": "text/html"}


def test_shared_shell_on_every_page(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (9, 'Dana', 'known', 0)")
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (11, 'project', 'Atlas')")
    for path in ("/", "/timeline", "/speakers", "/relationships", "/projects", "/graph",
                 "/chat", "/brief", "/goals", "/tasks", "/day", "/person/9", "/project/11"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert 'class="nav"' in r.text, f"{path} missing shared nav"
        assert '<main id="main"' in r.text, f"{path} missing main landmark"
        assert "Skip to content" in r.text, f"{path} missing skip link"
        assert "/static/app.js" in r.text, f"{path} missing shared JS helpers"
        assert 'aria-label="Primary"' in r.text, f"{path} nav unlabeled"


def test_nav_active_state(client):
    # note: 'aria-current="page">' (with >) matches rendered anchors only,
    # not the CSS attribute selector in the inline stylesheet.
    r = client.get("/goals")
    assert '<a href="/goals" aria-current="page">Goals</a>' in r.text
    assert r.text.count('aria-current="page">') == 1
    # prefix matching: subpages highlight their section
    r = client.get("/timeline/2026-06-16")
    assert '<a href="/timeline" aria-current="page">Timeline</a>' in r.text
    r = client.get("/day")
    assert '<a href="/day" aria-current="page">Day</a>' in r.text
    # home is exact-match only
    r = client.get("/")
    assert '<a href="/" aria-current="page">Home</a>' in r.text
    assert r.text.count('aria-current="page">') == 1


def test_nav_active_state_person_page(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (12, 'Dana', 'known', 0)")
    r = client.get("/person/12")
    assert '<a href="/speakers" aria-current="page">People</a>' in r.text


def test_error_pages_html_for_browsers_json_for_api(client):
    # API paths keep their exact JSON error contract (CLI / menu bar rely on it)
    r = client.get("/api/person/99999")
    assert r.status_code == 404 and r.json()["detail"] == "person not found"
    # non-browser clients (accept */*) on page routes also keep JSON
    r = client.get("/person/99999")
    assert r.status_code == 404 and r.json()["detail"] == "person not found"
    # browser navigations get a friendly page with the nav intact; the person
    # route names what's missing and points back at the People page
    r = client.get("/person/99999", headers=HTML)
    assert r.status_code == 404
    assert "Person not found" in r.text and 'class="nav"' in r.text
    assert 'href="/speakers"' in r.text
    r = client.get("/no-such-page", headers=HTML)
    assert r.status_code == 404 and "Page not found" in r.text
    # API 404s stay JSON even for browsers
    r = client.get("/api/person/99999", headers=HTML)
    assert r.json()["detail"] == "person not found"


def test_validation_error_pages(client):
    # default JSON validation shape preserved for non-HTML clients
    r = client.get("/person/notanumber")
    assert r.status_code == 422 and "detail" in r.json()
    # browsers get the error page
    r = client.get("/person/notanumber", headers=HTML)
    assert r.status_code == 422 and 'class="nav"' in r.text


def test_static_assets_and_favicon(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200 and "window.SB" in r.text
    r = client.get("/favicon.ico")
    assert r.status_code == 200 and "svg" in r.headers["content-type"]


def test_signout_control_requires_cookie_session(client):
    # With auth disabled there is no session to end, so the control must not
    # render even for non-loopback hosts like TestClient's "testclient" (the
    # cookie-authed rendering cases live in tests/test_auth.py).
    r = client.get("/goals")
    assert 'class="nav-signout"' not in r.text
    # The /logout JSON contract stays put for API clients regardless.
    assert client.post("/logout").json()["ok"] is True


# --- day view & transcript correction -----------------------------------------


def test_day_api_returns_segments_and_validates(client):
    r = client.get("/api/day/2026-06-16")
    assert r.status_code == 200
    body = r.json()
    assert body["day"] == "2026-06-16"
    assert len(body["segments"]) == 1
    seg = body["segments"][0]
    assert "conversation_id" in seg and "speaker_locked" in seg
    # malformed day -> 422 (JSON contract preserved for API clients)
    assert client.get("/api/day/not-a-date").status_code == 422


def test_day_page_navigation_and_content(client):
    r = client.get("/day?date=2026-06-16")
    assert r.status_code == 200
    assert 'href="/day?date=2026-06-15"' in r.text  # prev-day link
    assert 'href="/day?date=2026-06-17"' in r.text  # next-day link
    assert 'type="date"' in r.text  # jump-to-date picker
    assert "onboarding flow" in r.text  # the seeded segment renders
    assert 'href="/day"' in r.text  # Today shortcut / shared nav entry
    # /day without a date resolves to today: dated title, no dead heading
    r = client.get("/day")
    assert r.status_code == 200
    assert "<title>Day 2" in r.text


def test_day_page_invalid_date_falls_back_with_notice(client):
    r = client.get("/day?date=bogus")
    assert r.status_code == 200
    assert "look like a date" in r.text
    # extreme-but-parseable years can't overflow local-time math
    assert client.get("/day?date=0001-01-01").status_code == 200
    assert client.get("/day?date=9999-12-31").status_code == 200
    assert client.get("/api/day/0001-01-01").status_code == 422


def test_day_page_empty_state_jump_links(client):
    r = client.get("/day?date=1999-01-04")
    assert r.status_code == 200
    assert "Nothing recorded" in r.text
    assert "Next recorded day" in r.text  # jump link toward the data


def test_reassign_validates_segment_and_speaker(client, conn):
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    sp = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Alice','known')").lastrowid)
    # unknown target speaker -> 404, and nothing is written
    r = client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": 9999})
    assert r.status_code == 404 and r.json()["detail"] == "speaker not found"
    row = conn.execute(
        "SELECT speaker_id, speaker_locked FROM transcript_segments WHERE id=?", (seg_id,)
    ).fetchone()
    assert row["speaker_id"] is None and row["speaker_locked"] == 0
    # unknown segment -> 404
    r = client.post("/api/segments/99999/speaker", json={"speaker_id": sp})
    assert r.status_code == 404 and r.json()["detail"] == "segment not found"
    # valid correction -> attributed, locked, and the response says what happened
    r = client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": sp})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["speaker"] == "Alice" and body["locked"] is True
    assert body["learned"] is False  # seeded segment has no diarization embedding
    row = conn.execute(
        "SELECT speaker_id, speaker_locked, speaker_source, speaker_confidence "
        "FROM transcript_segments WHERE id=?",
        (seg_id,),
    ).fetchone()
    assert row["speaker_id"] == sp and row["speaker_locked"] == 1
    assert row["speaker_source"] == "user" and row["speaker_confidence"] == 1.0


def test_day_page_shows_locked_state_after_correction(client, conn):
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    sp = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Bea','known')").lastrowid)
    assert (
        client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": sp}).status_code == 200
    )
    r = client.get("/day?date=2026-06-16")
    assert "corrected by you" in r.text
    assert "Bea" in r.text


def test_reassign_rejects_opted_out_target(client, conn):
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    sp = int(
        conn.execute(
            "INSERT INTO speakers (name, kind, opted_out) VALUES ('Quiet','known',1)"
        ).lastrowid
    )
    r = client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": sp})
    assert r.status_code == 403 and r.json()["detail"] == "speaker opted out"
    row = conn.execute(
        "SELECT speaker_id, speaker_locked FROM transcript_segments WHERE id=?", (seg_id,)
    ).fetchone()
    assert row["speaker_id"] is None and row["speaker_locked"] == 0  # nothing written


def test_confirm_current_guess_locks_line(client, conn):
    # Confirming the CURRENT low-confidence guess (same speaker_id) is a
    # supported teaching action — it locks the line so the review queue drains.
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    sp = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Cara','known')").lastrowid)
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.2 WHERE id=?",
        (sp, seg_id),
    )
    r = client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": sp})
    assert r.status_code == 200 and r.json()["locked"] is True
    row = conn.execute(
        "SELECT speaker_id, speaker_locked, speaker_confidence, speaker_source "
        "FROM transcript_segments WHERE id=?",
        (seg_id,),
    ).fetchone()
    assert row["speaker_id"] == sp and row["speaker_locked"] == 1
    assert row["speaker_confidence"] == 1.0 and row["speaker_source"] == "user"


def test_locked_line_can_be_recorrected(client, conn):
    # A mis-correction must be recoverable: reassigning a locked segment works.
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    ann = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Ann','known')").lastrowid)
    ben = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Ben','known')").lastrowid)
    assert client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": ann}).status_code == 200
    r = client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": ben})
    assert r.status_code == 200 and r.json()["speaker"] == "Ben"
    row = conn.execute(
        "SELECT speaker_id, speaker_locked FROM transcript_segments WHERE id=?", (seg_id,)
    ).fetchone()
    assert row["speaker_id"] == ben and row["speaker_locked"] == 1
    # segment_count stays fresh on BOTH sides of the move
    counts = {
        r2["id"]: r2["segment_count"]
        for r2 in conn.execute("SELECT id, segment_count FROM speakers").fetchall()
    }
    assert counts[ann] == 0 and counts[ben] == 1


def test_day_page_offers_confirm_fix_and_change_affordances(client, conn):
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    dee = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Dee','known')").lastrowid)
    conn.execute("INSERT INTO speakers (name, kind) VALUES ('Eve','known')")
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.2 WHERE id=?",
        (dee, seg_id),
    )
    r = client.get("/day?date=2026-06-16")
    assert 'id="fix-template"' in r.text  # single cloned picker, not one per row
    assert "confirmbtn" in r.text and "It’s Dee" in r.text  # confirm-current affordance
    assert "(20%?)" in r.text  # confidence visible inline, not only in a tooltip
    assert "low-confidence guess, 20%" in r.text  # and exposed to screen readers
    assert f'data-current="{dee}"' in r.text
    # after locking, the row still offers a way out of a mis-correction
    assert client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": dee}).status_code == 200
    r = client.get("/day?date=2026-06-16")
    assert "corrected by you" in r.text and "Change…" in r.text


def test_day_page_disputes_sole_named_speaker_attribution(client, conn):
    # The uncorrectable dead-end: when the owner is the ONLY named person, a
    # high-confidence line wrongly attributed to them has no one else to move to
    # — the day view must still offer a way to dispute it ("Not me…") instead of
    # rendering an empty, inert fixwrap.
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    owner = int(
        conn.execute(
            "INSERT INTO speakers (name, kind, is_owner) VALUES ('Me','owner',1)"
        ).lastrowid
    )
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.93 WHERE id=?",
        (owner, seg_id),
    )
    r = client.get("/day?date=2026-06-16")
    assert r.status_code == 200
    # high confidence → no confirm affordance, and no other named target → the
    # picker isn't offered; the dispute affordance is what saves it.
    assert 'class="confirmbtn"' not in r.text
    assert "disputebtn" in r.text and "Not me…" in r.text
    assert 'data-owner-attr="1"' in r.text
    # and it routes somewhere real: name the person, or clear the line
    assert "no one else is named yet" in r.text.lower()


def test_unassign_clears_and_locks_segment(client, conn):
    # Clearing a disputed attribution locks the line as unattributed so
    # re-attribution never silently re-guesses the rejected voice.
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    owner = int(
        conn.execute(
            "INSERT INTO speakers (name, kind, is_owner) VALUES ('Me','owner',1)"
        ).lastrowid
    )
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.93, "
        "speaker_source='diarized' WHERE id=?",
        (owner, seg_id),
    )
    r = client.post(f"/api/segments/{seg_id}/unassign")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["speaker"] is None and body["locked"] is True
    row = conn.execute(
        "SELECT speaker_id, speaker_locked, speaker_confidence, speaker_source "
        "FROM transcript_segments WHERE id=?",
        (seg_id,),
    ).fetchone()
    assert row["speaker_id"] is None and row["speaker_locked"] == 1
    assert row["speaker_confidence"] is None and row["speaker_source"] == "user"
    # owner's segment count drops back to zero after the line leaves them
    assert conn.execute("SELECT segment_count FROM speakers WHERE id=?", (owner,)).fetchone()[0] == 0
    # unknown segment → 404 (bad input handled cleanly)
    assert client.post("/api/segments/99999/unassign").status_code == 404


def test_unassign_then_reattribution_leaves_line_unknown(client, conn):
    # A user-cleared line stays unattributed even if re-attribution runs: the
    # lock is the whole point of disputing a wrong-but-confident guess.
    from secondbrain.speaker import reattribute as ra

    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    owner = int(
        conn.execute(
            "INSERT INTO speakers (name, kind, is_owner) VALUES ('Me','owner',1)"
        ).lastrowid
    )
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.93 WHERE id=?",
        (owner, seg_id),
    )
    assert client.post(f"/api/segments/{seg_id}/unassign").status_code == 200
    ra.run_reattribution(conn, None)  # must not overwrite a locked, cleared line
    row = conn.execute(
        "SELECT speaker_id, speaker_locked FROM transcript_segments WHERE id=?", (seg_id,)
    ).fetchone()
    assert row["speaker_id"] is None and row["speaker_locked"] == 1


def test_day_page_reassign_uses_inline_confirm_not_native_dialog(client, conn):
    # The reassign teaching action must be confirmed in-page (pill/toast
    # language), never via a blocking native window.confirm().
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    dee = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Dee','known')").lastrowid)
    conn.execute("INSERT INTO speakers (name, kind) VALUES ('Eve','known')")
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.2 WHERE id=?",
        (dee, seg_id),
    )
    r = client.get("/day?date=2026-06-16")
    assert "window.confirm(" not in r.text  # never *calls* the native blocking dialog
    assert "inline-confirm" in r.text  # the in-page confirm affordance exists
    assert "function inlineConfirm" in r.text


def test_day_page_reassign_picker_marks_owner_for_first_person_copy(client, conn):
    # The picker template tags the owner option so the confirm dialog can read
    # "Move this line to you?" instead of the stilted "…to Me".
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    dee = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Dee','known')").lastrowid)
    conn.execute(
        "INSERT INTO speakers (name, kind, is_owner) VALUES ('Me','owner',1)"
    )
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.2 WHERE id=?",
        (dee, seg_id),
    )
    r = client.get("/day?date=2026-06-16")
    picker = r.text.split('id="fix-template"', 1)[1].split("</select>", 1)[0]
    assert 'data-owner="1"' in picker  # owner option flagged for first-person copy


def test_day_count_endpoint_is_cheap_and_validated(client):
    r = client.get("/api/day/2026-06-16/count")
    assert r.status_code == 200
    body = r.json()
    assert body["day"] == "2026-06-16" and body["count"] == 1
    assert "segments" not in body  # count-only, not the full payload
    # same strict date contract as /api/day
    assert client.get("/api/day/not-a-date/count").status_code == 422
    # unpadded but parseable normalizes like the other day routes
    assert client.get("/api/day/2026-6-16/count").json()["day"] == "2026-06-16"


def test_day_page_does_not_confirm_onto_unknown_clusters(client, conn):
    # A low-confidence guess onto an anonymous "Unknown #N" diarizer cluster is
    # not a real person: the page must NOT offer "✓ It's Unknown #1" (confirming
    # a placeholder identity contradicts the "who really spoke" framing), and the
    # reassign picker must not list unknown clusters as teaching targets.
    from secondbrain.speaker import registry

    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    unk = registry.create_unknown_speaker(conn)
    conn.execute("INSERT INTO speakers (name, kind) VALUES ('Ada','known')")
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.2 WHERE id=?",
        (unk, seg_id),
    )
    r = client.get("/day?date=2026-06-16")
    assert r.status_code == 200
    assert "It’s Unknown" not in r.text  # never confirm a placeholder identity
    assert 'class="confirmbtn"' not in r.text  # no confirm button rendered at all
    assert "Unknown #1" in r.text  # the guess is still shown, just not confirmable
    # the low-confidence cluster line still needs review and can be identified
    assert "needs-review" in r.text
    assert "Identify speaker…" in r.text
    # the picker (single cloned template) lists real people only, never clusters
    picker = r.text.split('id="fix-template"', 1)[1].split("</select>", 1)[0]
    assert ">Ada<" in picker and "Unknown #1" not in picker


def test_day_page_owner_confirm_reads_naturally(client, conn):
    # For the owner voice the confirm affordance must read "That's me", not the
    # stilted "It's Me".
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    conn.execute("INSERT INTO speakers (name, kind) VALUES ('Bo','known')")  # a picker target
    owner = int(
        conn.execute(
            "INSERT INTO speakers (name, kind, is_owner) VALUES ('Me','owner',1)"
        ).lastrowid
    )
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=0.2 WHERE id=?",
        (owner, seg_id),
    )
    r = client.get("/day?date=2026-06-16")
    assert "That’s me" in r.text and "It’s Me" not in r.text
    assert 'data-owner="1"' in r.text  # JS/analytics-free owner marker on the button


def test_day_page_renders_local_time_server_side(client, conn):
    # The transcript must render machine-local wall-clock immediately, so a
    # JS-disabled visitor (or the pre-hydration frame) never shows a raw UTC
    # slice or a "Times shown in UTC" note.
    from datetime import datetime

    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    ts = conn.execute(
        "SELECT start_at FROM transcript_segments WHERE id=?", (seg_id,)
    ).fetchone()["start_at"]
    local_hms = (
        datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    )
    r = client.get("/day?date=2026-06-16")
    assert f">{local_hms}</time>" in r.text  # localized, not the "09:00:00" UTC slice
    assert "Times shown in UTC" not in r.text
    assert 'data-ts="2026-06-16T09:00:00.000Z"' in r.text  # JS still refines per viewer TZ


def test_day_page_collapses_very_heavy_days(client, conn):
    af = models.insert_audio_file(
        conn, AudioFile(path="/b.flac", started_at="2026-03-03T09:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [
            Segment(t, af, i * 2.0, i * 2.0 + 1.0, f"line {i}",
                    start_at=f"2026-03-03T09:{i // 60:02d}:{i % 60:02d}.000Z")
            for i in range(320)
        ],
    )
    r = client.get("/day?date=2026-03-03")
    assert r.status_code == 200
    assert '<details class="convd">' in r.text  # collapsed for speed
    assert "Long day" in r.text
    # a normal day stays fully expanded
    r = client.get("/day?date=2026-06-16")
    assert '<details class="convd">' not in r.text


def test_day_accepts_nonpadded_date_and_normalizes(client):
    # strptime tolerates "2026-6-16"; the canonical form must be echoed back so
    # <input type=date value=…> (which rejects non-padded values) stays filled.
    r = client.get("/api/day/2026-6-16")
    assert r.status_code == 200
    body = r.json()
    assert body["day"] == "2026-06-16"
    assert len(body["segments"]) == 1
    r = client.get("/day?date=2026-6-16")
    assert r.status_code == 200
    assert 'value="2026-06-16"' in r.text  # date picker gets a valid value
    assert 'href="/day?date=2026-06-15"' in r.text  # prev/next math also padded
    assert 'href="/day?date=2026-06-17"' in r.text
    assert "onboarding flow" in r.text  # same day's data, canonical or not


def test_day_review_filter_css_actually_hides_rows(client):
    # Regression: rows are hidden via the hidden attribute, and .seg's
    # display:grid must never defeat it (that made the filter visually inert).
    r = client.get("/day?date=2026-06-16")
    assert ".seg[hidden]" in r.text  # page-level guard
    assert '[hidden]:not([hidden="until-found"]) { display: none !important; }' in r.text


def test_day_locked_badge_is_focus_target(client, conn):
    # applyCorrection() moves keyboard focus to the badge after re-correcting a
    # locked line; the server-rendered badge needs tabindex=-1 for that to work.
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    sp = int(conn.execute("INSERT INTO speakers (name, kind) VALUES ('Fay','known')").lastrowid)
    assert (
        client.post(f"/api/segments/{seg_id}/speaker", json={"speaker_id": sp}).status_code == 200
    )
    r = client.get("/day?date=2026-06-16")
    assert 'class="locked pill" tabindex="-1"' in r.text


def test_day_segments_carry_audio_status_and_play_control(client, conn):
    seg = client.get("/api/day/2026-06-16").json()["segments"][0]
    assert seg["audio_status"] == "recorded"  # additive field for playability
    r = client.get("/day?date=2026-06-16")
    assert 'class="playbtn"' in r.text
    assert f'data-clip="/api/segments/{seg["id"]}/clip"' in r.text
    # once retention sweeps the source audio, the play affordance disappears
    conn.execute("UPDATE audio_files SET status='deleted'")
    r = client.get("/day?date=2026-06-16")
    assert 'class="playbtn"' not in r.text
    assert client.get("/api/day/2026-06-16").json()["segments"][0]["audio_status"] == "deleted"


def test_segment_clip_validates_and_reports_expiry(client, conn):
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    # unknown segment -> 404
    assert client.get("/api/segments/99999/clip").status_code == 404
    # source file gone from disk (seeded path never existed) -> 410, not a 500
    r = client.get(f"/api/segments/{seg_id}/clip")
    assert r.status_code == 410
    assert "expired" in r.json()["detail"]
    # status says deleted -> 410 as well
    conn.execute("UPDATE audio_files SET status='deleted'")
    assert client.get(f"/api/segments/{seg_id}/clip").status_code == 410


def test_segment_clip_refuses_opted_out_speaker(client, conn):
    seg_id = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    sp = int(
        conn.execute(
            "INSERT INTO speakers (name, kind, opted_out) VALUES ('Quiet','known',1)"
        ).lastrowid
    )
    conn.execute("UPDATE transcript_segments SET speaker_id=? WHERE id=?", (sp, seg_id))
    r = client.get(f"/api/segments/{seg_id}/clip")
    assert r.status_code == 403 and r.json()["detail"] == "speaker opted out"


def test_segment_clip_serves_sliced_wav(client, conn, settings, tmp_path):
    sf = pytest.importorskip("soundfile")  # audio extra; skip on bare CI
    import numpy as np

    src = tmp_path / "seg-src.wav"
    sf.write(str(src), np.zeros(16000 * 3, dtype="float32"), 16000)  # 3 s of silence
    af = models.insert_audio_file(
        conn,
        AudioFile(path=str(src), started_at="2026-05-05T09:00:00.000Z", sample_rate=16000,
                  status="transcribed"),
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn, [Segment(t, af, 1.0, 2.0, "hello there", start_at="2026-05-05T09:00:01.000Z")]
    )
    seg_id = conn.execute("SELECT MAX(id) AS id FROM transcript_segments").fetchone()["id"]
    r = client.get(f"/api/segments/{seg_id}/clip")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    # cached for replays under a window-stamped name, in the retention-swept dir
    cached_all = list(settings.audio_processed_dir.glob(f"segclip_{seg_id}_*.wav"))
    assert len(cached_all) == 1
    audio, sr = sf.read(str(cached_all[0]))
    assert sr == 16000 and 0.9 <= len(audio) / sr <= 1.1  # the 1 s slice, not the file


def test_day_page_offers_refresh_pill_only_for_today(client):
    from secondbrain.query.service import local_today

    r = client.get(f"/day?date={local_today()}")
    assert 'id="fresh-note"' in r.text  # live-tail affordance on today
    r = client.get("/day?date=2026-06-16")
    assert 'id="fresh-note"' not in r.text  # past days are settled — no poll


# --- capture staleness / dashboard trust (home page) --------------------------


def test_status_capture_staleness_detection(client):
    # The seeded chunk started long ago while recording claims on → the status
    # API must say capture is stale instead of letting the UI assert "Recording".
    st = client.get("/api/status").json()
    assert st["recording"] is True
    assert st["last_capture_at"] == "2026-06-16T09:00:00.000Z"
    assert st["capture_stale"] is True
    assert st["capture_stale_for"]  # compact human label, e.g. "17d"

    # Paused → not recording, so there is no false "recording" claim to correct.
    client.post("/api/pause")
    st = client.get("/api/status").json()
    assert st["recording"] is False and st["capture_stale"] is False

    # Just resumed → grace window (the recorder needs a chunk-length to write
    # its first row); no false alarm the moment the owner resumes.
    client.post("/api/resume")
    st = client.get("/api/status").json()
    assert st["recording"] is True and st["capture_stale"] is False


def test_status_fresh_chunk_is_not_stale(client, conn):
    from secondbrain.storage.models import utcnow_iso

    models.insert_audio_file(
        conn, AudioFile(path="/fresh.flac", started_at=utcnow_iso(), sample_rate=16000)
    )
    st = client.get("/api/status").json()
    assert st["capture_stale"] is False
    assert st["last_capture_at"] is not None


def test_status_empty_corpus_is_not_stale(conn, settings):
    # A fresh install with nothing captured isn't a failure — no scary warning.
    st = TestClient(create_app(settings)).get("/api/status").json()
    assert st["last_capture_at"] is None
    assert st["capture_stale"] is False


def test_index_page_capture_stale_warning(client):
    import re

    html = client.get("/").text
    assert 'data-state="stale"' in html
    assert "no audio for" in html            # the pill says what's wrong…
    m = re.search(r'<a[^>]*id="pill-capture"[^>]*>', html)
    assert m and "hidden" not in m.group(0)  # …and the health link is visible
    assert "Check the microphone" in html


def test_index_page_capture_ok_hides_warning(client, conn):
    import re

    from secondbrain.storage.models import utcnow_iso

    models.insert_audio_file(
        conn, AudioFile(path="/fresh.flac", started_at=utcnow_iso(), sample_rate=16000)
    )
    html = client.get("/").text
    assert 'data-state="on"' in html
    m = re.search(r'<a[^>]*id="pill-capture"[^>]*>', html)
    assert m and "hidden" in m.group(0)


def test_index_page_shows_processing_jobs(client, conn):
    import re

    html = client.get("/").text
    m = re.search(r'<a[^>]*id="pill-active"[^>]*>', html)
    assert m and "hidden" in m.group(0)  # nothing queued → pill hidden
    conn.execute("INSERT INTO jobs (type, state) VALUES ('transcribe', 'pending')")
    conn.execute("INSERT INTO jobs (type, state) VALUES ('diarize_conversation', 'running')")
    html = client.get("/").text
    assert "2 jobs processing" in html
    m = re.search(r'<a[^>]*id="pill-active"[^>]*>', html)
    assert m and "hidden" not in m.group(0)
    st = client.get("/api/status").json()
    assert st["jobs"].get("pending") == 1 and st["jobs"].get("running") == 1


def test_stats_days_use_local_bucketing_and_pretty_caption(client):
    from secondbrain.query import service
    from secondbrain.query.api import _fmt_day

    day = service._local_day_of("2026-06-16T09:00:00.000Z")
    s = client.get("/api/stats").json()
    # Same local-day bucketing as segments_today / search groups / the day view.
    assert s["first_day"] == day and s["last_day"] == day
    # The dashboard caption renders it readable ("Jun 16, 2026"), not raw ISO.
    assert f"Captured {_fmt_day(day)}" in client.get("/").text


# --- search speaker filter -----------------------------------------------------


def test_search_speaker_filter(client, conn):
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, segment_count) "
        "VALUES (5, 'Dana', 'known', 0, 1)"
    )
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (6, 'Sam', 'known', 0)")
    conn.execute("UPDATE transcript_segments SET speaker_id=5")
    body = client.get("/api/search", params={"q": "onboarding", "speaker": 5}).json()
    assert body["speaker"] == 5 and len(body["results"]) == 1
    assert body["results"][0]["speaker"] == "Dana"
    # Someone else said nothing about onboarding.
    assert client.get(
        "/api/search", params={"q": "onboarding", "speaker": 6}
    ).json()["results"] == []
    # A merged id resolves to the canonical voice (stale bookmarks keep working).
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, merged_into) "
        "VALUES (7, 'D.', 'known', 0, 5)"
    )
    body = client.get("/api/search", params={"q": "onboarding", "speaker": 7}).json()
    assert body["speaker"] == 5 and len(body["results"]) == 1
    # Unknown ids are a 422, not a silent empty result.
    assert client.get("/api/search", params={"q": "onboarding", "speaker": 999}).status_code == 422
    assert client.get("/api/search", params={"q": "onboarding", "speaker": "abc"}).status_code == 422
    # '' means "no filter" (a cleared form field), same as omitting it.
    r = client.get("/api/search", params={"q": "onboarding", "speaker": ""})
    assert r.status_code == 200 and r.json()["count"] == 1 and r.json()["speaker"] is None


def test_search_filters_stay_exact_at_corpus_scale(client, conn):
    """Regression: speaker/date-filtered searches must return every match even
    when hundreds of better-ranked unfiltered hits exist (filters run in SQL
    before LIMIT, not as a post-filter over a capped candidate pool)."""
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, segment_count) "
        "VALUES (5, 'Dana', 'known', 0, 3)"
    )
    long_text = "the onboarding chat drifted across many other things entirely"
    for i in range(150):  # keyword-dense decoys owning the top bm25 ranks
        conn.execute(
            "INSERT INTO transcript_segments (transcript_id, audio_file_id, start_offset_s,"
            " end_offset_s, start_at, text) VALUES (1, 1, ?, ?, '2026-06-10T09:00:00.000Z',"
            " 'onboarding')",
            (float(i), i + 1.0),
        )
    for i in range(3):  # the hits the filter is actually after
        conn.execute(
            "INSERT INTO transcript_segments (transcript_id, audio_file_id, start_offset_s,"
            " end_offset_s, start_at, text, speaker_id) VALUES (1, 1, ?, ?,"
            " '2026-06-20T09:00:00.000Z', ?, 5)",
            (500.0 + i, 501.0 + i, f"{long_text} take {i}"),
        )
    body = client.get("/api/search", params={"q": "onboarding", "speaker": 5}).json()
    assert body["count"] == 3
    assert all(r["speaker"] == "Dana" for r in body["results"])
    from secondbrain.query import service

    day = service._local_day_of("2026-06-20T09:00:00.000Z")
    body = client.get("/api/search", params={"q": "onboarding", "since": day}).json()
    assert body["count"] == 3
    assert {r["day"] for r in body["results"]} == {day}


def test_index_page_speaker_filter_select(client, conn):
    html = client.get("/").text
    assert 'id="speaker"' not in html  # no attributed voices yet → no filter UI
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, segment_count) "
        "VALUES (5, 'Dana', 'known', 0, 1)"
    )
    conn.execute(
        "INSERT INTO speakers (id, display_label, kind, is_owner, segment_count) "
        "VALUES (6, 'Unknown #1', 'unknown', 0, 2)"
    )
    html = client.get("/").text
    assert 'id="speaker"' in html
    assert ">Anyone</option>" in html
    assert ">Dana</option>" in html
    assert ">Unknown #1</option>" in html  # unnamed voices use their label


def test_res_status_focus_target_exists(client):
    # End-of-pagination focus target: the results summary must be focusable so
    # keyboard users aren't dropped to <body> when "Show more" disappears.
    html = client.get("/").text
    assert 'id="res-status"' in html
    assert 'tabindex="-1"' in html


# --- cross-origin write protection ----------------------------------------------


def test_cross_origin_writes_are_rejected(client):
    # Loopback bypasses auth, so a drive-by page could otherwise POST to the
    # recorder from the owner's browser. Cross-origin writes are refused…
    r = client.post("/api/pause", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.get("/api/status").json()["paused"] is False  # nothing flipped
    # …including sandboxed-iframe tricks (Origin: null)…
    assert client.post("/api/resume", headers={"Origin": "null"}).status_code == 403
    # …while the app's own JS (same-origin, or the SB.api custom header) and
    # header-less native clients (CLI, menu bar, curl) keep working.
    assert client.post("/api/pause", headers={"Origin": "http://testserver"}).status_code == 200
    assert client.post(
        "/api/resume", headers={"Origin": "https://odd.proxy", "X-SecondBrain": "1"}
    ).status_code == 200
    assert client.post("/api/resume").status_code == 200
    assert client.get("/api/status").json()["recording"] is True
    # Cross-origin *reads* are unaffected (the browser walls off the response).
    assert client.get(
        "/api/status", headers={"Origin": "https://evil.example"}
    ).status_code == 200
