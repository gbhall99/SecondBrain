"""Decision tracking: supersede versioning, list/count/search, API + page."""

import pytest
from fastapi.testclient import TestClient

from secondbrain.knowledge import graph
from secondbrain.query import service
from secondbrain.query.api import create_app
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment

WHEN = "2026-06-16T09:00:00.000Z"
LATER = "2026-07-02T10:00:00.000Z"


@pytest.fixture
def client(conn, settings):
    return TestClient(create_app(settings))


def _node(conn, name, type="project"):
    return graph.create_node(conn, type=type, name=name, embedding=None,
                             confidence=0.9, extraction_id=None)


def _decision(conn, node, text, when=WHEN, segs=None, conversation_id=None):
    return graph.upsert_edge(
        conn, src_node_id=node, dst_node_id=None, predicate="decision",
        kind="decision", object_text=text, confidence=0.9, when=when,
        source_segment_ids=segs or [], conversation_id=conversation_id,
    )


def _seg(conn, text, start_at=WHEN):
    af = models.insert_audio_file(
        conn, AudioFile(path=f"/tmp/{start_at}.flac", started_at=start_at, sample_rate=16000)
    )
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 2.0, text, start_at=start_at)])
    return conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"]


# --- supersede versioning ------------------------------------------------------


def test_similar_decision_on_same_subject_supersedes(conn):
    atlas = _node(conn, "Atlas")
    old = _decision(conn, atlas, "ship the Atlas beta on March 3")
    new = _decision(conn, atlas, "ship the Atlas beta on March 10", when=LATER)
    row = conn.execute(
        "SELECT valid, superseded_by FROM kg_edges WHERE id=?", (old,)
    ).fetchone()
    # Superseded decision keeps valid=1 (history stays visible), but points at
    # its replacement.
    assert row["valid"] == 1 and row["superseded_by"] == new
    assert conn.execute(
        "SELECT superseded_by FROM kg_edges WHERE id=?", (new,)
    ).fetchone()["superseded_by"] is None


def test_unrelated_decision_on_same_subject_coexists(conn):
    atlas = _node(conn, "Atlas")
    a = _decision(conn, atlas, "ship the beta on March 3")
    b = _decision(conn, atlas, "hire two contractors for QA", when=LATER)
    rows = conn.execute(
        "SELECT id, superseded_by FROM kg_edges WHERE id IN (?, ?)", (a, b)
    ).fetchall()
    assert all(r["superseded_by"] is None for r in rows)


def test_similar_decision_on_other_subject_does_not_supersede(conn):
    atlas, borealis = _node(conn, "Atlas"), _node(conn, "Borealis")
    a = _decision(conn, atlas, "ship the beta on March 3")
    _decision(conn, borealis, "ship the beta on March 10", when=LATER)
    assert conn.execute(
        "SELECT superseded_by FROM kg_edges WHERE id=?", (a,)
    ).fetchone()["superseded_by"] is None


def test_identical_decision_rehear_merges_instead_of_superseding(conn):
    atlas = _node(conn, "Atlas")
    a = _decision(conn, atlas, "ship the beta on March 3", segs=[1])
    b = _decision(conn, atlas, "ship the beta on March 3", when=LATER, segs=[2])
    assert a == b  # same edge reused, citations merged — no fake supersede


def test_superseded_decision_can_be_superseded_again_via_chain(conn):
    atlas = _node(conn, "Atlas")
    e1 = _decision(conn, atlas, "launch pricing at 20 dollars")
    e2 = _decision(conn, atlas, "launch pricing at 25 dollars", when=LATER)
    e3 = _decision(conn, atlas, "launch pricing at 30 dollars",
                   when="2026-08-01T09:00:00.000Z")
    sup = {r["id"]: r["superseded_by"] for r in conn.execute(
        "SELECT id, superseded_by FROM kg_edges").fetchall()}
    assert sup[e1] == e2 and sup[e2] == e3 and sup[e3] is None


def test_decision_similarity_token_fallback():
    # No embedder in tests → token-overlap fallback.
    assert graph.decision_similarity(
        "ship the beta on March 3", "ship the beta on March 10"
    ) >= graph.DECISION_SUPERSEDE_THRESHOLD
    assert graph.decision_similarity(
        "ship the beta on March 3", "hire two QA contractors"
    ) < graph.DECISION_SUPERSEDE_THRESHOLD
    assert graph.decision_similarity("", "anything") == 0.0


# --- kg_edges_fts sync ---------------------------------------------------------


def test_edge_fts_stays_in_sync_with_triggers(conn):
    atlas = _node(conn, "Atlas")
    e = _decision(conn, atlas, "migrate billing to the new provider")
    hit = conn.execute(
        "SELECT rowid FROM kg_edges_fts WHERE kg_edges_fts MATCH ?", ('"billing"',)
    ).fetchall()
    assert [r["rowid"] for r in hit] == [e]
    conn.execute("DELETE FROM kg_edges WHERE id=?", (e,))
    assert conn.execute(
        "SELECT rowid FROM kg_edges_fts WHERE kg_edges_fts MATCH ?", ('"billing"',)
    ).fetchall() == []


# --- service: list/count/search ------------------------------------------------


def test_list_decisions_newest_first_with_provenance(conn, settings):
    seg = _seg(conn, "we decided to ship the beta on March 3")
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (7, ?, 'diarized')",
        (WHEN,),
    )
    atlas = _node(conn, "Atlas")
    old = _decision(conn, atlas, "ship the beta on March 3", segs=[seg],
                    conversation_id=7)
    new = _decision(conn, atlas, "ship the beta on March 10", when=LATER)
    items = service.list_decisions(conn, settings=settings)
    assert [d["edge_id"] for d in items] == [new, old]  # newest first
    newest, oldest = items
    assert oldest["node"]["label"] == "Atlas" and oldest["node"]["type"] == "project"
    assert oldest["is_superseded"] is True
    assert oldest["superseded_by"] == new
    assert oldest["superseded_by_text"] == "ship the beta on March 10"
    assert newest["is_superseded"] is False
    assert newest["supersedes"] == [
        {"edge_id": old, "object_text": "ship the beta on March 3"}
    ]
    # Provenance: cited segment + local day + quote text.
    assert oldest["source_seg"] == seg and oldest["source_day"]
    assert oldest["quotes"][0]["segment_id"] == seg
    assert service.count_decisions(conn) == 2


def test_list_decisions_filters(conn, settings):
    atlas, hr = _node(conn, "Atlas"), _node(conn, "Hiring", type="topic")
    a = _decision(conn, atlas, "ship the beta on March 3", when=WHEN)
    b = _decision(conn, hr, "hire two contractors", when=LATER)
    # q: full-text over decision text
    got = service.list_decisions(conn, q="contractors", settings=settings)
    assert [d["edge_id"] for d in got] == [b]
    assert service.count_decisions(conn, q="contractors") == 1
    # node filter
    got = service.list_decisions(conn, node_id=atlas, settings=settings)
    assert [d["edge_id"] for d in got] == [a]
    # date filters (local days around the two decisions)
    got = service.list_decisions(conn, since="2026-07-01", settings=settings)
    assert [d["edge_id"] for d in got] == [b]
    got = service.list_decisions(conn, until="2026-06-30", settings=settings)
    assert [d["edge_id"] for d in got] == [a]
    # limit/offset paging
    assert len(service.list_decisions(conn, limit=1, settings=settings)) == 1
    assert [d["edge_id"] for d in
            service.list_decisions(conn, limit=1, offset=1, settings=settings)] == [a]


def test_list_decisions_excludes_invalidated(conn, settings):
    atlas = _node(conn, "Atlas")
    e = _decision(conn, atlas, "ship the beta")
    assert service.invalidate_edge(conn, e) is True
    assert service.list_decisions(conn, settings=settings) == []
    assert service.count_decisions(conn) == 0
    assert service.revalidate_edge(conn, e) is True
    assert service.count_decisions(conn) == 1


def test_search_edges_covers_decisions_and_commitments(conn, settings):
    atlas = _node(conn, "Atlas")
    d = _decision(conn, atlas, "migrate billing to the new provider")
    c = graph.upsert_edge(conn, src_node_id=atlas, dst_node_id=None,
                          predicate="action_item", kind="action_item",
                          object_text="send the billing report", confidence=0.8,
                          when=LATER)
    got = service.search_edges(conn, "billing", settings=settings)
    assert {g["edge_id"] for g in got} == {d, c}
    assert all(g["node_label"] == "Atlas" for g in got)
    kinds = {g["edge_id"]: g["kind"] for g in got}
    assert kinds[d] == "decision" and kinds[c] == "action_item"
    # facts are not in the decisions scope
    graph.upsert_edge(conn, src_node_id=atlas, dst_node_id=None, predicate="uses",
                      kind="fact", object_text="billing provider X", when=WHEN)
    assert {g["kind"] for g in service.search_edges(conn, "billing",
                                                    settings=settings)} == {
        "decision", "action_item"}


# --- invalidate / revalidate generalization ------------------------------------


def test_invalidate_edge_covers_facts_and_mentions(conn):
    atlas = _node(conn, "Atlas")
    f = graph.upsert_edge(conn, src_node_id=atlas, dst_node_id=None, predicate="uses",
                          kind="fact", object_text="Postgres", when=WHEN)
    m = graph.upsert_edge(conn, src_node_id=atlas, dst_node_id=None, predicate=None,
                          kind="mention", object_text="mentioned Atlas", when=WHEN)
    for e in (f, m):
        assert service.invalidate_edge(conn, e) is True
        assert conn.execute("SELECT valid FROM kg_edges WHERE id=?", (e,)).fetchone()[0] == 0
        assert service.revalidate_edge(conn, e) is True
        assert conn.execute("SELECT valid FROM kg_edges WHERE id=?", (e,)).fetchone()[0] == 1
    assert service.invalidate_edge(conn, 99999) is False
    assert service.revalidate_edge(conn, 99999) is False


# --- re-extraction -------------------------------------------------------------


def test_reextract_conversation_clears_and_requeues(conn, settings):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status, knowledge_status) "
        "VALUES (7, ?, 'diarized', 'extracted')",
        (WHEN,),
    )
    atlas = _node(conn, "Atlas")
    kept = _decision(conn, atlas, "wrong decision", conversation_id=7)
    service.invalidate_edge(conn, kept)  # user said this one is wrong — keep it
    _decision(conn, atlas, "real decision", when=LATER, conversation_id=7)
    conn.execute(
        "INSERT INTO knowledge_extractions (conversation_id, chunk_index) VALUES (7, 0)"
    )

    res = service.reextract_conversation(conn, 7)
    assert res is not None and res["cleared_edges"] == 1 and res["job_id"] is not None
    # Valid extraction artifacts are gone; the user-invalidated edge stays.
    ids = [r["id"] for r in conn.execute("SELECT id FROM kg_edges").fetchall()]
    assert ids == [kept]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM knowledge_extractions WHERE conversation_id=7"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT knowledge_status FROM conversations WHERE id=7"
    ).fetchone()[0] == "pending"
    job = conn.execute(
        "SELECT type, state FROM jobs WHERE id=?", (res["job_id"],)
    ).fetchone()
    assert job["type"] == "extract_knowledge" and job["state"] == "pending"
    assert service.reextract_conversation(conn, 99999) is None


def test_reextract_unsupersedes_survivors(conn, settings):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (7, ?, 'diarized')",
        (LATER,),
    )
    atlas = _node(conn, "Atlas")
    old = _decision(conn, atlas, "ship the beta on March 3")  # no conversation
    _decision(conn, atlas, "ship the beta on March 10", when=LATER, conversation_id=7)
    assert conn.execute(
        "SELECT superseded_by FROM kg_edges WHERE id=?", (old,)
    ).fetchone()[0] is not None
    service.reextract_conversation(conn, 7)
    # The superseder is gone → the earlier decision is current again.
    assert conn.execute(
        "SELECT superseded_by FROM kg_edges WHERE id=?", (old,)
    ).fetchone()[0] is None


# --- API + page ----------------------------------------------------------------


def test_api_decisions_lists_and_filters(client, conn):
    atlas = _node(conn, "Atlas")
    old = _decision(conn, atlas, "ship the beta on March 3")
    new = _decision(conn, atlas, "ship the beta on March 10", when=LATER)
    body = client.get("/api/decisions").json()
    assert [d["edge_id"] for d in body["decisions"]] == [new, old]
    assert body["total"] == 2
    d_old = body["decisions"][1]
    assert d_old["is_superseded"] is True and d_old["superseded_by"] == new
    # q filter goes through FTS
    body = client.get("/api/decisions", params={"q": "March"}).json()
    assert body["total"] == 2
    body = client.get("/api/decisions", params={"q": "zebra"}).json()
    assert body["decisions"] == [] and body["total"] == 0
    # node filter + paging params
    body = client.get("/api/decisions", params={"node": atlas, "limit": 1}).json()
    assert len(body["decisions"]) == 1 and body["total"] == 2
    # validation: bad dates / node ids are 422, not silent empties
    assert client.get("/api/decisions", params={"since": "notadate"}).status_code == 422
    assert client.get("/api/decisions", params={"node": "abc"}).status_code == 422
    assert client.get("/api/decisions", params={"node": str(10**20)}).status_code == 422
    assert client.get("/api/decisions", params={"offset": -1}).status_code == 422


def test_decisions_page_renders_list_and_badges(client, conn):
    atlas = _node(conn, "Atlas")
    _decision(conn, atlas, "ship the beta on March 3")
    _decision(conn, atlas, "ship the beta on March 10", when=LATER)
    html = client.get("/decisions").text
    assert "ship the beta on March 10" in html
    assert "ship the beta on March 3" in html
    assert "superseded" in html and "supersedes →" in html
    assert 'class="nav"' in html  # shared shell
    assert "/project/" in html or "/graph#node=" in html  # about-node chip
    # search box + date filters present
    assert 'name="q"' in html and 'name="since"' in html and 'name="until"' in html
    # filtered views work server-side
    html = client.get("/decisions", params={"q": "zebra"}).text
    assert "No decisions match" in html


def test_decisions_page_empty_state(client):
    html = client.get("/decisions").text
    assert "No decisions yet" in html


def test_api_search_decisions_scope(client, conn):
    seg = _seg(conn, "let's migrate billing next sprint")
    atlas = _node(conn, "Atlas")
    _decision(conn, atlas, "migrate billing to the new provider", segs=[seg])
    body = client.get(
        "/api/search", params={"q": "billing", "scope": "decisions"}
    ).json()
    assert body["scope"] == "decisions" and body["count"] == 1
    hit = body["results"][0]
    assert hit["kind"] == "decision" and hit["node_label"] == "Atlas"
    assert hit["source_seg"] == seg and hit["source_day"]
    # unknown scope rejected
    assert client.get(
        "/api/search", params={"q": "x", "scope": "banana"}
    ).status_code == 422


def test_graph_edge_invalidate_endpoints(client, conn):
    atlas = _node(conn, "Atlas")
    f = graph.upsert_edge(conn, src_node_id=atlas, dst_node_id=None, predicate="uses",
                          kind="fact", object_text="Postgres", when=WHEN)
    assert client.post(f"/api/graph/edges/{f}/invalidate").json()["ok"] is True
    assert conn.execute("SELECT valid FROM kg_edges WHERE id=?", (f,)).fetchone()[0] == 0
    assert client.post(f"/api/graph/edges/{f}/revalidate").json()["ok"] is True
    assert conn.execute("SELECT valid FROM kg_edges WHERE id=?", (f,)).fetchone()[0] == 1
    assert client.post("/api/graph/edges/99999/invalidate").status_code == 404
    assert client.post(f"/api/graph/edges/{10**20}/invalidate").status_code == 422


def test_api_reextract_endpoint(client, conn):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (7, ?, 'diarized')",
        (WHEN,),
    )
    atlas = _node(conn, "Atlas")
    _decision(conn, atlas, "old extraction", conversation_id=7)
    body = client.post("/api/conversations/7/reextract").json()
    assert body["ok"] is True and body["cleared_edges"] == 1
    assert client.post("/api/conversations/99999/reextract").status_code == 404
    assert client.post(f"/api/conversations/{10**20}/reextract").status_code == 422
