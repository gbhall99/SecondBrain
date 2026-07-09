import json

import pytest
from fastapi.testclient import TestClient

from secondbrain.goals import link, store
from secondbrain.knowledge import graph
from secondbrain.llm.client import MockLLM
from secondbrain.query.api import create_app


def test_goal_crud(conn, settings):
    gid = store.create_goal(conn, title="Ship pricing v2", description="revamp pricing",
                            priority=1, settings=settings)
    assert store.get_goal(conn, gid)["goal"]["title"] == "Ship pricing v2"
    store.update_goal(conn, gid, settings=settings, title="Ship pricing v3")
    assert store.get_goal(conn, gid)["goal"]["title"] == "Ship pricing v3"
    store.set_status(conn, gid, "done")
    assert store.list_goals(conn, status="done")[0]["id"] == gid
    store.delete_goal(conn, gid)
    assert store.get_goal(conn, gid) is None


def test_relink_goal_keyword_path(conn, settings):
    # semantic disabled in conftest → deterministic keyword linking
    match = graph.create_node(conn, type="topic", name="pricing strategy", embedding=None,
                              confidence=0.9, extraction_id=None)
    graph.create_node(conn, type="topic", name="lunch menu", embedding=None,
                      confidence=0.9, extraction_id=None)
    gid = store.create_goal(conn, title="pricing strategy", settings=settings)
    n = link.relink_goal(conn, gid, settings)
    assert n == 1
    links = store.get_goal(conn, gid)["links"]
    assert links[0]["ref_id"] == match and links[0]["relation"] == "related"


def test_relink_is_idempotent(conn, settings):
    graph.create_node(conn, type="project", name="atlas", embedding=None,
                      confidence=0.9, extraction_id=None)
    gid = store.create_goal(conn, title="atlas", settings=settings)
    link.relink_goal(conn, gid, settings)
    link.relink_goal(conn, gid, settings)  # second run must not duplicate
    assert len(store.get_goal(conn, gid)["links"]) == 1


def test_list_goals_orders_by_status_rank_then_priority_then_date(conn, settings):
    def add(title, status, priority=2, target=None):
        gid = store.create_goal(conn, title=title, priority=priority,
                                target_date=target, settings=settings)
        store.set_status(conn, gid, status)
        return gid

    add("done", "done")
    add("dropped", "dropped")
    add("paused", "paused")
    add("active undated", "active", priority=1)
    add("active dated", "active", priority=1, target="2026-07-10")
    add("active low", "active", priority=3)
    titles = [g["title"] for g in store.list_goals(conn)]
    # active first (dated before undated within a priority), then paused,
    # then done, then dropped — a plain ORDER BY status would bury paused.
    assert titles == ["active dated", "active undated", "active low",
                      "paused", "done", "dropped"]


# --- API surface --------------------------------------------------------------


@pytest.fixture
def client(conn, settings):
    return TestClient(create_app(settings))


_PLAN = {
    "milestones": [
        {"title": "Set up tooling", "tasks": [
            {"title": "Pick an ESP", "estimate_minutes": 60, "effort": 2, "value": 4},
            {"title": "Create signup form", "estimate_minutes": 45, "effort": 2, "value": 3},
        ]},
        {"title": "Write first issue", "tasks": [
            {"title": "Draft outline", "estimate_minutes": 30, "effort": 3, "value": 5},
        ]},
    ]
}


def test_api_goals_survives_embedding_blob(client, conn, settings):
    # Regression: raw float32 embedding bytes must never reach the JSON encoder
    # (FastAPI raises UnicodeDecodeError on non-UTF-8 bytes → 500).
    gid = store.create_goal(conn, title="Ship pricing v2", settings=settings)
    conn.execute("UPDATE goals SET embedding=? WHERE id=?",
                 (b"\x00\x01\xfe\xff" * 96, gid))
    conn.commit()
    r = client.get("/api/goals")
    assert r.status_code == 200
    goals = r.json()["goals"]
    assert goals[0]["title"] == "Ship pricing v2"
    assert "embedding" not in goals[0]
    single = client.get(f"/api/goals/{gid}")
    assert single.status_code == 200
    assert "embedding" not in single.json()["goal"]
    assert client.get("/goals").status_code == 200  # page renders too


def test_api_goal_detail_includes_links_and_progress(client, conn, settings):
    nid = graph.create_node(conn, type="topic", name="pricing strategy", embedding=None,
                            confidence=0.9, extraction_id=None)
    conn.commit()
    r = client.post("/api/goals", json={"title": "pricing strategy", "priority": 1})
    assert r.status_code == 200
    gid = r.json()["id"]
    detail = client.get(f"/api/goals/{gid}").json()
    assert detail["goal"]["tasks_total"] == 0 and detail["goal"]["tasks_done"] == 0
    assert detail["links"] and detail["links"][0]["ref_id"] == nid
    assert client.get("/api/goals/99999").status_code == 404


def test_goal_links_carry_display_info(client, conn, settings):
    # Auto-link evidence must be renderable: each link resolves a human label,
    # its type, and (for edges) the source entity for deep-linking.
    nid = graph.create_node(conn, type="topic", name="pricing strategy", embedding=None,
                            confidence=0.9, extraction_id=None)
    eid = conn.execute(
        "INSERT INTO kg_edges (src_node_id, kind, object_text, valid) "
        "VALUES (?, 'decision', 'ship the pricing strategy deck', 1)",
        (nid,),
    ).lastrowid
    conn.commit()
    gid = client.post("/api/goals", json={"title": "pricing strategy"}).json()["id"]
    link.link_advance(conn, gid, eid)
    conn.commit()
    links = client.get(f"/api/goals/{gid}").json()["links"]
    by_kind = {link_["kind"]: link_ for link_ in links}
    assert by_kind["node"]["label"] == "pricing strategy"
    assert by_kind["node"]["ref_type"] == "topic"
    assert by_kind["edge"]["label"] == "ship the pricing strategy deck"
    assert by_kind["edge"]["ref_type"] == "decision"
    assert by_kind["edge"]["src_node_id"] == nid
    assert by_kind["edge"]["relation"] == "advances"
    # original contract keys are still present on every link
    assert {"kind", "ref_id", "relation", "score"} <= set(by_kind["node"])


def test_api_goals_list_reports_links_count(client, conn, settings):
    graph.create_node(conn, type="topic", name="pricing strategy", embedding=None,
                      confidence=0.9, extraction_id=None)
    conn.commit()
    client.post("/api/goals", json={"title": "pricing strategy"})
    client.post("/api/goals", json={"title": "unrelated thing"})
    goals = {g["title"]: g for g in client.get("/api/goals").json()["goals"]}
    assert goals["pricing strategy"]["links_count"] == 1
    assert goals["unrelated thing"]["links_count"] == 0
    page = client.get("/goals")
    assert "1 knowledge link" in page.text            # evidence disclosure
    assert "Nothing linked yet" in page.text          # honest empty state


def test_api_create_goal_validates_input(client):
    assert client.post("/api/goals", json={"title": "   "}).status_code == 422
    assert client.post("/api/goals", json={"title": "ok", "priority": 7}).status_code == 422
    assert client.post("/api/goals",
                       json={"title": "ok", "target_date": "banana"}).status_code == 422
    r = client.post("/api/goals", json={"title": "  padded  ", "target_date": "2026-12-01"})
    assert r.status_code == 200
    goal = client.get(f"/api/goals/{r.json()['id']}").json()["goal"]
    assert goal["title"] == "padded"           # stored trimmed
    assert goal["target_date"] == "2026-12-01"


def test_api_goals_status_filter(client):
    client.post("/api/goals", json={"title": "a"})
    gid = client.post("/api/goals", json={"title": "b"}).json()["id"]
    client.post(f"/api/goals/{gid}/status", json={"status": "paused"})
    assert [g["title"] for g in client.get("/api/goals?status=paused").json()["goals"]] == ["b"]
    assert client.get("/api/goals?status=weird").status_code == 422


def test_api_status_and_delete_validate(client):
    gid = client.post("/api/goals", json={"title": "temp"}).json()["id"]
    assert client.post(f"/api/goals/{gid}/status", json={"status": "sideways"}).status_code == 422
    assert client.post("/api/goals/99999/status", json={"status": "done"}).status_code == 404
    assert client.delete("/api/goals/99999").status_code == 404
    assert client.delete(f"/api/goals/{gid}").status_code == 200
    assert client.get(f"/api/goals/{gid}").status_code == 404


def test_api_patch_goal_edits_in_place(client, conn):
    gid = client.post("/api/goals", json={
        "title": "Ship v1", "description": "old", "target_date": "2026-08-01",
    }).json()["id"]
    r = client.patch(f"/api/goals/{gid}", json={"title": "Ship v2", "priority": 1})
    assert r.status_code == 200
    assert r.json()["goal"]["title"] == "Ship v2"
    assert r.json()["goal"]["priority"] == 1
    # empty strings clear optional fields; omitted fields stay put
    r = client.patch(f"/api/goals/{gid}", json={"description": "", "target_date": ""})
    goal = r.json()["goal"]
    assert goal["description"] is None and goal["target_date"] is None
    assert goal["title"] == "Ship v2"
    assert client.patch(f"/api/goals/{gid}", json={}).status_code == 422
    assert client.patch(f"/api/goals/{gid}", json={"title": "  "}).status_code == 422
    assert client.patch(f"/api/goals/{gid}", json={"priority": 9}).status_code == 422
    assert client.patch(f"/api/goals/{gid}", json={"status": "odd"}).status_code == 422
    assert client.patch(f"/api/goals/{gid}", json={"status": "paused"}).status_code == 200
    assert client.patch("/api/goals/99999", json={"title": "x"}).status_code == 404


def test_api_decompose_and_accept_plan(client, conn, monkeypatch):
    from secondbrain.tasks import decompose as dmod

    monkeypatch.setattr(dmod, "get_llm",
                        lambda settings: MockLLM(responses=[json.dumps(_PLAN)]))
    assert client.post("/api/goals/99999/decompose").status_code == 404
    gid = client.post("/api/goals", json={"title": "Launch newsletter"}).json()["id"]
    plan = client.post(f"/api/goals/{gid}/decompose")
    assert plan.status_code == 200
    assert len(plan.json()["milestones"]) == 2
    # proposing persists nothing
    assert client.get(f"/api/goals/{gid}").json()["goal"]["tasks_total"] == 0

    assert client.post("/api/goals/99999/plan/accept", json=_PLAN).status_code == 404
    bad = client.post(f"/api/goals/{gid}/plan/accept", json={"milestones": [{"nope": 1}]})
    assert bad.status_code == 422

    accepted = client.post(f"/api/goals/{gid}/plan/accept", json=plan.json())
    assert accepted.status_code == 200
    assert len(accepted.json()["task_ids"]) == 5  # 2 milestones + 3 steps
    goal = client.get(f"/api/goals/{gid}").json()["goal"]
    assert goal["tasks_total"] == 5 and goal["tasks_done"] == 0
    listed = client.get("/api/goals").json()["goals"][0]
    assert listed["tasks_total"] == 5 and listed["tasks_done"] == 0
    page = client.get("/goals")
    assert "0/5 tasks done" in page.text   # progress doubles as the drill-down toggle
    assert "toggleTasks" in page.text


def test_api_decompose_maps_llm_garbage_to_502(client, conn, monkeypatch):
    from secondbrain.tasks import decompose as dmod

    monkeypatch.setattr(dmod, "get_llm",
                        lambda settings: MockLLM(responses=["not json at all"]))
    gid = client.post("/api/goals", json={"title": "Launch newsletter"}).json()["id"]
    r = client.post(f"/api/goals/{gid}/decompose")
    assert r.status_code == 502
    assert "try again" in r.json()["detail"]


def test_goals_page_states(client, conn, settings):
    empty = client.get("/goals")
    assert empty.status_code == 200
    assert "No goals yet" in empty.text
    gid = client.post("/api/goals", json={
        "title": "Run a marathon", "target_date": "2020-01-01",  # long past → overdue
    }).json()["id"]
    page = client.get("/goals")
    assert "Run a marathon" in page.text
    assert "was due" in page.text          # overdue goals are called out
    assert "Break into tasks" in page.text  # decomposition reachable from the UI
    filtered = client.get("/goals?status=paused")
    assert filtered.status_code == 200
    assert "No paused goals" in filtered.text
    client.post(f"/api/goals/{gid}/status", json={"status": "paused"})
    assert "Resume" in client.get("/goals").text  # context-aware actions
