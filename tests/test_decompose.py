import json

from secondbrain.llm.client import MockLLM
from secondbrain.tasks import decompose, store


def _goal(conn, title="Launch newsletter"):
    return conn.execute("INSERT INTO goals (title,status) VALUES (?, 'active')", (title,)).lastrowid


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


def test_propose_does_not_commit(conn, settings):
    gid = _goal(conn)
    plan = decompose.propose_plan(conn, gid, llm=MockLLM(responses=[json.dumps(_PLAN)]), settings=settings)
    assert len(plan["milestones"]) == 2
    assert store.list_tasks(conn, goal_id=gid) == []   # nothing persisted yet


def test_accept_creates_task_tree(conn, settings):
    gid = _goal(conn)
    plan = decompose.propose_plan(conn, gid, llm=MockLLM(responses=[json.dumps(_PLAN)]), settings=settings)
    ids = decompose.accept_plan(conn, gid, plan)
    assert len(ids) == 5   # 2 milestones + 3 tasks
    tasks = store.list_tasks(conn, goal_id=gid)
    parents = [t for t in tasks if t["parent_task_id"] is None]
    children = [t for t in tasks if t["parent_task_id"] is not None]
    assert len(parents) == 2 and len(children) == 3
    assert all(t["source"] == "ai" for t in tasks)


def test_accept_chains_ordering_and_holds_container(conn, settings):
    gid = _goal(conn)
    plan = decompose.propose_plan(conn, gid, llm=MockLLM(responses=[json.dumps(_PLAN)]), settings=settings)
    decompose.accept_plan(conn, gid, plan)
    tasks = store.list_tasks(conn, goal_id=gid)
    by_title = {t["title"]: t for t in tasks}
    # Second step in a milestone depends on the first (runs in order).
    assert store.is_ready(conn, by_title["Pick an ESP"]["id"]) is True
    assert store.is_ready(conn, by_title["Create signup form"]["id"]) is False
    # The milestone container isn't schedulable until its sub-tasks are done.
    assert store.is_ready(conn, by_title["Set up tooling"]["id"]) is False
    ready = {t["id"] for t in store.ready_tasks(conn)}
    assert by_title["Set up tooling"]["id"] not in ready
