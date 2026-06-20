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
