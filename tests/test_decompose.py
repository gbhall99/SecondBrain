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


class _CaptureLLM(MockLLM):
    """MockLLM that records every prompt it was asked to complete."""

    def __init__(self, responses):
        super().__init__(responses=responses)
        self.prompts = []

    def complete(self, *, system, prompt, schema=None, temperature=0.0, max_tokens=None):
        self.prompts.append(prompt)
        return super().complete(system=system, prompt=prompt, schema=schema,
                                temperature=temperature, max_tokens=max_tokens)


def test_prompt_includes_target_links_and_open_tasks(conn, settings):
    from secondbrain.knowledge import graph

    gid = conn.execute(
        "INSERT INTO goals (title, status, target_date) "
        "VALUES ('Launch newsletter', 'active', '2026-10-01')"
    ).lastrowid
    nid = graph.create_node(conn, type="topic", name="email marketing", embedding=None,
                            confidence=0.9, extraction_id=None)
    conn.execute(
        "INSERT INTO goal_links (goal_id, kind, ref_id, relation, score) "
        "VALUES (?, 'node', ?, 'related', 0.9)", (gid, nid),
    )
    store.create_task(conn, title="Pick an ESP", goal_id=gid)
    llm = _CaptureLLM([json.dumps(_PLAN)])
    decompose.propose_plan(conn, gid, llm=llm, settings=settings)
    prompt = llm.prompts[0]
    assert "Target date: 2026-10-01" in prompt
    assert "email marketing" in prompt
    assert "don't repeat these" in prompt and "Pick an ESP" in prompt


def test_accept_plan_is_idempotent_per_title(conn, settings):
    gid = _goal(conn)
    plan = decompose.propose_plan(conn, gid, llm=MockLLM(responses=[json.dumps(_PLAN)]),
                                  settings=settings)
    first = decompose.accept_plan(conn, gid, plan)
    assert len(first) == 5
    again = decompose.accept_plan(conn, gid, plan)  # same plan accepted twice
    assert again == []
    assert len(store.list_tasks(conn, goal_id=gid)) == 5
    # a partially-new plan creates only the new step, under the EXISTING milestone
    plan2 = {"milestones": [{"title": "Set up tooling", "tasks": [
        {"title": "Pick an ESP"},               # exists → skipped
        {"title": "Configure DNS"},             # new
    ]}]}
    created = decompose.accept_plan(conn, gid, plan2)
    assert len(created) == 1
    t = store.get_task(conn, created[0])
    assert t["title"] == "Configure DNS"
    parents = {x["title"]: x["id"] for x in store.list_tasks(conn, goal_id=gid)}
    assert t["parent_task_id"] == parents["Set up tooling"]


def test_milestone_with_zero_kept_steps_is_not_created(conn, settings):
    gid = _goal(conn)
    plan = {"milestones": [
        {"title": "Empty shell", "tasks": []},
        {"title": "Real work", "tasks": [{"title": "Do a thing"}]},
    ]}
    decompose.accept_plan(conn, gid, plan)
    titles = {t["title"] for t in store.list_tasks(conn, goal_id=gid)}
    assert "Empty shell" not in titles and {"Real work", "Do a thing"} <= titles


def test_plan_task_due_date_passes_through(conn, settings):
    gid = _goal(conn)
    plan = {"milestones": [{"title": "M", "tasks": [
        {"title": "dated step", "due_date": "2026-09-15"},
        {"title": "garbage date", "due_date": "soonish"},
    ]}]}
    decompose.accept_plan(conn, gid, plan)
    by_title = {t["title"]: t for t in store.list_tasks(conn, goal_id=gid)}
    assert by_title["dated step"]["due_date"] == "2026-09-15"
    assert by_title["garbage date"]["due_date"] is None  # invalid dates are dropped
