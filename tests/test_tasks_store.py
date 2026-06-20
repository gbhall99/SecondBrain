from secondbrain.knowledge import graph
from secondbrain.tasks import store


def test_create_list_and_done_bumps_goal(conn):
    gid = conn.execute("INSERT INTO goals (title,status) VALUES ('G','active')").lastrowid
    tid = store.create_task(conn, title="Step 1", goal_id=gid, estimate_minutes=30)
    assert store.list_tasks(conn, goal_id=gid)[0]["id"] == tid
    store.set_status(conn, tid, "done")
    assert store.get_task(conn, tid)["status"] == "done"
    assert conn.execute("SELECT last_progress_at FROM goals WHERE id=?", (gid,)).fetchone()[0]


def test_dependencies_and_readiness(conn):
    a = store.create_task(conn, title="A")
    b = store.create_task(conn, title="B")
    store.add_dependency(conn, b, a)
    assert store.is_ready(conn, a)            # no deps
    assert not store.is_ready(conn, b)        # waits on A
    store.set_status(conn, a, "done")
    assert store.is_ready(conn, b)            # A done → B ready
    ready_ids = {t["id"] for t in store.ready_tasks(conn)}
    assert b in ready_ids and a not in ready_ids  # A is done, excluded


def test_promote_action_item_idempotent(conn):
    owner = graph.create_node(conn, type="person", name="Me", embedding=None,
                              confidence=1.0, extraction_id=None)
    edge = graph.upsert_edge(conn, src_node_id=owner, dst_node_id=None, predicate="action_item",
                             kind="action_item", object_text="send the report",
                             due_date="2026-07-01", source_segment_ids=[1])
    t1 = store.promote_action_item(conn, edge)
    t2 = store.promote_action_item(conn, edge)        # idempotent
    assert t1 == t2
    task = store.get_task(conn, t1)
    assert task["source"] == "conversation" and task["source_edge_id"] == edge
    assert task["due_date"] == "2026-07-01"
