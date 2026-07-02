from secondbrain.goals import link, store
from secondbrain.knowledge import graph


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
