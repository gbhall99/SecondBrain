from secondbrain.knowledge import graph, resolve
from secondbrain.knowledge.schema import ExEntity
from secondbrain.llm.client import MockLLM


def _ent(name, type="person", **kw):
    return ExEntity(type=type, name=name, **kw)


def test_name_match_links_existing(conn, settings):
    nid = graph.create_node(conn, type="organization", name="Acme Corp", embedding=None,
                            confidence=0.9, extraction_id=None)
    got = resolve.resolve_entity(conn, _ent("acme corp", type="organization"),
                                 extraction_id=None, when="2026-06-16T09:00:00.000Z", settings=settings)
    assert got == nid  # normalized-name match, no duplicate


def test_alias_match_links_existing(conn, settings):
    nid = graph.create_node(conn, type="person", name="Robert", embedding=None,
                            confidence=0.9, extraction_id=None)
    graph.add_alias(conn, nid, "Bob")
    got = resolve.resolve_entity(conn, _ent("bob"), extraction_id=None,
                                 when="2026-06-16T09:00:00.000Z", settings=settings)
    assert got == nid


def test_new_entity_created_when_no_match(conn, settings):
    got = resolve.resolve_entity(conn, _ent("Totally New Person"), extraction_id=None,
                                 when="2026-06-16T09:00:00.000Z", settings=settings)
    assert conn.execute("SELECT name FROM kg_nodes WHERE id=?", (got,)).fetchone()["name"] == "Totally New Person"


def test_embedding_auto_link(conn, settings, monkeypatch):
    from secondbrain.search import semantic

    class FakeEmb:
        def encode(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(semantic, "get_embedder", lambda *_a, **_k: FakeEmb())
    # existing node with the same embedding the fake returns
    nid = graph.create_node(conn, type="topic", name="Caching", embedding=[1.0, 0.0, 0.0, 0.0],
                            confidence=0.9, extraction_id=None)
    got = resolve.resolve_entity(conn, _ent("Memoization", type="topic"), extraction_id=None,
                                 when="2026-06-16T09:00:00.000Z", settings=settings)
    assert got == nid  # different name, but embedding cosine ≈ 1 ≥ match threshold


def test_llm_disambiguation_in_review_band(conn, settings, monkeypatch):
    import math

    from secondbrain.search import semantic

    cos = 0.75  # within [review=0.70, match=0.82)
    vecs = {"A": [1.0, 0.0], "B": [cos, math.sqrt(1 - cos * cos)]}

    class FakeEmb:
        def encode(self, texts):
            return [vecs.get("B", [0.0, 1.0]) for _ in texts]  # entity embeds as "B"

    monkeypatch.setattr(semantic, "get_embedder", lambda *_a, **_k: FakeEmb())
    nid = graph.create_node(conn, type="person", name="Existing", embedding=vecs["A"],
                            confidence=0.9, extraction_id=None)
    # LLM says "same" → link
    got = resolve.resolve_entity(conn, _ent("Other"), extraction_id=None,
                                 when="2026-06-16T09:00:00.000Z",
                                 llm=MockLLM(responses=['{"same": true}']), settings=settings)
    assert got == nid


def test_fact_versioning_supersedes(conn):
    a = graph.create_node(conn, type="person", name="Sarah", embedding=None, confidence=1.0, extraction_id=None)
    e1 = graph.upsert_edge(conn, src_node_id=a, dst_node_id=None, predicate="works_on",
                           kind="fact", object_text="Project A", source_segment_ids=[1])
    e2 = graph.upsert_edge(conn, src_node_id=a, dst_node_id=None, predicate="works_on",
                           kind="fact", object_text="Project B", source_segment_ids=[2])
    old = conn.execute("SELECT valid, superseded_by FROM kg_edges WHERE id=?", (e1,)).fetchone()
    assert old["valid"] == 0 and old["superseded_by"] == e2
    valid = conn.execute("SELECT COUNT(*) AS n FROM kg_edges WHERE valid=1 AND predicate='works_on'").fetchone()["n"]
    assert valid == 1


def test_identical_fact_merges_citations(conn):
    a = graph.create_node(conn, type="person", name="Sam", embedding=None, confidence=1.0, extraction_id=None)
    e1 = graph.upsert_edge(conn, src_node_id=a, dst_node_id=None, predicate="likes",
                           kind="fact", object_text="tea", source_segment_ids=[1])
    e2 = graph.upsert_edge(conn, src_node_id=a, dst_node_id=None, predicate="likes",
                           kind="fact", object_text="tea", source_segment_ids=[5])
    assert e1 == e2  # same edge reused
    import json
    cites = json.loads(conn.execute("SELECT source_segment_ids FROM kg_edges WHERE id=?", (e1,)).fetchone()[0])
    assert cites == [1, 5]


def test_merge_nodes_repoints_and_resolves(conn):
    src = graph.create_node(conn, type="person", name="Bobby", embedding=None, confidence=0.9, extraction_id=None)
    dst = graph.create_node(conn, type="person", name="Robert", embedding=None, confidence=0.9, extraction_id=None)
    graph.upsert_edge(conn, src_node_id=src, dst_node_id=None, predicate="likes",
                      kind="fact", object_text="coffee", source_segment_ids=[1])
    moved = graph.merge_nodes(conn, src, dst)
    assert moved == 1
    assert graph.resolve_node_id(conn, src) == dst
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_edges WHERE src_node_id=?", (dst,)).fetchone()["n"] == 1
    aliases = [r["alias"] for r in conn.execute("SELECT alias FROM kg_aliases WHERE node_id=?", (dst,)).fetchall()]
    assert "Bobby" in aliases
