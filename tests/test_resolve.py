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


def _speaker(conn, name):
    return int(conn.execute(
        "INSERT INTO speakers (name, kind) VALUES (?, 'known')", (name,)
    ).lastrowid)


def test_same_name_different_speaker_stays_distinct(conn, settings):
    """Two different people named the same must NOT merge into one node."""
    s1, s2 = _speaker(conn, "Alex R"), _speaker(conn, "Alex T")
    when = "2026-06-16T09:00:00.000Z"
    n1 = resolve.resolve_entity(conn, _ent("Alex"), extraction_id=None, when=when,
                                settings=settings, speaker_hint=s1)
    n2 = resolve.resolve_entity(conn, _ent("Alex"), extraction_id=None, when=when,
                                settings=settings, speaker_hint=s2)
    assert n1 != n2
    assert graph.get_node(conn, n1)["speaker_id"] == s1
    assert graph.get_node(conn, n2)["speaker_id"] == s2


def test_same_speaker_same_name_merges(conn, settings):
    s1 = _speaker(conn, "Dana")
    when = "2026-06-16T09:00:00.000Z"
    a = resolve.resolve_entity(conn, _ent("Dana"), extraction_id=None, when=when,
                               settings=settings, speaker_hint=s1)
    b = resolve.resolve_entity(conn, _ent("Dana"), extraction_id=None, when=when,
                               settings=settings, speaker_hint=s1)
    assert a == b  # same speaker → same node


def test_name_match_binds_unbound_node_to_speaker(conn, settings):
    # A person node created without a speaker binding should be adopted (not
    # duplicated) when the same name later arrives with a speaker hint.
    when = "2026-06-16T09:00:00.000Z"
    n0 = resolve.resolve_entity(conn, _ent("Sam"), extraction_id=None, when=when, settings=settings)
    assert graph.get_node(conn, n0)["speaker_id"] is None
    s = _speaker(conn, "Sam")
    n1 = resolve.resolve_entity(conn, _ent("Sam"), extraction_id=None, when=when,
                                settings=settings, speaker_hint=s)
    assert n1 == n0 and graph.get_node(conn, n1)["speaker_id"] == s


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


def test_generic_and_short_aliases_are_not_attached(conn, settings):
    got = resolve.resolve_entity(
        conn,
        _ent("Atlas", type="project",
             aliases=["the team", "AT", "me", "Project Atlas", "everyone"]),
        extraction_id=None, when="2026-06-16T09:00:00.000Z", settings=settings,
    )
    aliases = {
        r["alias"]
        for r in conn.execute("SELECT alias FROM kg_aliases WHERE node_id=?", (got,)).fetchall()
    }
    assert "Project Atlas" in aliases          # specific alias kept
    assert "Atlas" in aliases                  # the entity's own name always kept
    assert not {"the team", "AT", "me", "everyone"} & aliases


def test_alias_ok_gate():
    assert resolve.alias_ok("Project Atlas") is True
    assert resolve.alias_ok("ab") is False        # too short
    assert resolve.alias_ok("the team") is False  # stoplisted
    assert resolve.alias_ok("Everyone") is False
    assert resolve.alias_ok("Bob") is True


def test_candidates_cache_loads_once_per_type(conn, settings, monkeypatch):
    from secondbrain.search import semantic

    class FakeEmb:
        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(semantic, "get_embedder", lambda *_a, **_k: FakeEmb())
    graph.create_node(conn, type="topic", name="Caching", embedding=[1.0, 0.0],
                      confidence=0.9, extraction_id=None)
    calls = {"n": 0}
    real = graph.candidates

    def counting(conn_, node_type):
        calls["n"] += 1
        return real(conn_, node_type)

    monkeypatch.setattr(resolve.graph, "candidates", counting)
    cache: dict = {}
    when = "2026-06-16T09:00:00.000Z"
    for name in ("Memoization", "Result reuse", "Cache warming"):
        resolve.resolve_entity(conn, _ent(name, type="topic"), extraction_id=None,
                               when=when, settings=settings, cache=cache)
    assert calls["n"] == 1  # loaded once, reused for the rest of the run


def test_candidates_cache_invalidated_when_node_created(conn, settings, monkeypatch):
    from secondbrain.search import semantic

    # Embeddings are orthogonal per name → nothing matches, every entity
    # creates a node, and each creation must invalidate the type's cache.
    vecs = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]}

    class FakeEmb:
        def __init__(self):
            self.next = None

        def encode(self, texts):
            return [vecs[t[0]] for t in texts]

    monkeypatch.setattr(semantic, "get_embedder", lambda *_a, **_k: FakeEmb())
    cache: dict = {}
    when = "2026-06-16T09:00:00.000Z"
    a = resolve.resolve_entity(conn, _ent("Alpha", type="topic"), extraction_id=None,
                               when=when, settings=settings, cache=cache)
    # After creating Alpha the cache for 'topic' was dropped, so an identical
    # later mention (same embedding) can find it again through candidates.
    b = resolve.resolve_entity(conn, _ent("Aleph", type="topic"), extraction_id=None,
                               when=when, settings=settings, cache=cache)
    assert a == b  # same leading letter → same vector → matched via candidates


def test_resolve_node_id_cycle_returns_entry_node(conn):
    a = graph.create_node(conn, type="person", name="A", embedding=None,
                          confidence=0.9, extraction_id=None)
    b = graph.create_node(conn, type="person", name="B", embedding=None,
                          confidence=0.9, extraction_id=None)
    # Manufacture a corrupt merged_into cycle (merge_nodes itself refuses this).
    conn.execute("UPDATE kg_nodes SET merged_into=? WHERE id=?", (b, a))
    conn.execute("UPDATE kg_nodes SET merged_into=? WHERE id=?", (a, b))
    assert graph.resolve_node_id(conn, a) == a  # the entry node, deterministic
    assert graph.resolve_node_id(conn, b) == b


def test_rename_and_remove_alias_helpers(conn):
    nid = graph.create_node(conn, type="project", name="atlas", embedding=None,
                            confidence=0.9, extraction_id=None)
    graph.rename_node(conn, nid, "Atlas (platform)")
    row = conn.execute("SELECT name, display_label FROM kg_nodes WHERE id=?", (nid,)).fetchone()
    assert row["display_label"] == "Atlas (platform)" and row["name"] == "atlas"
    graph.add_alias(conn, nid, "The Platform")
    alias_id = conn.execute(
        "SELECT id FROM kg_aliases WHERE node_id=? AND alias='The Platform'", (nid,)
    ).fetchone()["id"]
    assert graph.remove_alias(conn, nid, alias_id) is True
    assert graph.remove_alias(conn, nid, alias_id) is False  # already gone
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM kg_aliases WHERE node_id=? AND alias='The Platform'", (nid,)
    ).fetchone()["n"] == 0
