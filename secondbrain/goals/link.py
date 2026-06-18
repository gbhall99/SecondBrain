"""Auto-link goals to knowledge-graph nodes/edges.

Embedding cosine (reusing the registry helpers + the text Embedder) when
available, with a deterministic normalized-keyword fallback so linking is
testable on CI without embeddings.
"""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.knowledge.graph import normalize_name
from secondbrain.search import semantic
from secondbrain.speaker import registry

# kg node types and edge kinds worth linking a goal to
_NODE_TYPES = ("project", "organization", "topic")
_EDGE_KINDS = ("idea", "decision", "action_item")


def _keyword_score(a_norm: str, b_norm: str) -> float:
    at, bt = set(a_norm.split()), set(b_norm.split())
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


def relink_goal(conn: sqlite3.Connection, goal_id: int, settings: Settings | None = None) -> int:
    """(Re)compute related links for a goal. Returns the number of links written."""
    settings = settings or get_settings()
    goal = conn.execute(
        "SELECT title, description, embedding FROM goals WHERE id=?", (goal_id,)
    ).fetchone()
    if goal is None:
        return 0
    goal_norm = normalize_name(f"{goal['title']} {goal['description'] or ''}")
    goal_vec = registry.deserialize_embedding(goal["embedding"])
    embedder = semantic.get_embedder(settings) if goal_vec else None
    threshold = settings.proactive.goal_link_threshold

    conn.execute("DELETE FROM goal_links WHERE goal_id=? AND relation='related'", (goal_id,))

    def _score(cand_text: str, cand_norm: str, cand_vec: list[float] | None) -> float:
        if goal_vec and cand_vec:
            return registry.cosine(goal_vec, cand_vec)
        if goal_vec and embedder is not None and cand_text:
            try:
                return registry.cosine(goal_vec, embedder.encode([cand_text])[0])
            except Exception:  # noqa: BLE001
                return _keyword_score(goal_norm, cand_norm)
        return _keyword_score(goal_norm, cand_norm)

    written = 0
    nph = ",".join("?" * len(_NODE_TYPES))
    for n in conn.execute(
        f"SELECT id, name, normalized_name, embedding FROM kg_nodes "
        f"WHERE merged_into IS NULL AND type IN ({nph})",
        _NODE_TYPES,
    ).fetchall():
        nvec = registry.deserialize_embedding(n["embedding"])
        score = _score(n["name"], n["normalized_name"] or "", nvec)
        if score >= threshold:
            _insert_link(conn, goal_id, "node", n["id"], "related", score)
            written += 1

    eph = ",".join("?" * len(_EDGE_KINDS))
    for e in conn.execute(
        f"SELECT id, object_text FROM kg_edges WHERE valid=1 AND kind IN ({eph})",
        _EDGE_KINDS,
    ).fetchall():
        text = e["object_text"] or ""
        score = _score(text, normalize_name(text), None)
        if score >= threshold:
            _insert_link(conn, goal_id, "edge", e["id"], "related", score)
            written += 1
    return written


def _insert_link(conn, goal_id, kind, ref_id, relation, score) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO goal_links (goal_id, kind, ref_id, relation, score) "
        "VALUES (?, ?, ?, ?, ?)",
        (goal_id, kind, ref_id, relation, round(float(score), 4)),
    )


def link_advance(conn: sqlite3.Connection, goal_id: int, edge_id: int, score: float = 1.0) -> None:
    _insert_link(conn, goal_id, "edge", edge_id, "advances", score)
