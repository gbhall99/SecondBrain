"""Auto-link goals to knowledge-graph nodes/edges.

Embedding cosine (reusing the registry helpers + the text Embedder) when
available, with a deterministic normalized-keyword fallback so linking is
testable on CI without embeddings. Cosine and keyword scores live on very
different scales, so each has its own threshold
(``proactive.goal_link_threshold`` vs ``proactive.goal_link_keyword_threshold``).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

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


@dataclass
class _Candidate:
    kind: str          # 'node' | 'edge'
    ref_id: int
    text: str
    norm: str
    vec: list[float] | None


def relink_goal(
    conn: sqlite3.Connection,
    goal_id: int,
    settings: Settings | None = None,
    *,
    since_edge_id: int | None = None,
) -> int:
    """(Re)compute related links for a goal. Returns how many candidates
    currently clear the threshold.

    Surviving links keep their rows (and ``created_at``); only links that fell
    below the threshold are deleted. With ``since_edge_id`` the (expensive)
    edge scan is incremental — only edges newer than that id are scored, and
    existing edge links are left alone — which is what the digest path uses so
    the morning brief doesn't re-encode the whole graph every day.
    """
    settings = settings or get_settings()
    goal = conn.execute(
        "SELECT title, description, embedding FROM goals WHERE id=?", (goal_id,)
    ).fetchone()
    if goal is None:
        return 0
    goal_norm = normalize_name(f"{goal['title']} {goal['description'] or ''}")
    goal_vec = registry.deserialize_embedding(goal["embedding"])
    embedder = semantic.get_embedder(settings) if goal_vec else None
    cos_threshold = settings.proactive.goal_link_threshold
    kw_threshold = settings.proactive.goal_link_keyword_threshold
    incremental = since_edge_id is not None

    candidates: list[_Candidate] = []
    nph = ",".join("?" * len(_NODE_TYPES))
    for n in conn.execute(
        f"SELECT id, name, normalized_name, embedding FROM kg_nodes "
        f"WHERE merged_into IS NULL AND type IN ({nph})",
        _NODE_TYPES,
    ).fetchall():
        candidates.append(_Candidate(
            "node", n["id"], n["name"] or "", n["normalized_name"] or "",
            registry.deserialize_embedding(n["embedding"]),
        ))
    eph = ",".join("?" * len(_EDGE_KINDS))
    edge_sql = f"SELECT id, object_text FROM kg_edges WHERE valid=1 AND kind IN ({eph})"
    edge_params: list = list(_EDGE_KINDS)
    if incremental:
        edge_sql += " AND id > ?"
        edge_params.append(since_edge_id)
    for e in conn.execute(edge_sql, edge_params).fetchall():
        text = e["object_text"] or ""
        candidates.append(_Candidate("edge", e["id"], text, normalize_name(text), None))

    # One batched encode for every candidate that needs a vector, instead of
    # one model call per edge.
    if goal_vec and embedder is not None:
        missing = [c for c in candidates if c.vec is None and c.text]
        if missing:
            try:
                vecs = embedder.encode([c.text for c in missing])
                for c, v in zip(missing, vecs, strict=True):
                    c.vec = list(v)
            except Exception:  # noqa: BLE001 - fall back to keyword scoring
                pass

    kept: dict[tuple[str, int], float] = {}
    for c in candidates:
        if goal_vec and c.vec:
            score, threshold = registry.cosine(goal_vec, c.vec), cos_threshold
        else:
            score, threshold = _keyword_score(goal_norm, c.norm), kw_threshold
        if score >= threshold:
            kept[(c.kind, c.ref_id)] = score

    # Remove only 'related' links that fell below the threshold; surviving
    # rows are untouched (created_at preserved). In incremental mode edge
    # links below the scan window are not re-judged, so they are kept.
    kept_nodes = [ref for (kind, ref) in kept if kind == "node"]
    _prune(conn, goal_id, "node", kept_nodes)
    if not incremental:
        kept_edges = [ref for (kind, ref) in kept if kind == "edge"]
        _prune(conn, goal_id, "edge", kept_edges)

    for (kind, ref_id), score in kept.items():
        _insert_link(conn, goal_id, kind, ref_id, "related", score)
    return len(kept)


def _prune(conn, goal_id: int, kind: str, kept_refs: list[int]) -> None:
    sql = "DELETE FROM goal_links WHERE goal_id=? AND relation='related' AND kind=?"
    params: list = [goal_id, kind]
    if kept_refs:
        sql += f" AND ref_id NOT IN ({','.join('?' * len(kept_refs))})"
        params.extend(kept_refs)
    conn.execute(sql, params)


def _insert_link(conn, goal_id, kind, ref_id, relation, score) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO goal_links (goal_id, kind, ref_id, relation, score) "
        "VALUES (?, ?, ?, ?, ?)",
        (goal_id, kind, ref_id, relation, round(float(score), 4)),
    )
    conn.execute(
        "UPDATE goal_links SET score=? WHERE goal_id=? AND kind=? AND ref_id=? AND relation=?",
        (round(float(score), 4), goal_id, kind, ref_id, relation),
    )


def link_advance(conn: sqlite3.Connection, goal_id: int, edge_id: int, score: float = 1.0) -> None:
    _insert_link(conn, goal_id, "edge", edge_id, "advances", score)
