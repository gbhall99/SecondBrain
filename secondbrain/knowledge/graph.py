"""Knowledge-graph storage: node/edge CRUD, fact versioning, merge.

Pure SQLite. Embeddings reuse the speaker-registry BLOB format/cosine; merge and
``resolve_node_id`` mirror ``speaker/registry.py`` so nodes and speakers behave
the same way.
"""

from __future__ import annotations

import json
import sqlite3

from secondbrain.speaker import registry
from secondbrain.storage.db import transaction
from secondbrain.storage.models import utcnow_iso


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


# A new decision on the same subject supersedes an earlier one when the two
# texts are at least this similar (embedding cosine, or token-overlap fallback
# when no embedder is available). Conservative: unrelated decisions coexist.
DECISION_SUPERSEDE_THRESHOLD = 0.75


def _token_overlap(a: str, b: str) -> float:
    """Overlap coefficient of the normalized word sets (0..1)."""
    ta, tb = set(normalize_name(a).split()), set(normalize_name(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def decision_similarity(a: str, b: str) -> float:
    """Topic similarity of two decision texts.

    Embedding cosine when the local embedder is available (best signal), else
    plain token overlap so the supersede logic still works with semantic
    search disabled. Best-effort: any embedding failure falls back too.
    """
    try:
        from secondbrain.config import get_settings
        from secondbrain.knowledge import resolve

        settings = get_settings()
        ea = resolve.embed_name(a, settings)
        eb = resolve.embed_name(b, settings)
        if ea and eb:
            return registry.cosine(ea, eb)
    except Exception:  # noqa: BLE001 - similarity is best-effort, fall back
        pass
    return _token_overlap(a, b)


# --- nodes -------------------------------------------------------------------


def create_node(
    conn: sqlite3.Connection,
    *,
    type: str,
    name: str,
    embedding: list[float] | None,
    confidence: float | None,
    extraction_id: int | None,
    speaker_id: int | None = None,
    when: str | None = None,
) -> int:
    when = when or utcnow_iso()
    cur = conn.execute(
        """
        INSERT INTO kg_nodes
            (type, name, normalized_name, display_label, speaker_id, embedding,
             confidence, source_extraction_id, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            type,
            name,
            normalize_name(name),
            name,
            speaker_id,
            registry.serialize_embedding(embedding) if embedding else None,
            confidence,
            extraction_id,
            when,
            when,
        ),
    )
    return int(cur.lastrowid)


def add_alias(conn: sqlite3.Connection, node_id: int, alias: str) -> None:
    norm = normalize_name(alias)
    exists = conn.execute(
        "SELECT 1 FROM kg_aliases WHERE node_id=? AND normalized_alias=?", (node_id, norm)
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO kg_aliases (node_id, alias, normalized_alias) VALUES (?, ?, ?)",
            (node_id, alias, norm),
        )


def resolve_node_id(conn: sqlite3.Connection, node_id: int) -> int:
    seen: set[int] = set()
    cur = node_id
    while cur not in seen:
        seen.add(cur)
        row = conn.execute("SELECT merged_into FROM kg_nodes WHERE id=?", (cur,)).fetchone()
        if row is None or row["merged_into"] is None:
            return cur
        cur = int(row["merged_into"])
    # Defensive: a merged_into cycle (should never be written — merge_nodes
    # guards against it). Every node in the loop is merged-away, so return the
    # entry node rather than an arbitrary merged-away node from the cycle.
    return node_id


def get_node(conn: sqlite3.Connection, node_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM kg_nodes WHERE id=?", (node_id,)).fetchone()


def candidates(conn: sqlite3.Connection, node_type: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM kg_nodes WHERE type=? AND merged_into IS NULL", (node_type,)
    ).fetchall()


def touch_node_seen(conn: sqlite3.Connection, node_id: int, when: str) -> None:
    conn.execute(
        "UPDATE kg_nodes SET last_seen = MAX(COALESCE(last_seen,''), ?) WHERE id=?",
        (when, node_id),
    )


def set_node_speaker(conn: sqlite3.Connection, node_id: int, speaker_id: int) -> None:
    conn.execute(
        "UPDATE kg_nodes SET speaker_id=? WHERE id=? AND speaker_id IS NULL",
        (speaker_id, node_id),
    )


def rename_node(conn: sqlite3.Connection, node_id: int, label: str) -> None:
    """Set the display label the UI shows (the stored name — what was actually
    heard — is kept for matching/aliases, mirroring the speaker rename)."""
    conn.execute("UPDATE kg_nodes SET display_label=? WHERE id=?", (label, node_id))


def remove_alias(conn: sqlite3.Connection, node_id: int, alias_id: int) -> bool:
    """Delete one alias row of a node. Returns False when it doesn't exist."""
    cur = conn.execute(
        "DELETE FROM kg_aliases WHERE id=? AND node_id=?", (alias_id, node_id)
    )
    return cur.rowcount > 0


# --- edges (with fact versioning) --------------------------------------------


def upsert_edge(
    conn: sqlite3.Connection,
    *,
    src_node_id: int,
    dst_node_id: int | None,
    predicate: str | None,
    kind: str,
    object_text: str = "",
    due_date: str | None = None,
    due_date_norm: str | None = None,
    confidence: float | None = None,
    extraction_id: int | None = None,
    conversation_id: int | None = None,
    source_segment_ids: list[int] | None = None,
    when: str | None = None,
) -> int:
    """Insert an edge, with fact versioning for kind='fact' and decision
    versioning for kind='decision'.

    - Identical (src, predicate, kind, dst, object_text) already valid (and not
      superseded) → bump last_seen and merge citations; return the existing id.
    - kind='fact' with same (src, predicate) but a different object → supersede
      the old edge (valid=0) and insert the new one.
    - kind='decision': a new decision on the same subject whose text is highly
      similar in topic (embedding cosine ≥ DECISION_SUPERSEDE_THRESHOLD, token
      overlap fallback) marks the old one superseded_by=new_id. Superseded
      decisions keep valid=1 — read paths expose is_superseded instead of
      hiding history.
    """
    when = when or utcnow_iso()
    seg_json = json.dumps(sorted(set(source_segment_ids or [])))

    existing = conn.execute(
        """
        SELECT id, source_segment_ids FROM kg_edges
        WHERE valid=1 AND superseded_by IS NULL AND kind=? AND src_node_id=?
          AND COALESCE(predicate,'')=COALESCE(?,'')
          AND COALESCE(dst_node_id,-1)=COALESCE(?,-1)
          AND COALESCE(object_text,'')=COALESCE(?,'')
        LIMIT 1
        """,
        (kind, src_node_id, predicate, dst_node_id, object_text),
    ).fetchone()
    if existing is not None:
        prior = set(json.loads(existing["source_segment_ids"] or "[]"))
        merged = sorted(prior | set(source_segment_ids or []))
        conn.execute(
            "UPDATE kg_edges SET last_seen=?, source_segment_ids=? WHERE id=?",
            (when, json.dumps(merged), existing["id"]),
        )
        return int(existing["id"])

    if kind == "fact" and predicate:
        conn.execute(
            "UPDATE kg_edges SET valid=0, superseded_by=NULL "
            "WHERE valid=1 AND kind='fact' AND src_node_id=? AND predicate=?",
            (src_node_id, predicate),
        )

    # Decisions: find same-subject earlier decisions this one replaces. Gated
    # conservatively (exact same subject node + high text similarity) so two
    # unrelated decisions about one project never clobber each other.
    superseded_decisions: list[int] = []
    if kind == "decision" and object_text:
        for old in conn.execute(
            "SELECT id, object_text FROM kg_edges WHERE valid=1 AND kind='decision' "
            "AND superseded_by IS NULL AND src_node_id=?",
            (src_node_id,),
        ).fetchall():
            if not old["object_text"]:
                continue
            if decision_similarity(object_text, old["object_text"]) >= (
                DECISION_SUPERSEDE_THRESHOLD
            ):
                superseded_decisions.append(int(old["id"]))

    cur = conn.execute(
        """
        INSERT INTO kg_edges
            (src_node_id, dst_node_id, predicate, kind, object_text, due_date,
             due_date_norm, confidence, source_extraction_id, conversation_id,
             source_segment_ids, valid, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        (
            src_node_id, dst_node_id, predicate, kind, object_text, due_date,
            due_date_norm, confidence, extraction_id, conversation_id, seg_json,
            when, when,
        ),
    )
    new_id = int(cur.lastrowid)
    if kind == "fact" and predicate:
        conn.execute(
            "UPDATE kg_edges SET superseded_by=? WHERE valid=0 AND kind='fact' "
            "AND src_node_id=? AND predicate=? AND superseded_by IS NULL",
            (new_id, src_node_id, predicate),
        )
    if superseded_decisions:
        ph = ",".join("?" * len(superseded_decisions))
        conn.execute(
            f"UPDATE kg_edges SET superseded_by=? WHERE id IN ({ph})",
            (new_id, *superseded_decisions),
        )
    return new_id


def record_extraction(
    conn: sqlite3.Connection,
    *,
    conversation_id: int,
    model: str | None,
    backend: str | None,
    chunk_index: int,
    segment_id_low: int | None,
    segment_id_high: int | None,
    raw_json: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO knowledge_extractions
            (conversation_id, model, backend, chunk_index, segment_id_low,
             segment_id_high, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (conversation_id, model, backend, chunk_index, segment_id_low, segment_id_high, raw_json),
    )
    return int(cur.lastrowid)


# --- merge (mirrors speaker merge) -------------------------------------------


def merge_nodes(conn: sqlite3.Connection, src_id: int, dst_id: int) -> int:
    src = resolve_node_id(conn, src_id)
    dst = resolve_node_id(conn, dst_id)
    if src == dst:
        return 0
    if resolve_node_id(conn, dst) == src:
        raise ValueError(f"merge nodes {src}->{dst} would create a cycle")
    with transaction(conn):
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM kg_edges WHERE src_node_id=? OR dst_node_id=?", (src, src)
        ).fetchone()["n"]
        conn.execute("UPDATE kg_edges SET src_node_id=? WHERE src_node_id=?", (dst, src))
        conn.execute("UPDATE kg_edges SET dst_node_id=? WHERE dst_node_id=?", (dst, src))
        src_row = get_node(conn, src)
        if src_row is not None and src_row["name"]:
            add_alias(conn, dst, src_row["name"])
        conn.execute("UPDATE kg_aliases SET node_id=? WHERE node_id=?", (dst, src))
        # carry a speaker link if dst lacks one
        if src_row is not None and src_row["speaker_id"] is not None:
            set_node_speaker(conn, dst, int(src_row["speaker_id"]))
        conn.execute("UPDATE kg_nodes SET merged_into=? WHERE id=?", (dst, src))
    return n
