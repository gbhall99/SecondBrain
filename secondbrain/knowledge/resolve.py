"""Entity resolution: link an extracted entity to an existing node, or create it.

Normalized-name match first (deterministic, works with no embedder — e.g. CI with
semantic disabled), then embedding cosine (reusing the text Embedder + registry
cosine), then an optional LLM yes/no disambiguation in the ambiguous band.
"""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.knowledge import graph
from secondbrain.knowledge.schema import ExEntity
from secondbrain.llm.client import LLM
from secondbrain.llm.jsonout import LLMJSONError, parse_json
from secondbrain.search import semantic
from secondbrain.speaker import registry


def embed_name(text: str, settings: Settings) -> list[float] | None:
    embedder = semantic.get_embedder(settings)
    if embedder is None:
        return None
    try:
        return embedder.encode([text])[0]
    except Exception:  # noqa: BLE001 - embedding is best-effort
        return None


def _name_match(conn: sqlite3.Connection, node_type: str, norm: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM kg_nodes WHERE type=? AND merged_into IS NULL "
        "AND normalized_name=? LIMIT 1",
        (node_type, norm),
    ).fetchone()
    if row:
        return int(row["id"])
    row = conn.execute(
        """
        SELECT n.id FROM kg_aliases a JOIN kg_nodes n ON n.id=a.node_id
        WHERE n.type=? AND n.merged_into IS NULL AND a.normalized_alias=? LIMIT 1
        """,
        (node_type, norm),
    ).fetchone()
    return int(row["id"]) if row else None


def _llm_same(llm: LLM, node_type: str, a: str, b: str) -> bool:
    schema = {"type": "object", "properties": {"same": {"type": "boolean"}}, "required": ["same"]}
    try:
        resp = llm.complete(
            system="You decide if two references denote the same real-world entity. "
            "Answer JSON {\"same\": true|false}.",
            prompt=f"Are these the same {node_type}?\nA: {a}\nB: {b}",
            schema=schema,
        )
        return bool(parse_json(resp.text).get("same"))
    except (LLMJSONError, Exception):  # noqa: BLE001 - conservative on failure
        return False


def resolve_entity(
    conn: sqlite3.Connection,
    ent: ExEntity,
    *,
    extraction_id: int | None,
    when: str,
    llm: LLM | None = None,
    settings: Settings | None = None,
    speaker_hint: int | None = None,
) -> int:
    settings = settings or get_settings()
    norm = graph.normalize_name(ent.name)

    node_id = _name_match(conn, ent.type, norm)
    if node_id is None:
        emb = embed_name(ent.name, settings)
        if emb is not None:
            best_id, best_sim = None, -1.0
            for cand in graph.candidates(conn, ent.type):
                cvec = registry.deserialize_embedding(cand["embedding"])
                if not cvec:
                    continue
                sim = registry.cosine(emb, cvec)
                if sim > best_sim:
                    best_id, best_sim = int(cand["id"]), sim
            d = settings.extraction
            if best_id is not None and best_sim >= d.entity_match_threshold or (
                best_id is not None
                and best_sim >= d.entity_review_threshold
                and llm is not None
                and _llm_same(llm, ent.type, ent.name, graph.get_node(conn, best_id)["name"])
            ):
                node_id = best_id
        if node_id is None:
            node_id = graph.create_node(
                conn,
                type=ent.type,
                name=ent.name,
                embedding=emb,
                confidence=ent.confidence,
                extraction_id=extraction_id,
                speaker_id=speaker_hint if ent.type == "person" else None,
                when=when,
            )

    # existing node: record alias + freshness, bind speaker if Person
    graph.add_alias(conn, node_id, ent.name)
    for alias in ent.aliases:
        graph.add_alias(conn, node_id, alias)
    graph.touch_node_seen(conn, node_id, when)
    if ent.type == "person" and speaker_hint is not None:
        graph.set_node_speaker(conn, node_id, speaker_hint)
    return node_id
