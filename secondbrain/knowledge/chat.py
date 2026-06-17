"""Graph-RAG grounded Q&A.

Retrieve cited transcript spans (FTS + optional semantic) and a small relevant
subgraph, assemble a context block, and ask the local LLM. Grounded claims must
cite [seg_id]; general knowledge is allowed but must be clearly labeled (per the
user's "grounded + general" choice).
"""

from __future__ import annotations

import json
import re
import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.knowledge import graph
from secondbrain.llm.client import LLM, get_llm
from secondbrain.search import combined

_CITE = re.compile(r"\[(?:seg_id=)?(\d+)\]")
_GENERAL_TAG = "(general knowledge"

_SYSTEM = (
    "You are the user's second brain. Answer the question using the provided context "
    "(transcript excerpts and known facts). Cite every claim drawn from the context "
    "with its [seg_id]. You MAY add helpful general knowledge to fill gaps, but you "
    "MUST prefix any such sentence with '(general knowledge — not from your data)'. "
    "If the context doesn't cover something and you don't know, say so."
)


def _seg_info(conn: sqlite3.Connection, seg_ids: list[int]) -> dict[int, dict]:
    if not seg_ids:
        return {}
    ph = ",".join("?" * len(seg_ids))
    rows = conn.execute(
        f"""
        SELECT ts.id, ts.text, ts.start_at, af.conversation_id,
               COALESCE(sp.name, sp.display_label) AS speaker,
               CASE WHEN sp.is_owner THEN 1 ELSE 0 END AS is_owner
        FROM transcript_segments ts
        JOIN audio_files af ON af.id = ts.audio_file_id
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE ts.id IN ({ph})
        """,
        seg_ids,
    ).fetchall()
    out = {}
    for r in rows:
        d = dict(r)
        d["speaker"] = "Me" if r["is_owner"] else (r["speaker"] or "Unknown")
        out[r["id"]] = d
    return out


def _seed_nodes(conn: sqlite3.Connection, seg_ids: list[int], question: str) -> list[int]:
    nodes: set[int] = set()
    # nodes whose edges cite the retrieved segments
    for r in conn.execute(
        "SELECT src_node_id, dst_node_id, source_segment_ids FROM kg_edges WHERE valid=1"
    ).fetchall():
        cited = set(json.loads(r["source_segment_ids"] or "[]"))
        if cited & set(seg_ids):
            nodes.add(graph.resolve_node_id(conn, r["src_node_id"]))
            if r["dst_node_id"]:
                nodes.add(graph.resolve_node_id(conn, r["dst_node_id"]))
    # nodes whose name appears in the question
    qnorm = graph.normalize_name(question)
    node_rows = conn.execute(
        "SELECT id, normalized_name FROM kg_nodes WHERE merged_into IS NULL"
    ).fetchall()
    for r in node_rows:
        if r["normalized_name"] and r["normalized_name"] in qnorm:
            nodes.add(int(r["id"]))
    return list(nodes)


def _subgraph_facts(
    conn: sqlite3.Connection, node_ids: list[int], settings: Settings
) -> list[dict]:
    if not node_ids:
        return []
    ph = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"""
        SELECT e.predicate, e.kind, e.object_text, e.due_date, e.source_segment_ids,
               s.name AS src_name, d.name AS dst_name
        FROM kg_edges e
        JOIN kg_nodes s ON s.id = e.src_node_id
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        WHERE e.valid=1 AND (e.src_node_id IN ({ph}) OR e.dst_node_id IN ({ph}))
        ORDER BY e.confidence DESC
        LIMIT ?
        """,
        [*node_ids, *node_ids, settings.extraction.chat_max_facts],
    ).fetchall()
    return [dict(r) for r in rows]


def _fact_line(f: dict) -> str:
    obj = f["dst_name"] or f["object_text"] or ""
    due = f" (due {f['due_date']})" if f.get("due_date") else ""
    cites = json.loads(f["source_segment_ids"] or "[]")
    cite = " " + " ".join(f"[{c}]" for c in cites) if cites else ""
    return f"- {f['src_name']} {f['predicate'] or f['kind']} {obj}{due}{cite}"


def answer(
    conn: sqlite3.Connection,
    question: str,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    llm = llm or get_llm(settings)

    hits = combined.search(conn, question, limit=8, settings=settings)
    seg_ids = [h.segment_id for h in hits]
    info = _seg_info(conn, seg_ids)
    seed = _seed_nodes(conn, seg_ids, question)
    facts = _subgraph_facts(conn, seed, settings)
    for f in facts:  # ensure fact-cited segments are resolvable too
        for c in json.loads(f["source_segment_ids"] or "[]"):
            seg_ids.append(c)
    info = _seg_info(conn, sorted(set(seg_ids)))

    def _excerpt(sid: int) -> str:
        s = info[sid]
        return f"[{sid}] {s['speaker']} ({(s['start_at'] or '')[:19]}): {s['text']}"

    excerpts = "\n".join(_excerpt(sid) for sid in seg_ids if sid in info)
    fact_block = "\n".join(_fact_line(f) for f in facts)
    context = ""
    if excerpts:
        context += f"Transcript excerpts:\n{excerpts}\n\n"
    if fact_block:
        context += f"Known facts:\n{fact_block}\n\n"
    context = context[: settings.extraction.chat_max_context_chars]

    prompt = f"{context}Question: {question}"
    resp = llm.complete(system=_SYSTEM, prompt=prompt)

    cited_ids = {int(m) for m in _CITE.findall(resp.text)}
    citations = [
        {
            "segment_id": sid,
            "conversation_id": info[sid]["conversation_id"],
            "start_at": info[sid]["start_at"],
            "speaker": info[sid]["speaker"],
            "text": info[sid]["text"],
        }
        for sid in cited_ids
        if sid in info
    ]
    return {
        "question": question,
        "answer": resp.text,
        "citations": citations,
        "general_used": _GENERAL_TAG in resp.text,
        "grounded": bool(citations),
    }
