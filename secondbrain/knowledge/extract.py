"""Per-conversation knowledge extraction (local LLM → graph).

Enqueued after a conversation is diarized; drained by the worker. Reads the
conversation's diarized, speaker-attributed segments (skipping redacted/opted-out
speech), chunks them to a context budget, asks the local LLM for structured
knowledge, resolves entities, and writes nodes/edges with provenance. Hard facts
from low-confidence speaker attributions are downgraded to 'mention'.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC

from secondbrain.config import Settings, get_settings
from secondbrain.knowledge import graph, resolve
from secondbrain.knowledge.schema import OWNER_REF, ExtractionResult, extraction_json_schema
from secondbrain.llm.client import LLM, get_llm
from secondbrain.llm.jsonout import parse_json
from secondbrain.speaker import registry
from secondbrain.storage.models import utcnow_iso

JOB_EXTRACT = "extract_knowledge"

_SYSTEM = (
    "You extract structured knowledge from a diarized meeting transcript. "
    "Use ONLY what is explicitly stated — do not invent. Every item must cite the "
    "segment id(s) it came from in source_segment_ids. Resolve 'I'/'me'/'my' to the "
    "owner using subject_ref = -1. Output JSON matching the provided schema."
)


def enqueue_extraction(conn: sqlite3.Connection, conversation_id: int) -> int | None:
    from secondbrain.pipeline import queue as q

    return q.enqueue(
        conn, JOB_EXTRACT, {"conversation_id": conversation_id}, dedupe_key="conversation_id"
    )


def _load_segments(
    conn: sqlite3.Connection, conversation_id: int, settings: Settings
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT ts.id, ts.text, ts.start_at, ts.start_offset_s, ts.speaker_id,
               ts.speaker_confidence, sp.name AS speaker_name, sp.display_label,
               sp.is_owner
        FROM transcript_segments ts
        JOIN audio_files af ON af.id = ts.audio_file_id
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE af.conversation_id = ?
        ORDER BY ts.start_at, ts.start_offset_s, ts.id
        """,
        (conversation_id,),
    ).fetchall()
    out = []
    for r in rows:
        if r["text"] == registry.REDACTED_TEXT:
            continue
        if r["speaker_id"] is not None and registry.is_opted_out(conn, r["speaker_id"], settings):
            continue
        out.append(dict(r))
    return out


def _label(seg: dict) -> str:
    if seg["is_owner"]:
        return "Me"
    return seg["speaker_name"] or seg["display_label"] or "Unknown"


def _chunk(segments: list[dict], settings: Settings) -> list[list[dict]]:
    budget = settings.extraction.max_context_chars
    overlap = max(0, settings.extraction.overlap_segments)
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for seg in segments:
        line = len(seg["text"]) + 40
        if cur and size + line > budget:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else []
            size = sum(len(s["text"]) + 40 for s in cur)
        cur.append(seg)
        size += line
    if cur:
        chunks.append(cur)
    return chunks


def _render(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        conf = "" if s["speaker_confidence"] is None else f" conf={s['speaker_confidence']:.2f}"
        when = (s["start_at"] or "")[11:19]
        lines.append(f"[seg_id={s['id']} | {_label(s)}{conf} | {when}] {s['text']}")
    return "\n".join(lines)


def _speaker_hint(conn: sqlite3.Connection, name: str) -> int | None:
    norm = graph.normalize_name(name)
    row = conn.execute(
        "SELECT id FROM speakers WHERE merged_into IS NULL AND lower(name)=? LIMIT 1", (norm,)
    ).fetchone()
    return int(row["id"]) if row else None


def _owner_node(conn: sqlite3.Connection, extraction_id: int, when: str) -> int:
    owner = conn.execute("SELECT id, name FROM speakers WHERE is_owner=1 LIMIT 1").fetchone()
    if owner is not None:
        existing = conn.execute(
            "SELECT id FROM kg_nodes WHERE type='person' AND speaker_id=? AND merged_into IS NULL",
            (owner["id"],),
        ).fetchone()
        if existing:
            return int(existing["id"])
        return graph.create_node(
            conn, type="person", name=owner["name"] or "Me", embedding=None,
            confidence=1.0, extraction_id=extraction_id, speaker_id=int(owner["id"]), when=when,
        )
    existing = conn.execute(
        "SELECT id FROM kg_nodes WHERE type='person' AND normalized_name='me' LIMIT 1"
    ).fetchone()
    if existing:
        return int(existing["id"])
    return graph.create_node(
        conn, type="person", name="Me", embedding=None, confidence=1.0,
        extraction_id=extraction_id, when=when,
    )


def _min_conf(segments_by_id: dict[int, dict], seg_ids: list[int]) -> float | None:
    confs = [
        segments_by_id[s]["speaker_confidence"]
        for s in seg_ids
        if s in segments_by_id and segments_by_id[s]["speaker_confidence"] is not None
    ]
    return min(confs) if confs else None


def run_extraction(
    conn: sqlite3.Connection,
    conversation_id: int,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> int:
    """Extract knowledge for one conversation. Returns number of edges written."""
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    low = settings.diarization.low_confidence_threshold

    segments = _load_segments(conn, conversation_id, settings)
    if not segments:
        conn.execute(
            "UPDATE conversations SET knowledge_status='extracted' WHERE id=?", (conversation_id,)
        )
        return 0
    seg_by_id = {s["id"]: s for s in segments}

    edges_written = 0
    for chunk_index, chunk in enumerate(_chunk(segments, settings)):
        when = utcnow_iso()
        resp = llm.complete(
            system=_SYSTEM, prompt=_render(chunk), schema=extraction_json_schema()
        )
        result = ExtractionResult.model_validate(parse_json(resp.text))
        ext_id = graph.record_extraction(
            conn,
            conversation_id=conversation_id,
            model=resp.model,
            backend=resp.backend,
            chunk_index=chunk_index,
            segment_id_low=chunk[0]["id"],
            segment_id_high=chunk[-1]["id"],
            raw_json=resp.text,
        )

        # 1. entities → node ids
        node_ids: list[int] = []
        for ent in result.entities:
            node_ids.append(
                resolve.resolve_entity(
                    conn, ent, extraction_id=ext_id, when=when, llm=llm, settings=settings,
                    speaker_hint=_speaker_hint(conn, ent.name) if ent.type == "person" else None,
                )
            )

        def ref(idx, _nodes=node_ids, _ext=ext_id, _when=when) -> int | None:
            if idx is None:
                return None
            if idx == OWNER_REF:
                return _owner_node(conn, _ext, _when)
            if 0 <= idx < len(_nodes):
                return _nodes[idx]
            return None

        def kind_for(base: str, seg_ids: list[int]) -> str:
            mc = _min_conf(seg_by_id, seg_ids)
            return "mention" if (mc is not None and mc < low) else base

        def clean_segs(seg_ids: list[int]) -> list[int]:
            # Drop citations the LLM may have hallucinated (not real segment ids).
            return [i for i in seg_ids if i in seg_by_id]

        # 2. facts
        for f in result.facts:
            src = ref(f.subject_ref)
            if src is None:
                continue
            edges_written += 1
            graph.upsert_edge(
                conn, src_node_id=src, dst_node_id=ref(f.object_ref), predicate=f.predicate,
                kind=kind_for("fact", f.source_segment_ids), object_text=f.object_text,
                confidence=f.confidence, extraction_id=ext_id, conversation_id=conversation_id,
                source_segment_ids=clean_segs(f.source_segment_ids), when=when,
            )

        # 3. action items
        for a in result.action_items:
            src = ref(a.owed_by_ref) or _owner_node(conn, ext_id, when)
            graph.upsert_edge(
                conn, src_node_id=src, dst_node_id=ref(a.owed_to_ref), predicate="action_item",
                kind=kind_for("action_item", a.source_segment_ids), object_text=a.description,
                due_date=a.due_date, confidence=a.confidence, extraction_id=ext_id,
                conversation_id=conversation_id, when=when,
                source_segment_ids=clean_segs(a.source_segment_ids),
            )
            edges_written += 1

        # 4. decisions + ideas
        for kind, items in (("decision", result.decisions), ("idea", result.ideas)):
            for it in items:
                parts = [ref(p) for p in it.participant_refs]
                src = next((p for p in parts if p is not None), None)
                if src is None:
                    src = _owner_node(conn, ext_id, when)
                graph.upsert_edge(
                    conn, src_node_id=src, dst_node_id=None, predicate=kind,
                    kind=kind_for(kind, it.source_segment_ids) if kind == "decision" else kind,
                    object_text=it.summary, confidence=it.confidence, extraction_id=ext_id,
                    conversation_id=conversation_id, when=when,
                    source_segment_ids=clean_segs(it.source_segment_ids),
                )
                edges_written += 1

    conn.execute(
        "UPDATE conversations SET knowledge_status='extracted' WHERE id=?", (conversation_id,)
    )
    if settings.proactive.enabled and settings.proactive.event_triggers:
        _maybe_nudge(conn, conversation_id, settings)
    return edges_written


def _maybe_nudge(conn: sqlite3.Connection, conversation_id: int, settings: Settings) -> None:
    """Enqueue a brief refresh if this conversation created an urgent commitment."""
    from datetime import datetime, timedelta

    from secondbrain.pipeline import queue as q
    from secondbrain.proactive.engine import JOB_PROACTIVE

    now = datetime.now(UTC)
    horizon = (now + timedelta(hours=settings.proactive.urgent_due_hours)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")
    urgent = conn.execute(
        """
        SELECT 1 FROM kg_edges
        WHERE kind='action_item' AND valid=1 AND conversation_id=?
          AND due_date IS NOT NULL AND substr(due_date,1,10) BETWEEN ? AND ?
        LIMIT 1
        """,
        (conversation_id, today, horizon),
    ).fetchone()
    if urgent:
        q.enqueue(conn, JOB_PROACTIVE, {"kind": "daily"}, dedupe_key="kind")
