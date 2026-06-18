"""Deterministic suggestion detectors (pure SQL + keyword/cosine; no LLM).

Each detector returns ``Suggestion`` records. The LLM is used only later for
brief synthesis (engine) and optional coaching — keeping detection explainable
and CI-testable without models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from secondbrain.config import Settings
from secondbrain.knowledge.graph import resolve_node_id


@dataclass
class Suggestion:
    kind: str
    title: str
    detail: str
    confidence: float
    payload: dict = field(default_factory=dict)
    citations: list[int] = field(default_factory=list)
    goal_id: int | None = None
    importance: float = 0.0

    @property
    def dedupe_hash(self) -> str:
        key = self.kind + "|" + json.dumps(self.payload.get("key", self.payload), sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]


def owner_node_id(conn) -> int | None:
    row = conn.execute("SELECT id FROM speakers WHERE is_owner=1 LIMIT 1").fetchone()
    if row is None:
        return None
    n = conn.execute(
        "SELECT id FROM kg_nodes WHERE speaker_id=? AND merged_into IS NULL LIMIT 1", (row["id"],)
    ).fetchone()
    return int(n["id"]) if n else None


def _as_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _cites(raw: str | None) -> list[int]:
    try:
        return list(json.loads(raw or "[]"))
    except json.JSONDecodeError:
        return []


# --- commitments (both directions) -------------------------------------------


def detect_commitments(
    conn, settings: Settings, *, owner_id: int | None, now: datetime
) -> list[Suggestion]:
    if owner_id is None:
        return []
    today = now.date()
    soon = today + timedelta(days=settings.proactive.due_soon_days)
    stale_before = today - timedelta(days=settings.proactive.stale_days)
    out: list[Suggestion] = []
    rows = conn.execute(
        """
        SELECT e.id, e.src_node_id, e.dst_node_id, e.object_text, e.due_date,
               e.confidence, e.source_segment_ids, e.last_seen,
               s.name AS src_name, d.name AS dst_name
        FROM kg_edges e
        JOIN kg_nodes s ON s.id = e.src_node_id
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        WHERE e.kind='action_item' AND e.valid=1
        """
    ).fetchall()
    for e in rows:
        src = resolve_node_id(conn, e["src_node_id"])
        dst = resolve_node_id(conn, e["dst_node_id"]) if e["dst_node_id"] else None
        due = _as_date(e["due_date"])
        conf = e["confidence"] if e["confidence"] is not None else 0.5
        desc = e["object_text"] or "(unspecified)"
        if src == owner_id and due is not None and today <= due <= soon:
            out.append(Suggestion(
                kind="commitment_owed",
                title=f"You owe: {desc}",
                detail=f"Due {e['due_date']}" + (f" to {e['dst_name']}" if e["dst_name"] else ""),
                confidence=conf, citations=_cites(e["source_segment_ids"]),
                payload={"key": {"edge": e["id"]}, "due_date": e["due_date"]},
            ))
        elif dst == owner_id:
            overdue = due is not None and due < today
            stale = due is None and (_as_date(e["last_seen"]) or today) < stale_before
            if overdue or stale:
                who = e["src_name"] or "Someone"
                det = f"Overdue ({e['due_date']})" if overdue else "No update in a while"
                out.append(Suggestion(
                    kind="commitment_overdue",
                    title=f"{who} owes you: {desc}",
                    detail=det,
                    confidence=conf, citations=_cites(e["source_segment_ids"]),
                    payload={"key": {"edge": e["id"]}, "due_date": e["due_date"]},
                ))
    return out


# --- connections -------------------------------------------------------------


def detect_connections(conn, settings: Settings, *, owner_id, now: datetime) -> list[Suggestion]:
    fmt = "%Y-%m-%dT%H:%M:%fZ"
    recent_cut = (now - timedelta(days=settings.proactive.recent_days)).strftime(fmt)
    look_cut = (now - timedelta(days=settings.proactive.lookback_days)).strftime(fmt)
    nodes = conn.execute(
        "SELECT id, name, normalized_name, last_seen FROM kg_nodes "
        "WHERE merged_into IS NULL AND type IN ('topic','project','idea','organization')"
    ).fetchall()
    recent = [n for n in nodes if (n["last_seen"] or "") >= recent_cut]
    older = [n for n in nodes if look_cut <= (n["last_seen"] or "") < recent_cut]
    seen: set[tuple] = set()
    out: list[Suggestion] = []
    for r in recent:
        for o in older:
            if r["id"] == o["id"]:
                continue
            score = _keyword(r["normalized_name"], o["normalized_name"])
            if score < settings.proactive.connection_threshold:
                continue
            pair = tuple(sorted((r["id"], o["id"])))
            if pair in seen:
                continue
            seen.add(pair)
            out.append(Suggestion(
                kind="connection",
                title=f"Possible link: {r['name']} ↔ {o['name']}",
                detail="These came up in different conversations and look related.",
                confidence=float(score),
                payload={"key": {"pair": list(pair)}},
            ))
    return out


def _keyword(a: str | None, b: str | None) -> float:
    at, bt = set((a or "").split()), set((b or "").split())
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


# --- goals: alignment + staleness --------------------------------------------


def detect_goal_alignment(conn, settings: Settings, *, owner_id, now: datetime) -> list[Suggestion]:
    today = now.strftime("%Y-%m-%d")
    out: list[Suggestion] = []
    for g in conn.execute("SELECT * FROM goals WHERE status='active'").fetchall():
        # advancing: a 'related' linked edge that was last seen today
        rows = conn.execute(
            """
            SELECT e.id, e.kind, e.object_text, e.source_segment_ids
            FROM goal_links gl JOIN kg_edges e ON e.id = gl.ref_id
            WHERE gl.goal_id=? AND gl.kind='edge' AND e.valid=1 AND substr(e.last_seen,1,10)=?
            """,
            (g["id"], today),
        ).fetchall()
        for e in rows:
            out.append(Suggestion(
                kind="goal_alignment",
                title=f"Progress on goal: {g['title']}",
                detail=f"{e['kind']}: {e['object_text']}",
                confidence=0.8, goal_id=g["id"], citations=_cites(e["source_segment_ids"]),
                payload={"key": {"goal": g["id"], "edge": e["id"]}},
            ))
        # contradiction candidate: a linked fact superseded today
        sup = conn.execute(
            """
            SELECT e.id, e.object_text FROM goal_links gl JOIN kg_edges e ON e.id = gl.ref_id
            WHERE gl.goal_id=? AND gl.kind='edge' AND e.valid=0 AND substr(e.last_seen,1,10)=?
            """,
            (g["id"], today),
        ).fetchall()
        for e in sup:
            out.append(Suggestion(
                kind="goal_alignment",
                title=f"Worth a look for goal: {g['title']}",
                detail=f"A related fact changed: {e['object_text']}",
                confidence=0.45, goal_id=g["id"],
                payload={"key": {"goal": g["id"], "superseded": e["id"]}},
            ))
    return out


def detect_stale_goals(conn, settings: Settings, *, owner_id, now: datetime) -> list[Suggestion]:
    delta = timedelta(days=settings.proactive.stale_goal_days)
    cutoff = (now - delta).strftime("%Y-%m-%dT%H:%M:%fZ")
    out: list[Suggestion] = []
    for g in conn.execute(
        "SELECT * FROM goals WHERE status='active' "
        "AND COALESCE(last_progress_at, created_at) < ?",
        (cutoff,),
    ).fetchall():
        out.append(Suggestion(
            kind="stale_goal",
            title=f"No recent progress: {g['title']}",
            detail="This active goal hasn't seen activity lately.",
            confidence=0.6, goal_id=g["id"], payload={"key": {"goal": g["id"]}},
        ))
    return out


DETECTORS = [detect_commitments, detect_connections, detect_goal_alignment, detect_stale_goals]
