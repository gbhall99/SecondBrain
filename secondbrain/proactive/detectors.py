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
from secondbrain.speaker import registry

# Keyword Jaccard overlap is a much coarser signal than embedding cosine, so
# the connection detector's keyword fallback uses its own threshold.
CONNECTION_KEYWORD_THRESHOLD = 0.35
# Bound the pairwise connection loop however big the graph gets.
_CONNECTION_RECENT_LIMIT = 150
_CONNECTION_OLDER_LIMIT = 500
# A goal under this completion fraction with a near target date is "at risk".
GOAL_AT_RISK_COMPLETION = 0.5
# Undated commitments heard within this many days get a "set a date?" nudge.
UNDATED_NUDGE_DAYS = 3


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

# A commitment whose promoted task was finished (or dropped) is handled — it
# must stop resurfacing in the brief. Promoted-but-open tasks still surface:
# the promise is live either way.
_NOT_FULFILLED = (
    "NOT EXISTS (SELECT 1 FROM tasks t WHERE t.source_edge_id = e.id "
    "AND t.status IN ('done','dropped'))"
)


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
        f"""
        SELECT e.id, e.src_node_id, e.dst_node_id, e.object_text, e.due_date,
               e.due_date_norm, e.confidence, e.source_segment_ids, e.last_seen,
               s.name AS src_name, d.name AS dst_name
        FROM kg_edges e
        JOIN kg_nodes s ON s.id = e.src_node_id
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        WHERE e.kind='action_item' AND e.valid=1 AND {_NOT_FULFILLED}
        """
    ).fetchall()
    for e in rows:
        src = resolve_node_id(conn, e["src_node_id"])
        dst = resolve_node_id(conn, e["dst_node_id"]) if e["dst_node_id"] else None
        due_raw = e["due_date_norm"] or e["due_date"]
        due = _as_date(due_raw)
        conf = e["confidence"] if e["confidence"] is not None else 0.5
        desc = e["object_text"] or "(unspecified)"

        # The due-state is part of the dedupe key: a dismissed "due soon" item
        # is re-admitted ONCE when it crosses into overdue (a fresh hash), and
        # dismissing the overdue version then sticks.
        def _key(state: str, edge_id=e["id"], due=due_raw) -> dict:
            return {"key": {"edge": edge_id, "state": state}, "due_date": due}

        if src == owner_id:
            if due is not None and today <= due <= soon:
                out.append(Suggestion(
                    kind="commitment_owed",
                    title=f"You owe: {desc}",
                    detail=f"Due {due_raw}"
                    + (f" to {e['dst_name']}" if e["dst_name"] else ""),
                    confidence=conf, citations=_cites(e["source_segment_ids"]),
                    payload=_key("due"),
                ))
            elif (due is not None and due < today) or (
                due is None and (_as_date(e["last_seen"]) or today) < stale_before
            ):
                # Your own commitment is overdue / has gone stale with no due date.
                overdue = due is not None
                det = f"Overdue ({due_raw})" if overdue else "No progress in a while"
                out.append(Suggestion(
                    kind="commitment_overdue",
                    title=f"You still owe: {desc}",
                    detail=det + (f" (to {e['dst_name']})" if e["dst_name"] else ""),
                    confidence=conf, citations=_cites(e["source_segment_ids"]),
                    payload=_key("overdue" if overdue else "stale"),
                ))
        elif dst == owner_id:
            overdue = due is not None and due < today
            stale = due is None and (_as_date(e["last_seen"]) or today) < stale_before
            if overdue or stale:
                who = e["src_name"] or "Someone"
                det = f"Overdue ({due_raw})" if overdue else "No update in a while"
                out.append(Suggestion(
                    kind="commitment_overdue",
                    title=f"{who} owes you: {desc}",
                    detail=det,
                    confidence=conf, citations=_cites(e["source_segment_ids"]),
                    payload=_key("overdue" if overdue else "stale"),
                ))
    return out


def detect_undated_commitments(
    conn, settings: Settings, *, owner_id, now: datetime
) -> list[Suggestion]:
    """A promise of yours heard in the last few days with no due date → nudge
    to give it one (undated commitments are the ones that silently rot)."""
    if owner_id is None:
        return []
    cut = (now - timedelta(days=UNDATED_NUDGE_DAYS)).strftime("%Y-%m-%dT%H:%M:%fZ")
    out: list[Suggestion] = []
    rows = conn.execute(
        f"""
        SELECT e.id, e.src_node_id, e.object_text, e.confidence, e.source_segment_ids,
               d.name AS dst_name
        FROM kg_edges e
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        WHERE e.kind='action_item' AND e.valid=1
          AND e.due_date IS NULL AND e.due_date_norm IS NULL
          AND COALESCE(e.first_seen, e.created_at) >= ?
          AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.source_edge_id = e.id)
          AND {_NOT_FULFILLED}
        """,
        (cut,),
    ).fetchall()
    for e in rows:
        if resolve_node_id(conn, e["src_node_id"]) != owner_id:
            continue
        desc = e["object_text"] or "(unspecified)"
        out.append(Suggestion(
            kind="commitment_undated",
            title=f"You promised: {desc} — add a due date?",
            detail="Heard recently with no deadline"
            + (f" (to {e['dst_name']})" if e["dst_name"] else "")
            + ". Undated promises are the ones that slip.",
            confidence=e["confidence"] if e["confidence"] is not None else 0.5,
            citations=_cites(e["source_segment_ids"]),
            payload={"key": {"edge": e["id"], "state": "undated"}, "link": "/tasks#actions-h"},
        ))
    return out


# --- tasks + plan ---------------------------------------------------------------


def detect_tasks_due(conn, settings: Settings, *, owner_id, now: datetime) -> list[Suggestion]:
    """Open tasks due today/tomorrow (or already overdue) → one summary item."""
    from secondbrain.tasks.store import ACTIVE_STATUSES

    today = now.date()
    soon = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    ph = ",".join("?" * len(ACTIVE_STATUSES))
    rows = conn.execute(
        f"SELECT id, title, due_date FROM tasks WHERE status IN ({ph}) "
        f"AND due_date IS NOT NULL AND due_date <= ? ORDER BY due_date, id",
        (*ACTIVE_STATUSES, soon),
    ).fetchall()
    if not rows:
        return []
    overdue = sum(1 for r in rows if (_as_date(r["due_date"]) or today) < today)
    n = len(rows)
    title = f"{n} task{'s' if n != 1 else ''} due by tomorrow"
    if overdue:
        title += f" ({overdue} overdue)"
    detail = "; ".join(r["title"] for r in rows[:3])
    if n > 3:
        detail += f" — and {n - 3} more"
    return [Suggestion(
        kind="tasks_due", title=title, detail=detail, confidence=0.9,
        payload={"key": {"day": today.strftime("%Y-%m-%d"),
                         "tasks": [r["id"] for r in rows]},
                 "link": "/tasks"},
    )]


def detect_plan_carryover(
    conn, settings: Settings, *, owner_id, now: datetime
) -> list[Suggestion]:
    """Yesterday's accepted-but-unfinished plan → one honest summary item."""
    yesterday = (now.date() - timedelta(days=1)).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT task_ids FROM day_plans WHERE date=?", (yesterday,)
    ).fetchone()
    if row is None:
        return []
    try:
        ids = [int(i) for i in json.loads(row["task_ids"] or "[]")]
    except (TypeError, ValueError):
        return []
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    slipped = conn.execute(
        f"SELECT id, title FROM tasks WHERE id IN ({ph}) "
        "AND status NOT IN ('done','dropped')",
        ids,
    ).fetchall()
    if not slipped:
        return []
    n = len(slipped)
    detail = "; ".join(r["title"] for r in slipped[:3])
    if n > 3:
        detail += f" — and {n - 3} more"
    return [Suggestion(
        kind="plan_carryover",
        title=f"{n} planned item{'s' if n != 1 else ''} slipped from yesterday",
        detail=detail, confidence=0.85,
        payload={"key": {"day": yesterday}, "link": "/tasks"},
    )]


# --- connections -------------------------------------------------------------


def _node_citations(conn, node_id: int) -> list[int]:
    row = conn.execute(
        "SELECT source_segment_ids FROM kg_edges "
        "WHERE (src_node_id=? OR dst_node_id=?) AND valid=1 "
        "ORDER BY last_seen DESC LIMIT 1",
        (node_id, node_id),
    ).fetchone()
    return _cites(row["source_segment_ids"])[:2] if row else []


def detect_connections(conn, settings: Settings, *, owner_id, now: datetime) -> list[Suggestion]:
    """A recent topic that looks like an older one → a possible link.

    Uses stored node embeddings (cosine, ``connection_threshold``) when both
    sides have one, with a keyword-Jaccard fallback at its own (lower)
    threshold. Both node sets are LIMITed so the pairwise loop stays bounded.
    """
    fmt = "%Y-%m-%dT%H:%M:%fZ"
    recent_cut = (now - timedelta(days=settings.proactive.recent_days)).strftime(fmt)
    look_cut = (now - timedelta(days=settings.proactive.lookback_days)).strftime(fmt)
    base = (
        "SELECT id, name, normalized_name, last_seen, embedding FROM kg_nodes "
        "WHERE merged_into IS NULL AND type IN ('topic','project','idea','organization') "
    )
    recent = conn.execute(
        base + "AND last_seen >= ? ORDER BY last_seen DESC LIMIT ?",
        (recent_cut, _CONNECTION_RECENT_LIMIT),
    ).fetchall()
    older = conn.execute(
        base + "AND last_seen >= ? AND last_seen < ? ORDER BY last_seen DESC LIMIT ?",
        (look_cut, recent_cut, _CONNECTION_OLDER_LIMIT),
    ).fetchall()
    recent_vecs = {r["id"]: registry.deserialize_embedding(r["embedding"]) for r in recent}
    older_vecs = {o["id"]: registry.deserialize_embedding(o["embedding"]) for o in older}
    seen: set[tuple] = set()
    out: list[Suggestion] = []
    for r in recent:
        rvec = recent_vecs[r["id"]]
        for o in older:
            if r["id"] == o["id"]:
                continue
            ovec = older_vecs[o["id"]]
            if rvec and ovec:
                score = registry.cosine(rvec, ovec)
                threshold = settings.proactive.connection_threshold
            else:
                score = _keyword(r["normalized_name"], o["normalized_name"])
                threshold = CONNECTION_KEYWORD_THRESHOLD
            if score < threshold:
                continue
            pair = tuple(sorted((r["id"], o["id"])))
            if pair in seen:
                continue
            seen.add(pair)
            cites = _node_citations(conn, r["id"]) + _node_citations(conn, o["id"])
            out.append(Suggestion(
                kind="connection",
                title=f"Possible link: {r['name']} ↔ {o['name']}",
                detail="These came up in different conversations and look related.",
                confidence=round(min(1.0, float(score)), 4),
                citations=list(dict.fromkeys(cites)),
                payload={"key": {"pair": list(pair)}, "nodes": list(pair)},
            ))
    return out


def _keyword(a: str | None, b: str | None) -> float:
    at, bt = set((a or "").split()), set((b or "").split())
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


# --- goals: alignment + staleness + risk --------------------------------------


def detect_goal_alignment(conn, settings: Settings, *, owner_id, now: datetime) -> list[Suggestion]:
    from secondbrain.goals import link as goal_link
    from secondbrain.goals import store as goal_store

    # Look back over the configured recency window (widened for the weekly review),
    # not just the exact run-day, so the weekly "goal progress" reflects the week.
    cutoff = (now - timedelta(days=settings.proactive.recent_days)).strftime("%Y-%m-%d")
    out: list[Suggestion] = []
    for g in conn.execute("SELECT * FROM goals WHERE status='active'").fetchall():
        # advancing: a 'related' linked edge last seen within the recency window
        rows = conn.execute(
            """
            SELECT e.id, e.kind, e.object_text, e.source_segment_ids, e.last_seen
            FROM goal_links gl JOIN kg_edges e ON e.id = gl.ref_id
            WHERE gl.goal_id=? AND gl.kind='edge' AND e.valid=1 AND substr(e.last_seen,1,10)>=?
            """,
            (g["id"], cutoff),
        ).fetchall()
        for e in rows:
            out.append(Suggestion(
                kind="goal_alignment",
                title=f"Progress on goal: {g['title']}",
                detail=f"{e['kind']}: {e['object_text']}",
                confidence=0.8, goal_id=g["id"], citations=_cites(e["source_segment_ids"]),
                payload={"key": {"goal": g["id"], "edge": e["id"]}},
            ))
            # The goal was actually discussed: record the progress (so the
            # stale-goal detector stops nagging about it) and mark decision /
            # action edges as advancing the goal.
            if e["last_seen"] and (
                not g["last_progress_at"] or e["last_seen"] > g["last_progress_at"]
            ):
                goal_store.mark_progress(conn, g["id"], when=e["last_seen"])
            if e["kind"] in ("decision", "action_item"):
                goal_link.link_advance(conn, g["id"], e["id"], score=0.9)
        # contradiction candidate: a linked fact superseded today
        sup = conn.execute(
            """
            SELECT e.id, e.object_text FROM goal_links gl JOIN kg_edges e ON e.id = gl.ref_id
            WHERE gl.goal_id=? AND gl.kind='edge' AND e.valid=0 AND substr(e.last_seen,1,10)>=?
            """,
            (g["id"], cutoff),
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


def detect_goal_at_risk(conn, settings: Settings, *, owner_id, now: datetime) -> list[Suggestion]:
    """Target date close + completion low → say so, with the numbers."""
    from secondbrain.goals.store import progress_counts

    today = now.date()
    horizon = today + timedelta(days=settings.proactive.goal_at_risk_days)
    counts = progress_counts(conn)
    out: list[Suggestion] = []
    for g in conn.execute(
        "SELECT id, title, target_date FROM goals "
        "WHERE status='active' AND target_date IS NOT NULL"
    ).fetchall():
        target = _as_date(g["target_date"])
        if target is None or not today <= target <= horizon:
            continue
        done, total = counts.get(g["id"], (0, 0))
        pct = (done / total) if total else 0.0
        if pct >= GOAL_AT_RISK_COMPLETION:
            continue
        days = (target - today).days
        when = "today" if days == 0 else f"in {days} day{'s' if days != 1 else ''}"
        out.append(Suggestion(
            kind="goal_at_risk",
            title=f"At risk: {g['title']} — {done}/{total} tasks done, due {when}"
            if total else f"At risk: {g['title']} — no tasks done, due {when}",
            detail=f"Target date {g['target_date']} with {round(pct * 100)}% of its "
                   "tasks complete. Break it down or move the date?",
            confidence=0.75, goal_id=g["id"],
            payload={"key": {"goal": g["id"], "due": g["target_date"]}},
        ))
    return out


def detect_stale_relationships(
    conn, settings: Settings, *, owner_id, now: datetime
) -> list[Suggestion]:
    """A known (named) person not seen in > reconnect_days → a reconnect nudge.

    Confidence scales with how much you've actually talked with them, so a
    one-off vendor doesn't rank like your skip-level.
    """
    cutoff = (now - timedelta(days=settings.proactive.reconnect_days)).strftime(
        "%Y-%m-%dT%H:%M:%fZ"
    )
    opted = registry.opted_out_speaker_ids(conn, settings)
    rows = conn.execute(
        """
        SELECT sp.id, sp.name, sp.display_label,
               MAX(ts.start_at) AS last_seen, COUNT(ts.id) AS seg_count
        FROM speakers sp JOIN transcript_segments ts ON ts.speaker_id = sp.id
        WHERE sp.is_owner=0 AND sp.merged_into IS NULL AND sp.kind='known'
        GROUP BY sp.id HAVING last_seen IS NOT NULL AND last_seen < ?
        """,
        (cutoff,),
    ).fetchall()
    out: list[Suggestion] = []
    for r in rows:
        if r["id"] in opted:
            continue
        name = r["name"] or r["display_label"] or f"Speaker {r['id']}"
        volume = int(r["seg_count"] or 0)
        confidence = round(min(0.8, 0.3 + volume / 200.0), 4)
        detail = f"You haven't spoken with {name} since {(r['last_seen'] or '')[:10]}."
        topic = _last_shared_topic(conn, r["id"])
        if topic:
            detail += f" Last discussed: {topic}."
        out.append(Suggestion(
            kind="relationship_reconnect",
            title=f"Reconnect with {name}",
            detail=detail,
            confidence=confidence,
            payload={"key": {"speaker": r["id"]}, "speaker_id": r["id"]},
        ))
    return out


def _last_shared_topic(conn, speaker_id: int) -> str | None:
    """The most recent non-person entity on an edge touching this person's
    graph node — one cheap indexed lookup; None when there's no node."""
    node = conn.execute(
        "SELECT id FROM kg_nodes WHERE speaker_id=? AND merged_into IS NULL LIMIT 1",
        (speaker_id,),
    ).fetchone()
    if node is None:
        return None
    row = conn.execute(
        """
        SELECT COALESCE(k.display_label, k.name) AS nm
        FROM kg_edges e
        JOIN kg_nodes k ON k.id = CASE WHEN e.src_node_id=? THEN e.dst_node_id
                                       ELSE e.src_node_id END
        WHERE (e.src_node_id=? OR e.dst_node_id=?) AND e.valid=1
          AND k.type IN ('topic','project','organization')
        ORDER BY e.last_seen DESC LIMIT 1
        """,
        (node["id"], node["id"], node["id"]),
    ).fetchone()
    return row["nm"] if row and row["nm"] else None


DETECTORS = [
    detect_commitments, detect_undated_commitments, detect_tasks_due,
    detect_plan_carryover, detect_connections, detect_goal_alignment,
    detect_goal_at_risk, detect_stale_goals, detect_stale_relationships,
]
