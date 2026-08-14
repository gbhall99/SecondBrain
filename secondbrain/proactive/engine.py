"""Proactive engine: run detectors → rank → persist → synthesize the brief.

Detection/ranking are deterministic; the LLM is used only to write the brief
prose and (opt-in) coaching. Mockable end-to-end for CI.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.knowledge import chat as chatmod
from secondbrain.llm.client import LLM, get_llm
from secondbrain.llm.jsonout import complete_json
from secondbrain.proactive import ranking, store
from secondbrain.proactive.detectors import DETECTORS, Suggestion, owner_node_id
from secondbrain.speaker import registry

JOB_PROACTIVE = "generate_digest"
DAILY_RUN_KEY = "proactive_last_daily"
WEEKLY_RUN_KEY = "proactive_last_weekly"
# Highest kg_edges id already scanned for goal links by a digest run — the
# next run relinks incrementally from here instead of re-scoring the graph.
RELINK_EDGE_KEY = "proactive_relink_edge_id"


class DigestInFlight(RuntimeError):
    """Another digest generation for this kind is already running.

    Raised instead of starting a second 1–2 minute LLM run that would race the
    first on the same digest row (web regenerate vs. a reloaded tab, the
    daemon's scheduled job, or the CLI). ``started_at`` is the UTC ISO stamp of
    the in-flight run.
    """

    def __init__(self, kind: str, started_at: str):
        self.kind = kind
        self.started_at = started_at
        super().__init__(
            f"a {'weekly review' if kind == 'weekly' else 'daily brief'} "
            f"is already being generated (started {started_at})"
        )

_DAILY_SYSTEM = (
    "Write a short, warm morning brief for the user, grouped into Commitments, "
    "Goals, Connections, and Coaching (omit empty groups). Use ONLY the provided "
    "items; cite each point with its [seg_id]. Phrase uncertain items as gentle "
    "suggestions, not assertions."
)
_WEEKLY_SYSTEM = (
    "Write a concise weekly review: goal progress, open/overdue commitments, "
    "blockers, and notable connections. Use ONLY the provided items and stats "
    "and cite [seg_id] where present."
)
_COACHING_SYSTEM = (
    "You are a candid but constructive coach. From the user's own recent statements, "
    "give at most 2 specific, actionable observations to help them be a better team "
    "member (e.g. unaddressed concerns, missed follow-ups, talk-time). Be direct. "
    "Cite each with the segment id. Output JSON "
    '{"observations":[{"text":"...","source_segment_ids":[..]}]} or {"observations":[]}.'
)


def _coaching(conn, settings: Settings, llm: LLM, now: datetime) -> list[Suggestion]:
    cut = (now - timedelta(days=settings.proactive.recent_days)).strftime("%Y-%m-%dT%H:%M:%fZ")
    rows = conn.execute(
        """
        SELECT ts.id, ts.text FROM transcript_segments ts
        JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE sp.is_owner=1 AND ts.start_at >= ? AND ts.text <> ?
        ORDER BY ts.start_at
        """,
        (cut, registry.REDACTED_TEXT),
    ).fetchall()
    if not rows:
        return []
    transcript = "\n".join(f"[seg_id={r['id']}] {r['text']}" for r in rows)
    try:
        data = complete_json(llm, system=_COACHING_SYSTEM, prompt=transcript)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for o in data.get("observations", []):
        cites = [int(c) for c in o.get("source_segment_ids", []) if str(c).isdigit()]
        if not cites:
            continue
        out.append(Suggestion(
            kind="coaching", title="Coaching", detail=o.get("text", ""),
            confidence=0.7, citations=cites, payload={"key": {"obs": o.get("text", "")[:60]}},
        ))
    return out


def week_monday(day: str) -> str:
    """The Monday of the week containing ``day`` (YYYY-MM-DD).

    Weekly digests are keyed to it so two generations in one week update the
    same row instead of minting a duplicate per run-day.
    """
    d = datetime.strptime(day, "%Y-%m-%d").date()
    return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def weekly_stats(conn: sqlite3.Connection, now: datetime) -> dict:
    """Deterministic aggregates for the weekly review (last 7 days).

    Computed from the DB, not the LLM — rendered as a stats row even when the
    prose ignores them, and stored in the digest payload.
    """
    ts_cut = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    day_cut = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")
    one = lambda sql, *p: conn.execute(sql, p).fetchone()[0]  # noqa: E731
    tasks_completed = one(
        "SELECT COUNT(*) FROM tasks WHERE status='done' AND completed_at >= ?", ts_cut
    )
    commitments_added = one(
        "SELECT COUNT(*) FROM kg_edges WHERE kind='action_item' "
        "AND COALESCE(first_seen, created_at) >= ?", ts_cut
    )
    commitments_cleared = one(
        "SELECT COUNT(*) FROM tasks WHERE source_edge_id IS NOT NULL "
        "AND status='done' AND completed_at >= ?", ts_cut
    )
    goals_progressed = one(
        "SELECT COUNT(*) FROM goals WHERE status='active' AND last_progress_at >= ?", ts_cut
    )
    meeting_seconds = one(
        "SELECT COALESCE(SUM((julianday(ended_at) - julianday(started_at)) * 86400.0), 0) "
        "FROM conversations WHERE started_at >= ? AND ended_at IS NOT NULL", ts_cut
    )
    # Plan adherence: of the tasks planned into a day this week (still pinned
    # via scheduled_for, or released with last_planned_for), how many got done.
    planned = conn.execute(
        """
        SELECT status FROM tasks
        WHERE (scheduled_for >= ? AND scheduled_for <= ?)
           OR (last_planned_for >= ? AND last_planned_for <= ?)
        """,
        (day_cut, today, day_cut, today),
    ).fetchall()
    adherence = (
        round(100 * sum(1 for r in planned if r["status"] == "done") / len(planned))
        if planned else None
    )
    return {
        "tasks_completed": int(tasks_completed),
        "commitments_added": int(commitments_added),
        "commitments_cleared": int(commitments_cleared),
        "goals_progressed": int(goals_progressed),
        "meeting_hours": round((meeting_seconds or 0.0) / 3600.0, 1),
        "plan_adherence_pct": adherence,
        "planned_tasks": len(planned),
    }


def _stats_lines(stats: dict) -> str:
    parts = [
        f"- Tasks completed this week: {stats['tasks_completed']}",
        f"- Commitments cleared vs added: {stats['commitments_cleared']} "
        f"vs {stats['commitments_added']}",
        f"- Goals with progress: {stats['goals_progressed']}",
        f"- Meeting hours: {stats['meeting_hours']}",
    ]
    if stats.get("plan_adherence_pct") is not None:
        parts.append(f"- Plan adherence: {stats['plan_adherence_pct']}% "
                     f"of {stats['planned_tasks']} planned tasks done")
    return "\n".join(parts)


def run_digest(
    conn: sqlite3.Connection,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
    kind: str = "daily",
    date: str | None = None,
) -> dict | None:
    settings = settings or get_settings()
    now = datetime.now(UTC)
    # digest_date is a *local* calendar day: digest_hour is documented as a
    # local hour, and the brief page labels "Today" from the local wall clock.
    digest_date = date or datetime.now().astimezone().strftime("%Y-%m-%d")
    if kind == "weekly":
        digest_date = week_monday(digest_date)

    in_flight = store.generating_since(conn, kind)
    if in_flight:
        raise DigestInFlight(kind, in_flight)
    store.mark_generating(conn, kind)
    try:
        llm = llm or get_llm(settings)
        work = settings
        if kind == "weekly":
            work = settings.model_copy(deep=True)
            work.proactive.recent_days = max(7, settings.proactive.recent_days)

        owner_id = owner_node_id(conn)

        # Keep goal links fresh before alignment detection — incrementally:
        # only edges newer than the last digest's high-water mark are
        # re-scored, so the morning brief doesn't rescan the whole graph.
        from secondbrain.goals.link import relink_goal
        from secondbrain.storage import state as app_state

        last_raw = app_state.get_state(conn, RELINK_EDGE_KEY) or ""
        since = int(last_raw) if last_raw.isdigit() else None
        max_edge = conn.execute("SELECT COALESCE(MAX(id), 0) FROM kg_edges").fetchone()[0]
        for g in conn.execute("SELECT id FROM goals WHERE status='active'").fetchall():
            relink_goal(conn, g["id"], work, since_edge_id=since)
        app_state.set_state(conn, RELINK_EDGE_KEY, str(int(max_edge)))

        suggestions: list[Suggestion] = []
        for detect in DETECTORS:
            suggestions.extend(detect(conn, work, owner_id=owner_id, now=now))
        if settings.proactive.coaching_enabled:
            suggestions.extend(_coaching(conn, work, llm, now))

        # rank() returns EVERY scored suggestion (persisted below); the top_n /
        # per-kind display cut happens at render time so nothing is lost.
        ranked = ranking.rank(conn, suggestions, work, now=now)
        ids = store.persist_suggestions(conn, digest_date, ranked)

        stats = weekly_stats(conn, now) if kind == "weekly" else None
        summary, model, backend = _synthesize(conn, ranked, llm, settings, kind, stats=stats)
        store.save_digest(conn, digest_date, kind, summary, ids, model, backend,
                          payload={"stats": stats} if stats else {})
        return store.get_digest(conn, digest_date, kind)
    finally:
        store.clear_generating(conn, kind)


def _synthesize(conn, ranked: list[Suggestion], llm: LLM, settings: Settings, kind: str,
                stats: dict | None = None):
    if not ranked and not stats:
        if kind == "weekly":
            return ("A quiet week — nothing notable to review.", None, None)
        return ("Nothing notable to surface today.", None, None)
    # The prose covers what the page shows above the fold (the capped list);
    # overflow items are still persisted and revealable on the page.
    visible, _ = ranking.apply_caps(ranked, settings)
    lines = []
    for s in visible:
        cites = " ".join(f"[{c}]" for c in s.citations)
        lines.append(f"- ({s.kind}) {s.title}: {s.detail} {cites}".rstrip())
    context = "\n".join(lines)[: settings.extraction.chat_max_context_chars]
    if stats:
        context = "This week's numbers (deterministic; weave them in):\n" \
            + _stats_lines(stats) + "\n\nItems:\n" + context
    system = _WEEKLY_SYSTEM if kind == "weekly" else _DAILY_SYSTEM
    resp = llm.complete(system=system, prompt=context)

    # keep only citations that map to real segments (drop hallucinated ids)
    allowed = {c for s in ranked for c in s.citations}
    cited = {int(m) for m in chatmod._CITE.findall(resp.text)}
    unknown = cited - allowed
    summary = resp.text
    if unknown:
        for u in unknown:
            summary = summary.replace(f"[{u}]", "").replace(f"[seg_id={u}]", "")
    return (summary.strip(), resp.model, resp.backend)


# --- daemon scheduling helpers ----------------------------------------------


def due_daily(conn, settings: Settings, now: datetime) -> bool:
    from secondbrain.storage import state

    today = now.strftime("%Y-%m-%d")
    last = (state.get_state(conn, DAILY_RUN_KEY) or "")[:10]
    return last != today and now.hour >= settings.proactive.digest_hour


def due_weekly(conn, settings: Settings, now: datetime) -> bool:
    from secondbrain.storage import state

    if now.weekday() != settings.proactive.weekly_review_weekday:
        return False
    week = now.strftime("%Y-W%W")
    last = state.get_state(conn, WEEKLY_RUN_KEY) or ""
    return last != week and now.hour >= settings.proactive.digest_hour
