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
from secondbrain.llm.jsonout import parse_json
from secondbrain.proactive import ranking, store
from secondbrain.proactive.detectors import DETECTORS, Suggestion, owner_node_id
from secondbrain.speaker import registry

JOB_PROACTIVE = "generate_digest"
DAILY_RUN_KEY = "proactive_last_daily"
WEEKLY_RUN_KEY = "proactive_last_weekly"

_DAILY_SYSTEM = (
    "Write a short, warm morning brief for the user, grouped into Commitments, "
    "Goals, Connections, and Coaching (omit empty groups). Use ONLY the provided "
    "items; cite each point with its [seg_id]. Phrase uncertain items as gentle "
    "suggestions, not assertions."
)
_WEEKLY_SYSTEM = (
    "Write a concise weekly review: goal progress, open/overdue commitments, "
    "blockers, and notable connections. Use ONLY the provided items and cite "
    "[seg_id] where present."
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
        data = parse_json(llm.complete(system=_COACHING_SYSTEM, prompt=transcript).text)
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


def run_digest(
    conn: sqlite3.Connection,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
    kind: str = "daily",
    date: str | None = None,
) -> dict | None:
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    now = datetime.now(UTC)
    digest_date = date or now.strftime("%Y-%m-%d")

    work = settings
    if kind == "weekly":
        work = settings.model_copy(deep=True)
        work.proactive.recent_days = max(7, settings.proactive.recent_days)

    owner_id = owner_node_id(conn)

    # keep goal links fresh before alignment detection
    from secondbrain.goals.link import relink_goal

    for g in conn.execute("SELECT id FROM goals WHERE status='active'").fetchall():
        relink_goal(conn, g["id"], work)

    suggestions: list[Suggestion] = []
    for detect in DETECTORS:
        suggestions.extend(detect(conn, work, owner_id=owner_id, now=now))
    if settings.proactive.coaching_enabled:
        suggestions.extend(_coaching(conn, work, llm, now))

    ranked = ranking.rank(conn, suggestions, work, now=now)
    ids = store.persist_suggestions(conn, digest_date, ranked)

    summary, model, backend = _synthesize(conn, ranked, llm, settings, kind)
    store.save_digest(conn, digest_date, kind, summary, ids, model, backend)
    return store.get_digest(conn, digest_date, kind)


def _synthesize(conn, ranked: list[Suggestion], llm: LLM, settings: Settings, kind: str):
    if not ranked:
        return ("Nothing notable to surface today.", None, None)
    lines = []
    for s in ranked:
        cites = " ".join(f"[{c}]" for c in s.citations)
        lines.append(f"- ({s.kind}) {s.title}: {s.detail} {cites}".rstrip())
    context = "\n".join(lines)[: settings.extraction.chat_max_context_chars]
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
