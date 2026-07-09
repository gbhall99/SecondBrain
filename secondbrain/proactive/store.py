"""Persistence + feedback/snooze state for the proactive engine."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from secondbrain.config import Settings
from secondbrain.proactive.detectors import Suggestion
from secondbrain.storage import state
from secondbrain.storage.models import utcnow_iso

FEEDBACK_WEIGHTS_KEY = "proactive_feedback_weights"
SNOOZE_PREFIX = "proactive_snooze:"
GENERATING_PREFIX = "proactive_generating:"
GENERATING_STALE_S = 15 * 60  # markers older than this are crash leftovers


def _local_today() -> str:
    """Today (YYYY-MM-DD) on the machine's wall clock — digest_date is a local
    calendar day, matching how the owner experiences 'today'."""
    return datetime.now().astimezone().strftime("%Y-%m-%d")


# --- feedback weights (transparent, no-ML local nudge) -----------------------


def get_feedback_weights(conn: sqlite3.Connection) -> dict[str, float]:
    raw = state.get_state(conn, FEEDBACK_WEIGHTS_KEY)
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def _nudge_weight(conn: sqlite3.Connection, kind: str, factor: float) -> None:
    weights = get_feedback_weights(conn)
    cur = weights.get(kind, 1.0) * factor
    weights[kind] = round(max(0.3, min(1.5, cur)), 4)
    state.set_state(conn, FEEDBACK_WEIGHTS_KEY, json.dumps(weights))


def bump_feedback_weight(conn: sqlite3.Connection, kind: str, vote: str) -> None:
    _nudge_weight(conn, kind, 1.05 if vote == "up" else 0.9)


def unbump_feedback_weight(conn: sqlite3.Connection, kind: str, vote: str) -> None:
    """Neutralize one earlier bump (inverse factor, same clamps) so switching a
    mis-clicked vote doesn't leave a permanent penalty/boost behind."""
    _nudge_weight(conn, kind, 1 / 1.05 if vote == "up" else 1 / 0.9)


# --- snooze ------------------------------------------------------------------


def snooze_kind(conn: sqlite3.Connection, kind: str, days: int = 7) -> None:
    until = (datetime.now(UTC) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%fZ")
    state.set_state(conn, SNOOZE_PREFIX + kind, until)


def snoozed_kinds(conn: sqlite3.Connection, now_iso: str) -> set[str]:
    out = set()
    for r in conn.execute(
        "SELECT key, value FROM app_state WHERE key LIKE ?", (SNOOZE_PREFIX + "%",)
    ).fetchall():
        if (r["value"] or "") > now_iso:
            out.add(r["key"][len(SNOOZE_PREFIX):])
    return out


# --- cross-day suppression ---------------------------------------------------


def suppressed_hashes(conn: sqlite3.Connection, settings: Settings, now_iso: str) -> set[str]:
    cutoff = (
        datetime.strptime(now_iso, "%Y-%m-%dT%H:%M:%fZ")
        - timedelta(days=settings.proactive.suppress_days)
    ).strftime("%Y-%m-%dT%H:%M:%fZ")
    out = set()
    for r in conn.execute(
        "SELECT DISTINCT dedupe_hash FROM suggestions "
        "WHERE status IN ('dismissed','done') AND dedupe_hash IS NOT NULL AND created_at >= ?",
        (cutoff,),
    ).fetchall():
        out.add(r["dedupe_hash"])
    for r in conn.execute(
        "SELECT DISTINCT dedupe_hash FROM suggestion_feedback "
        "WHERE vote='down' AND dedupe_hash IS NOT NULL AND created_at >= ?",
        (cutoff,),
    ).fetchall():
        out.add(r["dedupe_hash"])
    return out


# --- suggestion persistence --------------------------------------------------


def persist_suggestions(
    conn: sqlite3.Connection, digest_date: str, suggestions: list[Suggestion]
) -> list[int]:
    ids = []
    for s in suggestions:
        conn.execute(
            """
            INSERT OR IGNORE INTO suggestions
                (digest_date, kind, title, detail, payload, citations, importance,
                 confidence, goal_id, dedupe_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest_date, s.kind, s.title, s.detail, json.dumps(s.payload),
                json.dumps(s.citations), s.importance, s.confidence, s.goal_id, s.dedupe_hash,
            ),
        )
        row = conn.execute(
            "SELECT id FROM suggestions WHERE digest_date=? AND dedupe_hash=?",
            (digest_date, s.dedupe_hash),
        ).fetchone()
        if row:
            ids.append(int(row["id"]))
    return ids


def list_suggestions(
    conn: sqlite3.Connection, digest_date: str | None = None, status: str = "open"
) -> list[dict]:
    date = digest_date or _local_today()
    rows = conn.execute(
        # `voted` (latest thumbs vote, if any) lets the UI keep 👍/👎 pressed
        # across reloads. Additive column; existing consumers ignore it.
        """
        SELECT s.*, (SELECT f.vote FROM suggestion_feedback f
                     WHERE f.suggestion_id = s.id ORDER BY f.id DESC LIMIT 1) AS voted
        FROM suggestions s WHERE s.digest_date=? AND s.status=?
        ORDER BY s.importance DESC, s.id
        """,
        (date, status),
    ).fetchall()
    return [dict(r) for r in rows]


def suggestion_action(conn: sqlite3.Connection, suggestion_id: int, action: str) -> bool:
    """Apply ``action`` to a suggestion. Returns False when the id doesn't exist."""
    row = conn.execute(
        "SELECT kind, dedupe_hash, digest_date, status FROM suggestions WHERE id=?",
        (suggestion_id,),
    ).fetchone()
    if row is None:
        return False
    if action in ("dismiss", "done"):
        new = "dismissed" if action == "dismiss" else "done"
        conn.execute("UPDATE suggestions SET status=? WHERE id=?", (new, suggestion_id))
    elif action == "snooze":
        # Snooze promises to hide ALL items of this kind: suppress future
        # detections for 7 days AND park today's open same-kind siblings, not
        # just the acted row.
        snooze_kind(conn, row["kind"])
        conn.execute(
            "UPDATE suggestions SET status='snoozed' "
            "WHERE digest_date=? AND kind=? AND status='open'",
            (row["digest_date"], row["kind"]),
        )
        conn.execute("UPDATE suggestions SET status='snoozed' WHERE id=?", (suggestion_id,))
    elif action == "reopen":
        # Undo for done/dismiss/snooze. Reopening a snoozed item lifts the
        # kind-wide snooze — you asked to see this kind again — and restores
        # the same-day siblings that snoozing hid (symmetric with snooze).
        was_snoozed = row["status"] == "snoozed"
        conn.execute("UPDATE suggestions SET status='open' WHERE id=?", (suggestion_id,))
        if was_snoozed:
            conn.execute(
                "UPDATE suggestions SET status='open' "
                "WHERE digest_date=? AND kind=? AND status='snoozed'",
                (row["digest_date"], row["kind"]),
            )
        state.set_state(conn, SNOOZE_PREFIX + row["kind"], "")
    elif action in ("up", "down"):
        prev = conn.execute(
            "SELECT vote FROM suggestion_feedback WHERE suggestion_id=? "
            "ORDER BY id DESC LIMIT 1",
            (suggestion_id,),
        ).fetchone()
        prev_vote = prev["vote"] if prev else None
        if prev_vote == action:
            return True  # same vote again: nothing changes (idempotent)
        if prev_vote is not None:
            # A flip is a correction: drop the old vote's rows (a stale 'down'
            # would keep suppressing this item via suppressed_hashes) and
            # neutralize its weight nudge before applying the new one.
            conn.execute(
                "DELETE FROM suggestion_feedback WHERE suggestion_id=?", (suggestion_id,)
            )
            unbump_feedback_weight(conn, row["kind"], prev_vote)
        conn.execute(
            "INSERT INTO suggestion_feedback (suggestion_id, dedupe_hash, kind, vote) "
            "VALUES (?, ?, ?, ?)",
            (suggestion_id, row["dedupe_hash"], row["kind"], action),
        )
        bump_feedback_weight(conn, row["kind"], action)
    return True


# --- digests -----------------------------------------------------------------


def save_digest(
    conn: sqlite3.Connection, digest_date: str, kind: str, summary_md: str,
    suggestion_ids: list[int], model: str | None, backend: str | None,
) -> None:
    # Regenerating refreshes created_at: on quiet days the summary text can be
    # identical, so the "Generated <when>" stamp is the only visible proof that
    # a 1–2 minute regenerate actually did anything.
    conn.execute(
        """
        INSERT INTO digests (digest_date, kind, summary_md, suggestion_ids, model, backend)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(digest_date, kind) DO UPDATE SET
            summary_md=excluded.summary_md, suggestion_ids=excluded.suggestion_ids,
            model=excluded.model, backend=excluded.backend,
            created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
        """,
        (digest_date, kind, summary_md, json.dumps(suggestion_ids), model, backend),
    )


def get_digest(conn: sqlite3.Connection, digest_date: str, kind: str = "daily") -> dict | None:
    row = conn.execute(
        "SELECT * FROM digests WHERE digest_date=? AND kind=?", (digest_date, kind)
    ).fetchone()
    return dict(row) if row else None


def list_digest_dates(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Dates (newest first) that have a stored digest, per kind — powers the
    brief page's history navigation."""
    out: dict[str, list[str]] = {"daily": [], "weekly": []}
    for r in conn.execute(
        "SELECT digest_date, kind FROM digests ORDER BY digest_date DESC"
    ).fetchall():
        out.setdefault(r["kind"], []).append(r["digest_date"])
    return out


def digest_count(conn: sqlite3.Connection, digest_date: str | None = None) -> int:
    date = digest_date or _local_today()
    return conn.execute(
        "SELECT COUNT(*) AS n FROM suggestions WHERE digest_date=? AND status='open'", (date,)
    ).fetchone()["n"]


# --- in-flight generation marker ----------------------------------------------
# Generation is a 1–2 minute synchronous LLM run. The marker (an app_state key,
# committed immediately — connections are autocommit) lets the web UI resume its
# progress line after a reload and lets concurrent triggers (second tab, daemon
# job, CLI) refuse to start an overlapping run on the same digest row.


def mark_generating(conn: sqlite3.Connection, kind: str) -> None:
    state.set_state(conn, GENERATING_PREFIX + kind, utcnow_iso())


def clear_generating(conn: sqlite3.Connection, kind: str) -> None:
    state.set_state(conn, GENERATING_PREFIX + kind, "")


def generating_since(conn: sqlite3.Connection, kind: str) -> str | None:
    """Started-at (UTC ISO) of an in-flight digest run for ``kind``, or None.

    Markers older than GENERATING_STALE_S are leftovers from a crashed run
    (clear_generating runs in a ``finally``) and are ignored.
    """
    raw = state.get_state(conn, GENERATING_PREFIX + kind) or ""
    if not raw:
        return None
    try:
        started = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    if (datetime.now(UTC) - started).total_seconds() > GENERATING_STALE_S:
        return None
    return raw
