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


# --- feedback weights (transparent, no-ML local nudge) -----------------------


def get_feedback_weights(conn: sqlite3.Connection) -> dict[str, float]:
    raw = state.get_state(conn, FEEDBACK_WEIGHTS_KEY)
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def bump_feedback_weight(conn: sqlite3.Connection, kind: str, vote: str) -> None:
    weights = get_feedback_weights(conn)
    cur = weights.get(kind, 1.0)
    cur = min(1.5, cur * 1.05) if vote == "up" else max(0.3, cur * 0.9)
    weights[kind] = round(cur, 4)
    state.set_state(conn, FEEDBACK_WEIGHTS_KEY, json.dumps(weights))


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
    date = digest_date or utcnow_iso()[:10]
    rows = conn.execute(
        "SELECT * FROM suggestions WHERE digest_date=? AND status=? ORDER BY importance DESC",
        (date, status),
    ).fetchall()
    return [dict(r) for r in rows]


def suggestion_action(conn: sqlite3.Connection, suggestion_id: int, action: str) -> None:
    row = conn.execute(
        "SELECT kind, dedupe_hash FROM suggestions WHERE id=?", (suggestion_id,)
    ).fetchone()
    if row is None:
        return
    if action in ("dismiss", "done"):
        new = "dismissed" if action == "dismiss" else "done"
        conn.execute("UPDATE suggestions SET status=? WHERE id=?", (new, suggestion_id))
    elif action == "snooze":
        snooze_kind(conn, row["kind"])
        conn.execute("UPDATE suggestions SET status='snoozed' WHERE id=?", (suggestion_id,))
    elif action in ("up", "down"):
        conn.execute(
            "INSERT INTO suggestion_feedback (suggestion_id, dedupe_hash, kind, vote) "
            "VALUES (?, ?, ?, ?)",
            (suggestion_id, row["dedupe_hash"], row["kind"], action),
        )
        bump_feedback_weight(conn, row["kind"], action)


# --- digests -----------------------------------------------------------------


def save_digest(
    conn: sqlite3.Connection, digest_date: str, kind: str, summary_md: str,
    suggestion_ids: list[int], model: str | None, backend: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO digests (digest_date, kind, summary_md, suggestion_ids, model, backend)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(digest_date, kind) DO UPDATE SET
            summary_md=excluded.summary_md, suggestion_ids=excluded.suggestion_ids,
            model=excluded.model, backend=excluded.backend
        """,
        (digest_date, kind, summary_md, json.dumps(suggestion_ids), model, backend),
    )


def get_digest(conn: sqlite3.Connection, digest_date: str, kind: str = "daily") -> dict | None:
    row = conn.execute(
        "SELECT * FROM digests WHERE digest_date=? AND kind=?", (digest_date, kind)
    ).fetchone()
    return dict(row) if row else None


def digest_count(conn: sqlite3.Connection, digest_date: str | None = None) -> int:
    date = digest_date or utcnow_iso()[:10]
    return conn.execute(
        "SELECT COUNT(*) AS n FROM suggestions WHERE digest_date=? AND status='open'", (date,)
    ).fetchone()["n"]
