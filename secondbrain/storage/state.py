"""Runtime key/value state shared across the daemon, API, and menu bar.

The pause toggle lives here (not in the TOML config) so the menu bar / API can
flip it live and the recorder picks it up immediately. A DB value overrides the
static config default when present.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from secondbrain.storage.models import parse_iso, utcnow_iso

PAUSED = "recording_paused"
PAUSE_UNTIL = "pause_until"


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO app_state(key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, utcnow_iso()),
    )


def is_paused(conn: sqlite3.Connection, default: bool = False) -> bool:
    val = get_state(conn, PAUSED)
    if val is None:
        return default
    if val != "1":
        return False
    # A timed pause ("Pause 15 min" in the menu bar) carries an expiry; once it
    # passes, recording auto-resumes on the next check.
    until = get_state(conn, PAUSE_UNTIL)
    if until:
        try:
            expired = parse_iso(until) <= datetime.now(UTC)
        except ValueError:
            expired = False
        if expired:
            set_paused(conn, False)
            return False
    return True


def set_paused(conn: sqlite3.Connection, paused: bool, until_iso: str | None = None) -> None:
    """Flip the pause toggle. ``until_iso`` (UTC ISO) makes it a timed pause
    that auto-expires; omitted/None means "until resumed"."""
    set_state(conn, PAUSED, "1" if paused else "0")
    set_state(conn, PAUSE_UNTIL, until_iso if (paused and until_iso) else "")


def pause_changed_at(conn: sqlite3.Connection) -> str | None:
    """UTC ISO time the pause toggle last flipped; None if it never has.

    Used as a grace window for capture-staleness reporting: right after a
    resume the recorder legitimately hasn't written a chunk yet.
    """
    row = conn.execute("SELECT updated_at FROM app_state WHERE key=?", (PAUSED,)).fetchone()
    return row["updated_at"] if row else None
