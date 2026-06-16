"""Runtime key/value state shared across the daemon, API, and menu bar.

The pause toggle lives here (not in the TOML config) so the menu bar / API can
flip it live and the recorder picks it up immediately. A DB value overrides the
static config default when present.
"""

from __future__ import annotations

import sqlite3

from secondbrain.storage.models import utcnow_iso

PAUSED = "recording_paused"


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
    return val == "1"


def set_paused(conn: sqlite3.Connection, paused: bool) -> None:
    set_state(conn, PAUSED, "1" if paused else "0")
