"""SQLite connection management and runtime index setup."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.storage.schema import apply_base_schema

DbPath = Path | str | None


def connect(db_path: DbPath = None, *, settings: Settings | None = None) -> sqlite3.Connection:
    """Open a tuned SQLite connection (WAL, foreign keys, dict-like rows)."""
    settings = settings or get_settings()
    path = Path(db_path) if db_path is not None else settings.db_path
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: DbPath = None, *, settings: Settings | None = None) -> sqlite3.Connection:
    """Connect and ensure the base schema exists. Safe to call repeatedly."""
    conn = connect(db_path, settings=settings)
    apply_base_schema(conn)
    return conn


@contextmanager
def db_session(
    db_path: DbPath = None, *, settings: Settings | None = None
) -> Iterator[sqlite3.Connection]:
    """Context-managed connection that always closes."""
    conn = connect(db_path, settings=settings)
    try:
        yield conn
    finally:
        conn.close()


def try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load the sqlite-vec extension. Returns True on success.

    Semantic search degrades gracefully to full-text-only when this fails (e.g.
    the extension isn't installed, as on a minimal CI box).
    """
    try:
        import sqlite_vec  # type: ignore
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except (AttributeError, sqlite3.OperationalError):
        # enable_load_extension may be compiled out of the bundled sqlite3.
        return False
