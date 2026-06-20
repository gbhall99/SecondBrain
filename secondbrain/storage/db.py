"""SQLite connection management and runtime index setup."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.storage.schema import apply_base_schema

DbPath = Path | str | None


def sqlcipher_available() -> bool:
    """True if a SQLCipher Python driver is importable (the `secure` extra)."""
    try:
        import sqlcipher3  # type: ignore  # noqa: F401

        return True
    except ImportError:
        try:
            import pysqlcipher3.dbapi2  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False


def _sqlite_module(settings: Settings):
    """Return the DBAPI module to use: SQLCipher when encryption is enabled,
    else the stdlib sqlite3 (the CI/default path)."""
    if not settings.security.encrypt_db:
        return sqlite3
    try:
        import sqlcipher3.dbapi2 as mod  # type: ignore

        return mod
    except ImportError:
        try:
            import pysqlcipher3.dbapi2 as mod  # type: ignore

            return mod
        except ImportError as exc:
            raise RuntimeError(
                "security.encrypt_db is true but no SQLCipher driver is installed. "
                "Install with: pip install -e '.[secure]'"
            ) from exc


def connect(db_path: DbPath = None, *, settings: Settings | None = None) -> sqlite3.Connection:
    """Open a tuned SQLite connection (WAL, foreign keys, dict-like rows).

    When ``security.encrypt_db`` is set, opens via SQLCipher and applies
    ``PRAGMA key`` before any other statement.
    """
    settings = settings or get_settings()
    path = Path(db_path) if db_path is not None else settings.db_path
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    module = _sqlite_module(settings)
    conn = module.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if settings.security.encrypt_db:
        passphrase = settings.security.db_passphrase
        if not passphrase:
            raise RuntimeError("security.encrypt_db is true but security.db_passphrase is empty")
        try:
            conn.execute("PRAGMA key = ?", (passphrase,))
        except Exception:  # noqa: BLE001 - never surface the passphrase in a traceback
            raise RuntimeError("SQLCipher key setup failed (check db_passphrase)") from None
        # At-rest hygiene: v4 page format (encrypts the WAL too) + scrub freed pages.
        conn.execute("PRAGMA cipher_compatibility = 4")
        conn.execute("PRAGMA secure_delete = ON")
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
