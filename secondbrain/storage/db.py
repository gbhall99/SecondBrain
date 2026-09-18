"""SQLite connection management and runtime index setup."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.storage.schema import apply_base_schema

DbPath = Path | str | None


def _sqlcipher_dbapi():
    """Return the SQLCipher DB-API module if a driver is installed, else None.

    Prefers ``sqlcipher3`` (the ``[secure]`` extra); falls back to the older
    ``pysqlcipher3`` for environments that already have it.
    """
    try:
        import sqlcipher3.dbapi2 as mod  # type: ignore

        return mod
    except ImportError:
        try:
            import pysqlcipher3.dbapi2 as mod  # type: ignore

            return mod
        except ImportError:
            return None


def sqlcipher_available() -> bool:
    """True if a SQLCipher Python driver is importable (the `secure` extra)."""
    return _sqlcipher_dbapi() is not None


def _sqlite_module(settings: Settings):
    """Return the DBAPI module to use: SQLCipher when encryption is enabled,
    else the stdlib sqlite3 (the CI/default path)."""
    if not settings.security.encrypt_db:
        return sqlite3
    mod = _sqlcipher_dbapi()
    if mod is None:
        raise RuntimeError(
            "security.encrypt_db is true but no SQLCipher driver is installed. "
            "Install with: pip install -e '.[secure]'"
        )
    return mod


def _driver_errors(name: str) -> tuple[type[Exception], ...]:
    """Exception classes named ``name`` across every driver we may open a
    connection with. The SQLCipher drivers define their own DB-API exception
    hierarchy that does NOT subclass the stdlib ``sqlite3`` one, so code that
    catches database errors must catch both or it silently stops working the
    moment ``security.encrypt_db`` is turned on."""
    classes: list[type[Exception]] = [getattr(sqlite3, name)]
    mod = _sqlcipher_dbapi()
    if mod is not None:
        classes.append(getattr(mod, name))
    return tuple(classes)


#: ``except DB_ERRORS:`` — any database error, from whichever driver opened the connection.
DB_ERRORS = _driver_errors("Error")
#: ``except OPERATIONAL_ERRORS:`` — e.g. a missing table/extension, from whichever driver.
OPERATIONAL_ERRORS = _driver_errors("OperationalError")


def _sql_string_literal(value: str) -> str:
    """Escape ``value`` for embedding inside a single-quoted SQL string literal."""
    return value.replace("'", "''")


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
    # The driver's own Row class: stdlib sqlite3.Row rejects a SQLCipher cursor.
    conn.row_factory = module.Row
    if settings.security.encrypt_db:
        passphrase = settings.security.db_passphrase
        if not passphrase:
            raise RuntimeError("security.encrypt_db is true but security.db_passphrase is empty")
        try:
            # SQLite PRAGMAs cannot take bound parameters (``PRAGMA key = ?`` is a
            # syntax error), so the passphrase must be inlined as a SQL string
            # literal. Doubling single quotes is the complete escaping rule for
            # SQL string literals, so any passphrase is safe here.
            conn.execute(f"PRAGMA key = '{_sql_string_literal(passphrase)}'")
        except Exception as exc:  # noqa: BLE001 - never surface the passphrase in a traceback
            raise RuntimeError(
                f"SQLCipher key setup failed ({type(exc).__name__}; check db_passphrase)"
            ) from None
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


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Group writes into one atomic unit (BEGIN IMMEDIATE / COMMIT / ROLLBACK).

    Connections are opened in autocommit mode (``isolation_level=None``), so each
    statement otherwise commits on its own; wrapping a multi-step write here makes
    it all-or-nothing if an error or crash hits mid-sequence. Reentrant: if a
    transaction is already active it yields without starting a nested one (SQLite
    has none), so callers can wrap helpers that are also used standalone.
    """
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


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
    except (AttributeError, *OPERATIONAL_ERRORS):
        # enable_load_extension may be compiled out of the bundled sqlite3.
        return False
