"""Health checks: `sb doctor` preflight + the /health endpoint.

Every check is best-effort and degradable — a failing dependency yields an
``ok=False`` Check, never an exception — so this is safe to call anywhere and
testable with mocked backends.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from secondbrain.config import Settings, get_settings
from secondbrain.storage import retention, state
from secondbrain.storage.schema import SCHEMA_VERSION


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _migration(conn: sqlite3.Connection) -> Check:
    try:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        ver = row["version_num"] if row else None
        return Check("migrations", ver == SCHEMA_VERSION, f"{ver} (head {SCHEMA_VERSION})")
    except sqlite3.Error as exc:
        return Check("migrations", False, str(exc))


def _disk(settings: Settings) -> Check:
    try:
        free = round(retention.free_disk_gb(settings.data_path), 2)
        return Check("disk", retention.disk_ok(settings), f"{free} GB free")
    except OSError as exc:
        return Check("disk", False, str(exc))


def _counts(conn: sqlite3.Connection) -> Check:
    try:
        n = conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"]
        spk = conn.execute("SELECT COUNT(*) AS n FROM speakers").fetchone()["n"]
        return Check("database", True, f"{n} segments, {spk} speakers")
    except sqlite3.Error as exc:
        return Check("database", False, str(exc))


def _llm(settings: Settings) -> Check:
    if settings.llm.backend != "ollama":
        return Check("llm", True, f"backend={settings.llm.backend}")
    try:
        import httpx

        r = httpx.get(f"{settings.llm.host}/api/tags", timeout=2.0)
        return Check("llm", r.status_code == 200, f"ollama {r.status_code}")
    except Exception as exc:  # noqa: BLE001 - reachability is best-effort
        return Check("llm", False, f"ollama unreachable: {exc}")


def _encryption(settings: Settings) -> Check:
    if not settings.security.encrypt_db:
        return Check("encryption", True, "disabled (FileVault recommended)")
    from secondbrain.storage import db

    has_driver = db.sqlcipher_available()
    has_pass = bool(settings.security.db_passphrase)
    return Check("encryption", has_driver and has_pass,
                 f"sqlcipher={'ok' if has_driver else 'missing'}, "
                 f"passphrase={'set' if has_pass else 'missing'}")


def _recording(conn: sqlite3.Connection, settings: Settings) -> Check:
    paused = state.is_paused(conn, default=settings.consent.paused)
    on = settings.consent.recording_enabled and not paused
    return Check("recording", True, "on" if on else "paused/off")


def _daemon(conn: sqlite3.Connection) -> Check:
    """Report whether the daemon's loops are alive (via their heartbeats)."""
    from datetime import UTC, datetime

    from secondbrain.storage.models import parse_iso

    beats = {
        name: state.get_state(conn, f"heartbeat:{name}") for name in ("worker", "maintenance")
    }
    if not any(beats.values()):
        return Check("daemon", True, "no heartbeat yet (daemon may not be running)")
    now = datetime.now(UTC)
    stale = []
    for name, ts in beats.items():
        if ts is None:
            stale.append(f"{name}:missing")
            continue
        try:
            age = (now - parse_iso(ts)).total_seconds()
        except ValueError:
            continue
        if age > 7200:  # > 2h since last heartbeat → likely dead loop
            stale.append(f"{name}:{int(age)}s")
    return Check("daemon", not stale, "ok" if not stale else "stale " + ", ".join(stale))


def run_checks(conn: sqlite3.Connection, settings: Settings | None = None) -> list[Check]:
    settings = settings or get_settings()
    return [
        _migration(conn),
        _disk(settings),
        _counts(conn),
        _llm(settings),
        _encryption(settings),
        _recording(conn, settings),
        _daemon(conn),
    ]


def summary(conn: sqlite3.Connection, settings: Settings | None = None) -> dict:
    checks = run_checks(conn, settings)
    return {
        "status": "ok" if all(c.ok for c in checks) else "degraded",
        "version": SCHEMA_VERSION,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
    }
