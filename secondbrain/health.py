"""Health checks: `sb doctor` preflight + the /health endpoint.

Every check is best-effort and degradable — a failing dependency yields an
``ok=False`` Check, never an exception — so this is safe to call anywhere and
testable with mocked backends.

Checks carry a ``severity``: 'error' means the system is broken or data is at
risk; 'warn' is advisory (stale backups, unreachable LLM, growing backlog).
``sb doctor`` exits non-zero only for errors.
"""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.storage import retention, state
from secondbrain.storage.schema import SCHEMA_VERSION

# Heartbeat staleness tiers (daemon loops write one heartbeat per iteration).
HEARTBEAT_WARN_S = 15 * 60
HEARTBEAT_ERROR_S = 2 * 3600

# Queue backlog thresholds (advisory).
BACKLOG_WARN_PENDING = 100
BACKLOG_WARN_OLDEST_S = 2 * 3600

# Dead/muted-mic detection: this many consecutive chunks below the RMS floor.
DEAD_MIC_CHUNKS = 5
DEAD_MIC_RMS = 1e-5

# "No backups yet" stops being fine once the corpus is this old.
BACKUP_NAG_DAYS = 7


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "error"  # 'error' | 'warn'
    # Actionable next step, rendered by `sb doctor` as "→ fix: …" (empty = none).
    hint: str = ""


def _migration(conn: sqlite3.Connection) -> Check:
    hint = "run `sb repair` (or `alembic upgrade head`) to bring the schema up to date"
    try:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        ver = row["version_num"] if row else None
        return Check("migrations", ver == SCHEMA_VERSION, f"{ver} (head {SCHEMA_VERSION})",
                     hint=hint)
    except sqlite3.Error as exc:
        return Check("migrations", False, str(exc), hint=hint)


def _disk(settings: Settings) -> Check:
    hint = "free disk space, or run `sb sweep` to delete expired raw audio"
    try:
        free = round(retention.free_disk_gb(settings.data_path), 2)
        return Check("disk", retention.disk_ok(settings), f"{free} GB free", hint=hint)
    except OSError as exc:
        return Check("disk", False, str(exc), hint=hint)


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
    model = settings.llm.model
    try:
        import httpx

        r = httpx.get(f"{settings.llm.host}/api/tags", timeout=5.0)
        if r.status_code != 200:
            return Check("llm", False, f"ollama {r.status_code}", severity="warn",
                         hint="is Ollama healthy? try `ollama serve` / check its logs")
        # Verify the configured model is actually pulled, not just that the
        # server answers — a missing model fails only at first real use.
        try:
            names = {m.get("name", "") for m in (r.json().get("models") or [])}
        except ValueError:
            names = None  # unparsable body: don't guess about pulled models
        if names is not None and model not in names and f"{model}:latest" not in names:
            return Check("llm", False, f"model {model} not pulled", severity="warn",
                         hint=f"run `ollama pull {model}`")
        return Check("llm", True, f"ollama 200, model {model} available", severity="warn")
    except Exception as exc:  # noqa: BLE001 - reachability is best-effort
        return Check("llm", False, f"ollama unreachable: {exc}", severity="warn",
                     hint=f"start it with `ollama serve` (expected at {settings.llm.host})")


def _encryption(settings: Settings) -> Check:
    if not settings.security.encrypt_db:
        return Check("encryption", True, "disabled (FileVault recommended)")
    from secondbrain.storage import db

    has_driver = db.sqlcipher_available()
    has_pass = bool(settings.security.db_passphrase)
    if not (has_driver and has_pass):
        return Check("encryption", False,
                     f"sqlcipher={'ok' if has_driver else 'missing'}, "
                     f"passphrase={'set' if has_pass else 'missing'}",
                     hint="install with `pip install -e '.[secure]'` and set "
                          "[security].db_passphrase in config.local.toml")
    # A real keyed open: a wrong passphrase must fail doctor, not the daemon's
    # next runtime connection.
    try:
        c = db.connect(settings=settings)
        try:
            c.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            c.close()
    except Exception as exc:  # noqa: BLE001 - driver-specific DatabaseError types
        return Check("encryption", False, f"encrypted DB failed to open: {exc}",
                     hint="check [security].db_passphrase — it doesn't unlock this database")
    return Check("encryption", True, "sqlcipher ok, passphrase unlocks the database")


def _backups(conn: sqlite3.Connection, settings: Settings) -> Check:
    """Encourage backups. Advisory: warn when snapshots have gone stale (>30d),
    or when there are none at all yet the corpus is old enough to be worth one."""
    from secondbrain.storage import backup
    from secondbrain.storage.models import iso_from_dt

    snaps = backup.list_backups(settings)
    if not snaps:
        try:
            cutoff = iso_from_dt(datetime.now(UTC) - timedelta(days=BACKUP_NAG_DAYS))
            has_old_content = conn.execute(
                "SELECT 1 FROM transcripts WHERE created_at <= ? LIMIT 1", (cutoff,)
            ).fetchone() is not None
        except sqlite3.Error:
            has_old_content = False
        if has_old_content:
            return Check(
                "backups", False,
                f"none yet, but transcripts are >{BACKUP_NAG_DAYS}d old — run `sb backup`",
                severity="warn", hint="run `sb backup` (the daemon also snapshots daily)",
            )
        return Check("backups", True, "none yet — run `sb backup`", severity="warn")
    newest = snaps[0]
    try:
        age_days = (datetime.now(UTC) - datetime.fromisoformat(newest["modified"])).days
    except ValueError:
        return Check("backups", True, f"{len(snaps)} snapshot(s)", severity="warn")
    ok = age_days <= 30
    return Check("backups", ok, f"{len(snaps)} snapshot(s), newest {age_days}d ago",
                 severity="warn", hint="" if ok else "run `sb backup` to take a fresh snapshot")


def _secrets() -> Check:
    """Warn if a secret was placed in the version-controlled config.toml."""
    from secondbrain.config import committed_secrets

    leaked = committed_secrets()
    if leaked:
        return Check(
            "secrets", False,
            "in committed config.toml: " + ", ".join(leaked)
            + " (move to config.local.toml or env)",
        )
    return Check("secrets", True, "no secrets in committed config.toml")


def _recording(conn: sqlite3.Connection, settings: Settings) -> Check:
    paused = state.is_paused(conn, default=settings.consent.paused)
    on = settings.consent.recording_enabled and not paused
    return Check("recording", True, "on" if on else "paused/off")


def _microphone(settings: Settings) -> Check:
    """Best-effort capture readiness: is there a usable input device?

    Surfaces the silent-failure case where the daemon's capture loop crash-loops
    because no mic is available or macOS Microphone (TCC) permission was denied.
    Degradable: never raises; on dev/CI without the audio extra it passes.
    """
    try:
        from secondbrain.capture.devices import (
            DeviceNotFoundError,
            list_input_devices,
            resolve_device,
        )
    except ImportError:
        return Check("microphone", True, "audio extra not installed")
    try:
        devices = list_input_devices()
    except Exception as exc:  # noqa: BLE001 - PortAudio/CoreAudio best-effort
        return Check("microphone", True, f"could not enumerate devices: {exc}")
    if not devices:
        return Check(
            "microphone", False,
            "no input devices — check System Settings → Privacy & Security → Microphone",
            hint="grant Microphone permission (or plug in a mic), then restart the daemon",
        )
    name = settings.capture.input_device
    if name:
        try:
            resolve_device(name)
        except DeviceNotFoundError:
            return Check("microphone", False, f"device {name!r} not found (run `sb devices`)",
                         hint="run `sb devices` and set [capture].input_device to a listed name")
    detail = f"{len(devices)} input device(s)" + (f", using {name!r}" if name else ", default")
    return Check("microphone", True, detail)


def _input_device_alarm(conn: sqlite3.Connection) -> Check:
    """Surface the recorder's refusing-to-record alarm (configured mic missing)."""
    try:
        alarm = state.get_state(conn, "alarm:input_device")
    except sqlite3.Error as exc:
        return Check("input_device", False, str(exc))
    if alarm:
        return Check("input_device", False, alarm,
                     hint="run `sb devices` and fix [capture].input_device, or plug the mic in")
    return Check("input_device", True, "no alarm")


def _daemon(conn: sqlite3.Connection) -> Check:
    """Report whether the daemon's loops are alive (via their heartbeats).

    Tiered: a heartbeat older than 15 minutes warns; older than 2 hours is an
    error (the loop is almost certainly dead)."""
    from secondbrain.storage.models import parse_iso

    beats = {
        name: state.get_state(conn, f"heartbeat:{name}")
        for name in ("capture", "worker", "maintenance")
    }
    if not any(beats.values()):
        return Check("daemon", True, "no heartbeat yet (daemon may not be running)")
    now = datetime.now(UTC)
    warn, error = [], []
    for name, ts in beats.items():
        if ts is None:
            warn.append(f"{name}:missing")
            continue
        try:
            age = (now - parse_iso(ts)).total_seconds()
        except ValueError:
            continue
        if age > HEARTBEAT_ERROR_S:
            error.append(f"{name}:{int(age)}s")
        elif age > HEARTBEAT_WARN_S:
            warn.append(f"{name}:{int(age)}s")
    if error:
        return Check("daemon", False, "stale " + ", ".join(error + warn))
    if warn:
        return Check("daemon", False, "stale " + ", ".join(warn), severity="warn")
    return Check("daemon", True, "ok")


def _capture_fresh(conn: sqlite3.Connection, settings: Settings) -> Check:
    """Is capture actually producing audio (not just switched on)?

    The recorder registers a chunk roughly every ``capture.chunk_seconds`` while
    unpaused — even in a silent room — so a long gap while recording is on means
    capture is broken. A grace window after the latest pause→resume flip avoids
    flagging the ordinary just-resumed minute; an empty corpus is never flagged.
    """
    from secondbrain.storage.models import parse_iso

    paused = state.is_paused(conn, default=settings.consent.paused)
    recording = settings.consent.recording_enabled and not paused
    if not recording:
        return Check("capture", True, "recording paused/off")
    if state.get_state(conn, "heartbeat:capture") is None:
        # Never any capture heartbeat → the daemon isn't (yet) running here;
        # staleness would be noise. _daemon covers a dead capture loop instead.
        return Check("capture", True, "no capture heartbeat yet (daemon may not be running)")
    row = conn.execute("SELECT MAX(started_at) AS ts FROM audio_files").fetchone()
    if row is None or row["ts"] is None:
        return Check("capture", True, "no chunks yet")
    now = datetime.now(UTC)
    try:
        gap = (now - parse_iso(row["ts"])).total_seconds()
    except ValueError:
        return Check("capture", True, f"unparseable last chunk time {row['ts']!r}")
    threshold = max(3 * settings.capture.chunk_seconds, 120)
    flipped = state.pause_changed_at(conn)
    since_flip = gap
    if flipped:
        with contextlib.suppress(ValueError):
            since_flip = (now - parse_iso(flipped)).total_seconds()
    if gap > threshold and since_flip > threshold:
        return Check("capture", False, f"no audio captured for {int(gap)}s while recording is on")
    return Check("capture", True, f"last chunk {int(gap)}s ago")


def _mic_signal(conn: sqlite3.Connection) -> Check:
    """Dead/muted-mic detection: N consecutive chunks with near-zero RMS."""
    try:
        rows = conn.execute(
            "SELECT rms_level FROM audio_files WHERE rms_level IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (DEAD_MIC_CHUNKS,),
        ).fetchall()
    except sqlite3.Error as exc:
        return Check("mic_signal", False, str(exc), severity="warn")
    if len(rows) < DEAD_MIC_CHUNKS:
        return Check("mic_signal", True, "not enough chunks yet")
    if all(r["rms_level"] < DEAD_MIC_RMS for r in rows):
        return Check(
            "mic_signal", False,
            f"last {DEAD_MIC_CHUNKS} chunks near-silent (rms < {DEAD_MIC_RMS}) — "
            "mic may be dead or muted",
            severity="warn",
            hint="check the mic's mute switch/input level, or pick another with `sb devices`",
        )
    return Check("mic_signal", True, "signal present")


def _failed_jobs(conn: sqlite3.Connection) -> Check:
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE state='failed'"
        ).fetchone()["n"]
    except sqlite3.Error as exc:
        return Check("failed_jobs", False, str(exc), severity="warn")
    if n:
        return Check("failed_jobs", False,
                     f"{n} dead-lettered job(s) — run `sb queue --retry-failed`",
                     severity="warn", hint="run `sb queue --retry-failed`")
    return Check("failed_jobs", True, "none")


def _queue_backlog(conn: sqlite3.Connection) -> Check:
    """Backlog depth + oldest-pending age (advisory: the queue drains async)."""
    from secondbrain.storage.models import parse_iso

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n, MIN(scheduled_at) AS oldest FROM jobs "
            "WHERE state='pending'"
        ).fetchone()
    except sqlite3.Error as exc:
        return Check("queue", False, str(exc), severity="warn")
    n = row["n"]
    oldest_s = 0
    if row["oldest"]:
        try:
            oldest_s = int((datetime.now(UTC) - parse_iso(row["oldest"])).total_seconds())
        except ValueError:
            oldest_s = 0
    detail = f"{n} pending, oldest {oldest_s}s"
    ok = n <= BACKLOG_WARN_PENDING and oldest_s <= BACKLOG_WARN_OLDEST_S
    return Check("queue", ok, detail, severity="warn")


def _local_config_perms() -> Check:
    """config.local.toml holds secrets (hf_token, db_passphrase): warn when it's
    readable by group/others."""
    from secondbrain.config import REPO_ROOT

    path = REPO_ROOT / "config.local.toml"
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        return Check("config_perms", True, "no config.local.toml", severity="warn")
    except OSError as exc:
        return Check("config_perms", True, str(exc), severity="warn")
    if mode & 0o077:
        return Check(
            "config_perms", False,
            f"config.local.toml is group/world-readable (mode {oct(mode)})",
            severity="warn",
            hint=f"run `chmod 600 {path}` — it can hold secrets",
        )
    return Check("config_perms", True, f"config.local.toml mode {oct(mode)}", severity="warn")


def _launchd_plists() -> Check:
    """Warn when installed launchd plists point at a different interpreter or
    repo than the one running now (venv moved / repo relocated → agents run old
    code). Off-macOS or with nothing installed the check passes."""
    try:
        from secondbrain import deploy

        problems = deploy.stale_plists()
    except Exception as exc:  # noqa: BLE001 - best-effort inspection
        return Check("launchd_plists", True, f"could not inspect plists: {exc}",
                     severity="warn")
    if problems:
        return Check("launchd_plists", False, "; ".join(problems), severity="warn",
                     hint="re-run `sb deploy launchd --load` to point the agents here")
    return Check("launchd_plists", True, "installed plists match this environment",
                 severity="warn")


def run_checks(conn: sqlite3.Connection, settings: Settings | None = None) -> list[Check]:
    settings = settings or get_settings()
    return [
        _migration(conn),
        _disk(settings),
        _counts(conn),
        _llm(settings),
        _encryption(settings),
        _secrets(),
        _local_config_perms(),
        _launchd_plists(),
        _backups(conn, settings),
        _microphone(settings),
        _input_device_alarm(conn),
        _recording(conn, settings),
        _capture_fresh(conn, settings),
        _mic_signal(conn),
        _failed_jobs(conn),
        _queue_backlog(conn),
        _daemon(conn),
    ]


def summary(conn: sqlite3.Connection, settings: Settings | None = None) -> dict:
    checks = run_checks(conn, settings)
    return {
        "status": "ok" if all(c.ok for c in checks) else "degraded",
        "version": SCHEMA_VERSION,
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail, "severity": c.severity,
             "hint": c.hint}
            for c in checks
        ],
    }
