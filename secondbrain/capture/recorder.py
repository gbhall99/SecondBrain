"""Always-on rolling recorder: room audio -> FLAC chunks -> queue.

The platform-agnostic bits (registering a chunk in the DB, enqueueing it, the
consent/disk pre-checks) are separated from the sounddevice loop so they're unit
-testable on Linux/CI. The live capture loop itself runs on the Mac mini.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline.worker import enqueue_transcription
from secondbrain.storage import retention, state
from secondbrain.storage.models import AudioFile, insert_audio_file, iso_from_dt, utcnow_iso

log = logging.getLogger(__name__)

# app_state keys the recorder maintains for health checks.
ALARM_INPUT_DEVICE = "alarm:input_device"
HEARTBEAT_CAPTURE = "heartbeat:capture"

# Log each distinct blocked-capture reason at most this often.
BLOCKED_LOG_INTERVAL_S = 60.0


def chunk_filename(started_at: datetime) -> str:
    return started_at.strftime("%Y%m%d-%H%M%S") + ".flac"


def register_chunk(
    conn: sqlite3.Connection,
    path: Path,
    started_at: str,
    ended_at: str,
    duration_s: float,
    settings: Settings,
    *,
    rms_level: float | None = None,
    overflow_count: int | None = None,
) -> int:
    """Record a finished chunk in the DB and enqueue it for transcription."""
    af = AudioFile(
        path=str(path),
        started_at=started_at,
        ended_at=ended_at,
        sample_rate=settings.capture.sample_rate,
        channels=settings.capture.channels,
        duration_s=duration_s,
        status="recorded",
        rms_level=rms_level,
        overflow_count=overflow_count,
    )
    audio_id = insert_audio_file(conn, af)
    enqueue_transcription(conn, audio_id)
    return audio_id


def should_record(settings: Settings, conn: sqlite3.Connection | None = None) -> tuple[bool, str]:
    """Consent + disk pre-checks. Returns (ok, reason-if-not).

    The live pause toggle (DB ``app_state``) overrides the static config default
    so the menu bar / API can pause capture without restarting the daemon.
    """
    if not settings.consent.recording_enabled:
        return False, "recording disabled in consent settings"
    paused = settings.consent.paused
    if conn is not None:
        paused = state.is_paused(conn, default=settings.consent.paused)
    if paused:
        return False, "recording paused"
    if not retention.disk_ok(settings):
        return False, "low disk space (guardrail)"
    return True, ""


class Recorder:
    """Continuous capture into fixed-length FLAC chunks via sounddevice."""

    def __init__(self, conn: sqlite3.Connection, settings: Settings | None = None):
        self.conn = conn
        self.settings = settings or get_settings()
        self._stop = threading.Event()
        self._last_blocked_log: dict[str, float] = {}

    def stop(self) -> None:
        self._stop.set()

    def _log_blocked(self, reason: str) -> None:
        """WARN about a blocked capture reason, at most once/minute per reason."""
        now = time.monotonic()
        last = self._last_blocked_log.get(reason)
        if last is None or now - last >= BLOCKED_LOG_INTERVAL_S:
            log.warning("capture blocked: %s", reason)
            self._last_blocked_log[reason] = now

    def _heartbeat(self) -> None:
        state.set_state(self.conn, HEARTBEAT_CAPTURE, utcnow_iso())

    def run(self) -> None:
        """Blocking capture loop. Call from a dedicated thread."""
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        from secondbrain.capture.devices import DeviceNotFoundError, resolve_device

        cfg = self.settings.capture
        self.settings.ensure_dirs()
        frames_per_chunk = cfg.sample_rate * cfg.chunk_seconds

        # Outer retry: a transient device error (mic unplugged, CoreAudio glitch)
        # must not kill capture permanently — back off and reopen the stream.
        backoff = 1.0
        max_backoff = 60.0
        while not self._stop.is_set():
            try:
                # Refuse to record on the WRONG microphone: a configured device
                # that can't be found raises, raises an alarm for `sb doctor` /
                # /health, and retries — it never falls back to the default mic.
                try:
                    device = resolve_device(cfg.input_device)
                except DeviceNotFoundError as exc:
                    log.error("refusing to record: %s", exc)
                    state.set_state(self.conn, ALARM_INPUT_DEVICE, str(exc))
                    self._stop.wait(backoff)
                    backoff = min(max_backoff, backoff * 2.0)
                    continue
                state.set_state(self.conn, ALARM_INPUT_DEVICE, "")
                with sd.InputStream(
                    samplerate=cfg.sample_rate,
                    channels=cfg.channels,
                    device=device,
                    dtype="float32",
                ) as stream:
                    backoff = 1.0  # reset after a clean open
                    while not self._stop.is_set():
                        self._heartbeat()
                        ok, reason = should_record(self.settings, self.conn)
                        if not ok:
                            self._log_blocked(reason)
                            self._stop.wait(1.0)
                            continue

                        started = datetime.now(UTC)
                        buf = np.empty((frames_per_chunk, cfg.channels), dtype="float32")
                        filled = 0
                        overflows = 0
                        discard = False
                        while filled < frames_per_chunk and not self._stop.is_set():
                            # Re-check pause/consent mid-chunk (reads are ≤1s) so a
                            # pause takes effect within ~1s, not a whole chunk later.
                            ok, reason = should_record(self.settings, self.conn)
                            if not ok:
                                self._log_blocked(reason)
                                discard = True
                                break
                            block, overflowed = stream.read(
                                min(cfg.sample_rate, frames_per_chunk - filled)
                            )
                            if overflowed:
                                overflows += 1
                            n = len(block)
                            buf[filled : filled + n] = block
                            filled += n
                        if discard or filled == 0:
                            continue  # partial buffer from a pause is never persisted

                        if overflows:
                            log.warning(
                                "capture: %d input overflow(s) in chunk (audio dropped)",
                                overflows,
                            )
                        rms = float(np.sqrt(np.mean(np.square(buf[:filled]))))
                        path = self.settings.audio_raw_dir / chunk_filename(started)
                        sf.write(str(path), buf[:filled], cfg.sample_rate, format="FLAC")
                        duration = filled / cfg.sample_rate
                        register_chunk(
                            self.conn,
                            path,
                            iso_from_dt(started),
                            utcnow_iso(),
                            duration,
                            self.settings,
                            rms_level=rms,
                            overflow_count=overflows,
                        )
                        self._heartbeat()
            except Exception:  # noqa: BLE001 - transient audio/device error: back off + reopen
                if self._stop.is_set():
                    break
                log.warning("capture stream error; reopening in %.0fs", backoff, exc_info=True)
                self._stop.wait(backoff)
                backoff = min(max_backoff, backoff * 2.0)
