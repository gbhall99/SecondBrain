"""Always-on rolling recorder: room audio -> FLAC chunks -> queue.

The platform-agnostic bits (registering a chunk in the DB, enqueueing it, the
consent/disk pre-checks) are separated from the sounddevice loop so they're unit
-testable on Linux/CI. The live capture loop itself runs on the Mac mini.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline.worker import enqueue_transcription
from secondbrain.storage import retention, state
from secondbrain.storage.models import AudioFile, insert_audio_file, iso_from_dt, utcnow_iso


def chunk_filename(started_at: datetime) -> str:
    return started_at.strftime("%Y%m%d-%H%M%S") + ".flac"


def register_chunk(
    conn: sqlite3.Connection,
    path: Path,
    started_at: str,
    ended_at: str,
    duration_s: float,
    settings: Settings,
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

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Blocking capture loop. Call from a dedicated thread."""
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        from secondbrain.capture.devices import resolve_device

        cfg = self.settings.capture
        self.settings.ensure_dirs()
        device = resolve_device(cfg.input_device)
        frames_per_chunk = cfg.sample_rate * cfg.chunk_seconds

        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            device=device,
            dtype="float32",
        ) as stream:
            while not self._stop.is_set():
                ok, _ = should_record(self.settings, self.conn)
                if not ok:
                    self._stop.wait(1.0)
                    continue

                started = datetime.now(UTC)
                buf = np.empty((frames_per_chunk, cfg.channels), dtype="float32")
                filled = 0
                while filled < frames_per_chunk and not self._stop.is_set():
                    block, _ = stream.read(min(cfg.sample_rate, frames_per_chunk - filled))
                    n = len(block)
                    buf[filled : filled + n] = block
                    filled += n
                if filled == 0:
                    continue

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
                )
