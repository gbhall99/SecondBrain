from __future__ import annotations

import logging
import math
import sys
import threading
import types
from pathlib import Path

import pytest

from secondbrain.capture import devices, recorder
from secondbrain.pipeline import queue as q
from secondbrain.storage import models, state


def test_register_chunk_inserts_and_enqueues(conn, settings):
    af_id = recorder.register_chunk(
        conn,
        path=settings.audio_raw_dir / "20260616-090000.flac",
        started_at="2026-06-16T09:00:00.000Z",
        ended_at="2026-06-16T09:01:00.000Z",
        duration_s=60.0,
        settings=settings,
    )
    assert models.get_audio_file(conn, af_id)["status"] == "recorded"
    # a transcription job should be queued for it
    job = q.claim_next(conn, "transcribe")
    assert job is not None and job.payload["audio_file_id"] == af_id


def test_should_record_respects_consent_and_pause(conn, settings):
    ok, _ = recorder.should_record(settings, conn)
    assert ok is True

    state.set_paused(conn, True)
    ok, reason = recorder.should_record(settings, conn)
    assert ok is False and "paused" in reason
    state.set_paused(conn, False)

    settings.consent.recording_enabled = False
    ok, reason = recorder.should_record(settings, conn)
    assert ok is False and "disabled" in reason


def test_should_record_disk_guardrail(conn, settings):
    settings.consent.recording_enabled = True
    settings.capture.min_free_disk_gb = 10**9  # impossibly high -> guardrail trips
    ok, reason = recorder.should_record(settings, conn)
    assert ok is False and "disk" in reason


def test_register_chunk_stores_rms_and_overflow(conn, settings):
    af_id = recorder.register_chunk(
        conn,
        path=settings.audio_raw_dir / "20260616-091000.flac",
        started_at="2026-06-16T09:10:00.000Z",
        ended_at="2026-06-16T09:11:00.000Z",
        duration_s=60.0,
        settings=settings,
        rms_level=0.042,
        overflow_count=3,
    )
    row = models.get_audio_file(conn, af_id)
    assert row["rms_level"] == pytest.approx(0.042)
    assert row["overflow_count"] == 3


def test_transcribe_jobs_enqueued_with_priority(conn, settings):
    recorder.register_chunk(
        conn,
        path=settings.audio_raw_dir / "20260616-092000.flac",
        started_at="2026-06-16T09:20:00.000Z",
        ended_at="2026-06-16T09:21:00.000Z",
        duration_s=60.0,
        settings=settings,
    )
    row = conn.execute("SELECT priority FROM jobs WHERE type='transcribe'").fetchone()
    assert row["priority"] > 0  # outranks heavy (priority 0) jobs


# --- device resolution -------------------------------------------------------


def test_resolve_device_blank_means_default(monkeypatch):
    assert devices.resolve_device("") is None  # intentional system default


def test_resolve_device_unknown_name_raises(monkeypatch):
    monkeypatch.setattr(devices, "list_input_devices", lambda: [
        devices.InputDevice(index=0, name="Built-in", channels=1, default=True)
    ])
    with pytest.raises(devices.DeviceNotFoundError):
        devices.resolve_device("Ghost Mic")
    assert devices.resolve_device("built-in") == 0  # substring match still works


# --- blocked-reason rate limiting + heartbeat --------------------------------


def test_blocked_reason_logged_at_most_once_per_minute(conn, settings, caplog, monkeypatch):
    rec = recorder.Recorder(conn, settings)
    clock = iter([0.0, 10.0, 30.0, 61.0])
    monkeypatch.setattr(recorder.time, "monotonic", lambda: next(clock))
    with caplog.at_level(logging.WARNING, logger=recorder.log.name):
        rec._log_blocked("recording paused")   # t=0 → logged
        rec._log_blocked("recording paused")   # t=10 → suppressed
        rec._log_blocked("low disk space")     # t=30 → different reason → logged
        rec._log_blocked("recording paused")   # t=61 → window elapsed → logged
    msgs = [r.message for r in caplog.records]
    assert msgs.count("capture blocked: recording paused") == 2
    assert msgs.count("capture blocked: low disk space") == 1


def test_heartbeat_written_to_app_state(conn, settings):
    rec = recorder.Recorder(conn, settings)
    assert state.get_state(conn, recorder.HEARTBEAT_CAPTURE) is None
    rec._heartbeat()
    assert state.get_state(conn, recorder.HEARTBEAT_CAPTURE) is not None


# --- the live capture loop (fake sounddevice/soundfile/numpy) ----------------


class _StopOnWait(threading.Event):
    """Event whose wait() sets itself — the loop exits at its first sleep."""

    def wait(self, timeout=None):  # noqa: ARG002
        self.set()
        return True


def _fake_numpy():
    np = types.ModuleType("numpy")

    def empty(shape, dtype=None):  # noqa: ARG001
        rows, ch = shape
        return [[0.0] * ch for _ in range(rows)]

    np.empty = empty
    np.square = lambda x: [[v * v for v in row] for row in x]
    np.mean = lambda x: (
        sum(v for row in x for v in row) / max(1, sum(len(row) for row in x))
    )
    np.sqrt = math.sqrt
    return np


def _fake_soundfile(written: list):
    sf = types.ModuleType("soundfile")

    def write(path, data, samplerate, format=None):  # noqa: A002, ARG001
        Path(path).write_bytes(b"FLAC")
        written.append((path, len(data)))

    sf.write = write
    return sf


def _fake_sounddevice(read_fn):
    sd = types.ModuleType("sounddevice")

    class InputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n):
            return read_fn(n)

    sd.InputStream = InputStream
    return sd


def _install_fake_audio(monkeypatch, read_fn, written):
    monkeypatch.setitem(sys.modules, "numpy", _fake_numpy())
    monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile(written))
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice(read_fn))


def test_run_refuses_to_record_on_missing_device_and_alarms(conn, settings, monkeypatch, caplog):
    settings.capture.input_device = "Ghost Mic"
    monkeypatch.setattr(devices, "list_input_devices", list)

    def read_fn(n):  # noqa: ARG001
        raise AssertionError("stream must never be opened for a missing device")

    _install_fake_audio(monkeypatch, read_fn, [])
    rec = recorder.Recorder(conn, settings)
    rec._stop = _StopOnWait()
    with caplog.at_level(logging.ERROR, logger=recorder.log.name):
        rec.run()
    alarm = state.get_state(conn, recorder.ALARM_INPUT_DEVICE)
    assert alarm and "Ghost Mic" in alarm
    assert any("refusing to record" in r.message for r in caplog.records)
    # nothing was captured or enqueued
    assert conn.execute("SELECT COUNT(*) AS n FROM audio_files").fetchone()["n"] == 0


def test_run_writes_chunk_with_rms_overflow_and_clears_alarm(conn, settings, monkeypatch):
    # 1-second chunks at 4 Hz → 4 frames per chunk, one read fills a chunk
    settings.capture.chunk_seconds = 1
    settings.capture.sample_rate = 4
    state.set_state(conn, recorder.ALARM_INPUT_DEVICE, "stale alarm")
    written: list = []
    rec = recorder.Recorder(conn, settings)

    def read_fn(n):
        rec._stop.set()  # one chunk, then exit
        return [[0.5] for _ in range(n)], True  # constant signal + overflow flag

    _install_fake_audio(monkeypatch, read_fn, written)
    rec.run()

    assert written, "chunk file was written"
    row = conn.execute("SELECT * FROM audio_files").fetchone()
    assert row["status"] == "recorded"
    assert abs(row["rms_level"] - 0.5) < 1e-9  # no pytest.approx: fake numpy is installed
    assert row["overflow_count"] == 1
    assert state.get_state(conn, recorder.ALARM_INPUT_DEVICE) == ""  # cleared
    assert state.get_state(conn, recorder.HEARTBEAT_CAPTURE) is not None
    job = q.claim_next(conn, "transcribe")
    assert job is not None and job.payload["audio_file_id"] == row["id"]


def test_pause_mid_chunk_discards_partial_buffer(conn, settings, monkeypatch):
    # 10-second chunks at 4 Hz → 40 frames; reads deliver 4 frames (~1s) each
    settings.capture.chunk_seconds = 10
    settings.capture.sample_rate = 4
    written: list = []
    reads = {"n": 0}

    def read_fn(n):
        reads["n"] += 1
        if reads["n"] == 2:
            state.set_paused(conn, True)  # pause lands mid-chunk
        return [[0.5] for _ in range(n)], False

    _install_fake_audio(monkeypatch, read_fn, written)
    rec = recorder.Recorder(conn, settings)
    rec._stop = _StopOnWait()  # the blocked wait after the pause ends the loop
    rec.run()

    assert reads["n"] <= 3          # pause honored within ~one read, not a whole chunk
    assert written == []            # the partial buffer was discarded…
    assert conn.execute("SELECT COUNT(*) AS n FROM audio_files").fetchone()["n"] == 0
