from fastapi.testclient import TestClient

from secondbrain import health
from secondbrain.query.api import create_app


def test_summary_ok_with_mock_backends(conn, settings):
    s = health.summary(conn, settings)
    assert s["version"] == "0008_reliability"
    names = {c["name"] for c in s["checks"]}
    assert {"migrations", "disk", "database", "llm", "encryption", "recording"} <= names
    # llm backend is mock + encryption off → those checks pass
    by = {c["name"]: c for c in s["checks"]}
    assert by["llm"]["ok"] and by["encryption"]["ok"] and by["migrations"]["ok"]


def test_health_endpoint_no_auth(conn, settings):
    settings.security.require_auth = True  # health must remain open
    from secondbrain.security import auth

    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] in ("ok", "degraded")


def test_doctor_checks_run(conn, settings):
    checks = health.run_checks(conn, settings)
    assert all(hasattr(c, "ok") for c in checks)
    assert "microphone" in {c.name for c in checks}


def test_microphone_check_import_error_degrades_ok(settings, monkeypatch):
    # No audio extra installed (CI): the check must pass, not raise.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "secondbrain.capture.devices":
            raise ImportError("no sounddevice")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    c = health._microphone(settings)
    assert c.ok and "audio extra" in c.detail


def test_microphone_check_no_devices_fails(settings, monkeypatch):
    monkeypatch.setattr(
        "secondbrain.capture.devices.list_input_devices", list, raising=False
    )
    c = health._microphone(settings)
    assert not c.ok and "Microphone" in c.detail


def test_microphone_check_configured_device_missing_fails(settings, monkeypatch):
    from secondbrain.capture.devices import InputDevice

    monkeypatch.setattr(
        "secondbrain.capture.devices.list_input_devices",
        lambda: [InputDevice(index=0, name="Built-in", channels=1, default=True)],
        raising=False,
    )
    settings.capture.input_device = "Nonexistent Mic"
    c = health._microphone(settings)
    assert not c.ok and "not found" in c.detail


# --- severity ----------------------------------------------------------------


def test_summary_includes_severity_and_stays_backward_compatible(conn, settings):
    s = health.summary(conn, settings)
    for c in s["checks"]:
        assert {"name", "ok", "detail", "severity"} <= set(c)
        assert c["severity"] in ("error", "warn")
    # advisory checks are 'warn'
    by = {c["name"]: c for c in s["checks"]}
    assert by["backups"]["severity"] == "warn"


def test_llm_unreachable_is_warn_not_error(settings):
    settings.llm.backend = "ollama"
    settings.llm.host = "http://127.0.0.1:9"  # nothing listens here
    c = health._llm(settings)
    assert not c.ok and c.severity == "warn"


def test_summary_degraded_on_warn_only_failure(conn, settings):
    from secondbrain.pipeline import queue as q

    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    job = q.claim_next(conn)
    q.fail(conn, job, "boom")  # dead-lettered → failed_jobs warns
    s = health.summary(conn, settings)
    assert s["status"] == "degraded"  # JSON contract unchanged: any failure degrades


# --- new checks --------------------------------------------------------------


def _by_name(conn, settings):
    return {c.name: c for c in health.run_checks(conn, settings)}


def test_failed_jobs_check_warns_with_retry_hint(conn, settings):
    from secondbrain.pipeline import queue as q

    assert _by_name(conn, settings)["failed_jobs"].ok
    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    q.fail(conn, q.claim_next(conn), "boom")
    c = _by_name(conn, settings)["failed_jobs"]
    assert not c.ok and c.severity == "warn" and "--retry-failed" in c.detail


def test_queue_backlog_check_flags_old_pending(conn, settings):
    from secondbrain.pipeline import queue as q

    assert _by_name(conn, settings)["queue"].ok  # empty queue is fine
    q.enqueue(conn, "transcribe", {"audio_file_id": 1})
    conn.execute("UPDATE jobs SET scheduled_at='2000-01-01T00:00:00.000Z'")
    c = _by_name(conn, settings)["queue"]
    assert not c.ok and c.severity == "warn" and "oldest" in c.detail


def test_daemon_heartbeat_tiers(conn, settings):
    from datetime import UTC, datetime, timedelta

    from secondbrain.storage import state
    from secondbrain.storage.models import iso_from_dt, utcnow_iso

    # no heartbeats at all → ok (daemon may simply not be running)
    assert _by_name(conn, settings)["daemon"].ok

    now = datetime.now(UTC)
    state.set_state(conn, "heartbeat:worker", utcnow_iso())
    state.set_state(conn, "heartbeat:maintenance", utcnow_iso())
    state.set_state(conn, "heartbeat:capture", utcnow_iso())
    assert _by_name(conn, settings)["daemon"].ok

    # 20 minutes stale → warn
    state.set_state(conn, "heartbeat:worker", iso_from_dt(now - timedelta(minutes=20)))
    c = _by_name(conn, settings)["daemon"]
    assert not c.ok and c.severity == "warn" and "worker" in c.detail

    # 3 hours stale → error
    state.set_state(conn, "heartbeat:worker", iso_from_dt(now - timedelta(hours=3)))
    c = _by_name(conn, settings)["daemon"]
    assert not c.ok and c.severity == "error"


def test_capture_freshness_check(conn, settings):
    from datetime import UTC, datetime, timedelta

    from secondbrain.storage import models, state
    from secondbrain.storage.models import AudioFile, iso_from_dt, utcnow_iso

    old = iso_from_dt(datetime.now(UTC) - timedelta(hours=2))
    models.insert_audio_file(
        conn, AudioFile(path="/tmp/old.flac", started_at=old, sample_rate=16000)
    )
    # without a capture heartbeat the daemon isn't running → never flagged
    assert _by_name(conn, settings)["capture"].ok
    # a capture heartbeat exists but chunks stopped arriving → error
    state.set_state(conn, "heartbeat:capture", utcnow_iso())
    c = _by_name(conn, settings)["capture"]
    assert not c.ok and c.severity == "error" and "no audio captured" in c.detail
    # paused → not flagged (and the pause-flip grace also covers a fresh resume)
    state.set_paused(conn, True)
    assert _by_name(conn, settings)["capture"].ok


def test_mic_signal_dead_mic_detection(conn, settings):
    from secondbrain.storage import models
    from secondbrain.storage.models import AudioFile

    # not enough chunks yet → ok
    assert _by_name(conn, settings)["mic_signal"].ok
    for i in range(health.DEAD_MIC_CHUNKS):
        models.insert_audio_file(
            conn, AudioFile(path=f"/tmp/z{i}.flac", started_at="2026-06-16T09:00:00.000Z",
                            sample_rate=16000, rms_level=0.0))
    c = _by_name(conn, settings)["mic_signal"]
    assert not c.ok and "mic may be dead or muted" in c.detail
    # one healthy chunk breaks the run
    models.insert_audio_file(
        conn, AudioFile(path="/tmp/loud.flac", started_at="2026-06-16T09:10:00.000Z",
                        sample_rate=16000, rms_level=0.2))
    assert _by_name(conn, settings)["mic_signal"].ok


def test_input_device_alarm_check(conn, settings):
    from secondbrain.storage import state

    assert _by_name(conn, settings)["input_device"].ok
    state.set_state(conn, "alarm:input_device", "input device 'Ghost' not found")
    c = _by_name(conn, settings)["input_device"]
    assert not c.ok and c.severity == "error" and "Ghost" in c.detail
    state.set_state(conn, "alarm:input_device", "")  # recorder clears it on recovery
    assert _by_name(conn, settings)["input_device"].ok


def test_backups_none_yet_degrades_once_corpus_is_old(conn, settings):
    from secondbrain.storage import models
    from secondbrain.storage.models import AudioFile

    # fresh install: no backups, no old transcripts → just a hint
    c = _by_name(conn, settings)["backups"]
    assert c.ok and "none yet" in c.detail

    af = models.insert_audio_file(
        conn, AudioFile(path="/tmp/a.flac", started_at="2026-06-01T09:00:00.000Z",
                        sample_rate=16000, status="transcribed"))
    models.insert_transcript(conn, af, "mock", "mock", "en")
    conn.execute("UPDATE transcripts SET created_at='2026-06-01T09:00:00.000Z'")
    c = _by_name(conn, settings)["backups"]
    assert not c.ok and c.severity == "warn" and "run `sb backup`" in c.detail


def test_doctor_exit_zero_on_warn_only(conn, settings, monkeypatch):
    from typer.testing import CliRunner

    from secondbrain import cli
    from secondbrain.pipeline import queue as q

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    runner = CliRunner()

    # a dead-lettered job → warn-only failure → exit 0
    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    q.fail(conn, q.claim_next(conn), "boom")
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "warning(s)" in result.output

    # an error-severity failure (schema out of date) → exit 1
    conn.execute("UPDATE alembic_version SET version_num='bogus'")
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1, result.output
