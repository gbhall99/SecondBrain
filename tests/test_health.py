from fastapi.testclient import TestClient

from secondbrain import health
from secondbrain.query.api import create_app


def test_summary_ok_with_mock_backends(conn, settings):
    s = health.summary(conn, settings)
    assert s["version"] == "0010_planner"
    names = {c["name"] for c in s["checks"]}
    assert {"migrations", "disk", "database", "llm", "encryption", "recording"} <= names
    # llm backend is mock + encryption off → those checks pass
    by = {c["name"]: c for c in s["checks"]}
    assert by["llm"]["ok"] and by["encryption"]["ok"] and by["migrations"]["ok"]


def test_health_endpoint_no_auth(conn, settings):
    settings.security.require_auth = True  # health must remain open
    from secondbrain.security import auth

    auth.set_password(conn, "owner", "opensesame")
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


# --- batch 3: model-pulled check, hints, perms, encryption, --json ------------


class _TagsResp:
    def __init__(self, status_code=200, models=None):
        self.status_code = status_code
        self._models = models

    def json(self):
        return {"models": [{"name": n} for n in (self._models or [])]}


def test_llm_check_flags_unpulled_model(settings, monkeypatch):
    import httpx

    settings.llm.backend = "ollama"
    monkeypatch.setattr(
        httpx, "get", lambda url, timeout: _TagsResp(models=["some-other:7b"])
    )
    c = health._llm(settings)
    assert not c.ok and c.severity == "warn"
    assert "not pulled" in c.detail
    assert f"ollama pull {settings.llm.model}" in c.hint


def test_llm_check_accepts_pulled_model_and_latest_tag(settings, monkeypatch):
    import httpx

    settings.llm.backend = "ollama"
    monkeypatch.setattr(
        httpx, "get", lambda url, timeout: _TagsResp(models=[settings.llm.model])
    )
    assert health._llm(settings).ok
    settings.llm.model = "llama3"
    monkeypatch.setattr(
        httpx, "get", lambda url, timeout: _TagsResp(models=["llama3:latest"])
    )
    assert health._llm(settings).ok


def test_llm_check_uses_5s_timeout(settings, monkeypatch):
    import httpx

    seen = {}

    def fake_get(url, timeout):
        seen["timeout"] = timeout
        return _TagsResp(models=[settings.llm.model])

    settings.llm.backend = "ollama"
    monkeypatch.setattr(httpx, "get", fake_get)
    health._llm(settings)
    assert seen["timeout"] == 5.0


def test_checks_carry_hints_and_summary_exposes_them(conn, settings):
    s = health.summary(conn, settings)
    for c in s["checks"]:
        assert "hint" in c
    # a failing actionable check renders a hint
    from secondbrain.pipeline import queue as q

    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    q.fail(conn, q.claim_next(conn), "boom")
    by = {c.name: c for c in health.run_checks(conn, settings)}
    assert by["failed_jobs"].hint == "run `sb queue --retry-failed`"


def test_doctor_renders_fix_hints(conn, settings, monkeypatch):
    from typer.testing import CliRunner

    from secondbrain import cli
    from secondbrain.pipeline import queue as q

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    q.fail(conn, q.claim_next(conn), "boom")
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "→ fix: run `sb queue --retry-failed`" in result.output


def test_doctor_json_outputs_summary(conn, settings, monkeypatch):
    import json

    from typer.testing import CliRunner

    from secondbrain import cli

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    result = CliRunner().invoke(cli.app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] in ("ok", "degraded")
    names = {c["name"] for c in data["checks"]}
    assert {"migrations", "llm", "config_perms", "launchd_plists"} <= names


def test_local_config_perms_check(monkeypatch, tmp_path):
    import secondbrain.config as config_mod

    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    # no file → fine
    assert health._local_config_perms().ok
    p = tmp_path / "config.local.toml"
    p.write_text('[diarization]\nhf_token = "secret"\n')
    p.chmod(0o644)
    c = health._local_config_perms()
    assert not c.ok and c.severity == "warn" and "world-readable" in c.detail
    assert "chmod 600" in c.hint
    p.chmod(0o600)
    assert health._local_config_perms().ok


def test_encryption_check_fails_on_wrong_passphrase(settings, monkeypatch):
    from secondbrain.storage import db as db_mod

    settings.security.encrypt_db = True
    settings.security.db_passphrase = "wrong-passphrase"
    monkeypatch.setattr(db_mod, "sqlcipher_available", lambda: True)

    def bad_connect(*a, **k):
        raise RuntimeError("SQLCipher key setup failed (check db_passphrase)")

    monkeypatch.setattr(db_mod, "connect", bad_connect)
    c = health._encryption(settings)
    assert not c.ok
    assert "failed to open" in c.detail
    assert "db_passphrase" in c.hint


def test_encryption_check_passes_when_keyed_open_works(settings, monkeypatch):
    from secondbrain.storage import db as db_mod

    settings.security.encrypt_db = True
    settings.security.db_passphrase = "right-passphrase"
    monkeypatch.setattr(db_mod, "sqlcipher_available", lambda: True)

    class _Conn:
        def execute(self, sql):
            class _Cur:
                def fetchone(self):
                    return (0,)

            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr(db_mod, "connect", lambda *a, **k: _Conn())
    c = health._encryption(settings)
    assert c.ok and "unlocks" in c.detail


def test_stale_plist_check_warns_on_mismatch(monkeypatch, tmp_path):
    from secondbrain import deploy

    monkeypatch.setattr(deploy, "stale_plists", lambda: ["com.secondbrain.daemon: runs /old"])
    c = health._launchd_plists()
    assert not c.ok and c.severity == "warn"
    assert "sb deploy launchd" in c.hint
    monkeypatch.setattr(deploy, "stale_plists", list)
    assert health._launchd_plists().ok
