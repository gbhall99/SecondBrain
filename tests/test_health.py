from fastapi.testclient import TestClient

from secondbrain import health
from secondbrain.query.api import create_app


def test_summary_ok_with_mock_backends(conn, settings):
    s = health.summary(conn, settings)
    assert s["version"] == "0007_perf_indexes"
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
