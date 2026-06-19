from fastapi.testclient import TestClient

from secondbrain import health
from secondbrain.query.api import create_app


def test_summary_ok_with_mock_backends(conn, settings):
    s = health.summary(conn, settings)
    assert s["version"] == "0005_tasks"
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
