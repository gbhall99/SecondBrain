from fastapi.testclient import TestClient

from secondbrain.query.api import create_app
from secondbrain.security import auth


def test_password_hash_roundtrip():
    h = auth.hash_password("hunter2")
    assert auth.verify_password(h, "hunter2")
    assert not auth.verify_password(h, "wrong")
    assert not auth.verify_password("garbage", "hunter2")


def test_cookie_sign_verify_tamper_expiry():
    secret = b"s3cr3t"
    cookie = auth.make_cookie("owner", secret, max_age_days=1)
    assert auth.verify_cookie(cookie, secret) == "owner"
    assert auth.verify_cookie(cookie, b"other") is None          # wrong secret
    assert auth.verify_cookie(cookie + "x", secret) is None       # tampered
    expired = auth.make_cookie("owner", secret, max_age_days=-1)
    assert auth.verify_cookie(expired, secret) is None


def test_loopback_and_exempt():
    assert auth.is_loopback("127.0.0.1") and auth.is_loopback("::1")
    assert not auth.is_loopback("100.64.0.1")
    assert auth.is_exempt("/health") and auth.is_exempt("/static/x.css")
    assert not auth.is_exempt("/api/status")
    # Boundary: a bare-prefix collision must NOT be exempt (auth-bypass foot-gun).
    assert not auth.is_exempt("/loginfo")
    assert not auth.is_exempt("/healthz")
    assert auth.is_exempt("/login")  # exact still exempt


def test_rate_limiter_does_not_retain_empty_entries():
    auth._login_failures.clear()
    assert auth.login_allowed("9.9.9.9") is True
    # A caller with no failures must not leave a dangling key behind.
    assert "9.9.9.9" not in auth._login_failures


def test_set_password_and_authenticate(conn):
    auth.set_password(conn, "owner", "pw")
    assert auth.has_password(conn)
    assert auth.authenticate(conn, "owner", "pw")
    assert not auth.authenticate(conn, "owner", "nope")
    assert not auth.authenticate(conn, "intruder", "pw")


def test_api_requires_auth_for_remote(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))  # TestClient host is non-loopback

    # health is always open
    assert client.get("/health").status_code == 200
    # api blocked without a session
    assert client.get("/api/status").status_code == 401
    # bad login rejected
    assert client.post("/login", json={"username": "owner", "password": "x"}).status_code == 401
    # good login sets cookie (persisted by TestClient) → access granted
    assert client.post("/login", json={"username": "owner", "password": "pw"}).status_code == 200
    assert client.get("/api/status").status_code == 200
    # logout revokes
    client.post("/logout")
    assert client.get("/api/status").status_code == 401


def test_api_open_when_auth_disabled(conn, settings):
    # default require_auth=False → no login needed (existing behaviour)
    client = TestClient(create_app(settings))
    assert client.get("/api/status").status_code == 200


def test_health_redacts_detail_for_unauthenticated_remote(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))  # non-loopback, no cookie
    body = client.get("/health").json()
    assert set(body) == {"status", "version"}  # no 'checks' (secret/device leak)
    # After login, full detail is returned.
    client.post("/login", json={"username": "owner", "password": "pw"})
    assert "checks" in client.get("/health").json()


def test_security_headers_on_401(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    r = client.get("/api/status")  # 401, an early-return path
    assert r.status_code == 401
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
