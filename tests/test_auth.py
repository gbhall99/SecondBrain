import base64
import hashlib
import hmac
import time

import pytest
from fastapi.testclient import TestClient

from secondbrain.query.api import create_app
from secondbrain.security import auth

LOOPBACK = ("127.0.0.1", 50000)  # TestClient default host is non-loopback ("testclient")


@pytest.fixture(autouse=True)
def _isolated_rate_limiter():
    """Login-failure state is a module-global keyed by client IP, and every test
    here shares TestClient's 'testclient' host — isolate tests from each other."""
    auth._login_failures.clear()
    yield
    auth._login_failures.clear()


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


def test_cookie_generation_revocation():
    secret = b"s3cr3t"
    cookie = auth.make_cookie("owner", secret, max_age_days=1, generation=3)
    assert auth.verify_cookie(cookie, secret, generation=3) == "owner"
    assert auth.verify_cookie(cookie, secret, generation=4) is None  # revoked by bump
    # Legacy (pre-generation, username:exp) cookies are rejected outright —
    # holders just sign in once more.
    payload = f"owner:{int(time.time()) + 3600}"
    sig = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
    legacy = base64.urlsafe_b64encode(payload.encode()).decode() + "." + sig
    assert auth.verify_cookie(legacy, secret) is None


def test_session_generation_bump_persists(conn):
    assert auth.session_generation(conn) == 0
    assert auth.bump_session_generation(conn) == 1
    assert auth.session_generation(conn) == 1  # stored in app_state, not memory


def test_loopback_and_exempt():
    assert auth.is_loopback("127.0.0.1") and auth.is_loopback("::1")
    assert not auth.is_loopback("100.64.0.1")
    assert auth.is_exempt("/health") and auth.is_exempt("/static/x.css")
    assert not auth.is_exempt("/api/status")
    # Boundary: a bare-prefix collision must NOT be exempt (auth-bypass foot-gun).
    assert not auth.is_exempt("/loginfo")
    assert not auth.is_exempt("/healthz")
    assert auth.is_exempt("/login")  # exact still exempt
    # Browsers fetch the favicon before any login; it carries nothing personal.
    assert auth.is_exempt("/favicon.ico")
    assert not auth.is_exempt("/favicon.icons")


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


def test_health_html_page_only_for_authenticated_browsers(conn, settings):
    # The health *page* carries the same verbose detail as the authed JSON
    # (device names, job errors), so an unauthenticated remote browser must
    # keep getting the redacted probe JSON — never the page.
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    html = {"accept": "text/html"}
    r = client.get("/health", headers=html)
    assert r.headers["content-type"].startswith("application/json")
    assert set(r.json()) == {"status", "version"}
    # Once signed in, the same request renders the in-shell page.
    client.post("/login", json={"username": "owner", "password": "pw"})
    r = client.get("/health", headers=html)
    assert r.headers["content-type"].startswith("text/html")
    assert "System health" in r.text


def test_security_headers_on_401(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    r = client.get("/api/status")  # 401, an early-return path
    assert r.status_code == 401
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("Referrer-Policy") == "no-referrer"
    csp = r.headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "form-action 'self'" in csp


def test_html_no_store_only_when_auth_enabled(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    client.post("/login", json={"username": "owner", "password": "pw"})
    # Personal HTML pages must not outlive the session in a cache…
    assert client.get("/").headers.get("Cache-Control") == "no-store"
    # …but JSON API responses keep their exact header set (CLI/menu bar contract).
    assert client.get("/api/status").headers.get("Cache-Control") is None


def test_login_next_roundtrip(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    # Unauthenticated page request → login redirect remembering the target.
    r = client.get("/timeline?q=x", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=%2Ftimeline%3Fq%3Dx"
    # The home page gets a clean /login with no next clutter.
    r = client.get("/", follow_redirects=False)
    assert r.headers["location"] == "/login"
    # API paths still get JSON 401s, never redirects.
    assert client.get("/api/status", follow_redirects=False).status_code == 401
    # Once signed in, /login?next=… forwards to the original target.
    client.post("/login", json={"username": "owner", "password": "pw"})
    r = client.get("/login", params={"next": "/timeline?q=x"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/timeline?q=x"
    # Absolute, scheme-relative, and backslash targets are rejected → home.
    for evil in ("https://evil.example/", "//evil.example", "/\\evil.example", "timeline"):
        r = client.get("/login", params={"next": evil}, follow_redirects=False)
        assert r.headers["location"] == "/"


def test_login_page_redirects_when_auth_disabled(conn, settings):
    # With auth off the form can never succeed (authenticate() is always False),
    # so /login must not be a dead end.
    client = TestClient(create_app(settings))
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_login_page_states(conn, settings):
    settings.security.require_auth = True
    client = TestClient(create_app(settings))
    # No password configured yet: a setup notice instead of a doomed form.
    r = client.get("/login")
    assert r.status_code == 200
    assert "sb auth set-password" in r.text
    assert 'name="password"' not in r.text
    # Password set: real form with native semantics + password-manager hooks.
    auth.set_password(conn, "owner", "pw")
    r = client.get("/login")
    assert 'name="username"' in r.text and 'name="password"' in r.text
    assert 'autocomplete="username"' in r.text
    assert 'autocomplete="current-password"' in r.text
    assert "required" in r.text
    assert 'action="/login"' in r.text and 'method="post"' in r.text
    assert 'autofocus' in r.text
    # Signed-out confirmation notice (SB.signout lands on /login?signedout=1).
    r = client.get("/login", params={"signedout": "1"})
    assert "signed out" in r.text
    # Already signed in → no dead-end form, straight home.
    client.post("/login", json={"username": "owner", "password": "pw"})
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_form_login_fallback_no_js(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    # Wrong password: re-rendered form, inline error, username preserved.
    r = client.post("/login", data={"username": "owner", "password": "x", "next": "/timeline"})
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("text/html")
    assert "Wrong username or password." in r.text
    assert 'value="owner"' in r.text
    # Missing fields: friendly 422, still HTML.
    r = client.post("/login", data={"username": "owner", "password": ""})
    assert r.status_code == 422 and "Enter both" in r.text
    # Right password: 303 to the validated next target, session works.
    r = client.post(
        "/login",
        data={"username": "owner", "password": "pw", "next": "/timeline"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == "/timeline"
    assert client.get("/api/status").status_code == 200
    # Hostile next targets fall back to home.
    client.post("/logout")
    r = client.post(
        "/login",
        data={"username": "owner", "password": "pw", "next": "https://evil.example/"},
        follow_redirects=False,
    )
    assert r.headers["location"] == "/"


def test_json_login_contract_unchanged(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    r = client.post("/login", json={"username": "owner", "password": "pw"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert auth.COOKIE_NAME in r.headers.get("set-cookie", "")
    assert client.post("/logout").json() == {"ok": True}
    # Malformed bodies get clean errors, not tracebacks.
    assert client.post("/login", json={"username": "owner"}).status_code == 422
    r = client.post(
        "/login", content=b"not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 400


def test_login_rate_limit_message_and_form_variant(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    for _ in range(5):
        assert client.post("/login", json={"username": "owner", "password": "x"}).status_code == 401
    r = client.post("/login", json={"username": "owner", "password": "pw"})
    assert r.status_code == 429
    assert "try again" in r.json()["detail"]
    # The no-JS form variant shows the throttle inline instead of raw JSON.
    r = client.post("/login", data={"username": "owner", "password": "pw"})
    assert r.status_code == 429 and "Too many attempts" in r.text


def test_signout_control_only_for_cookie_sessions(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    app = create_app(settings)
    # Loopback bypasses auth entirely → no Sign out control to click.
    local = TestClient(app, client=LOOPBACK)
    r = local.get("/")
    assert r.status_code == 200
    assert 'class="nav-signout"' not in r.text
    # A remote cookie session gets the control in the shared nav.
    remote = TestClient(app)
    remote.post("/login", json={"username": "owner", "password": "pw"})
    r = remote.get("/")
    assert r.status_code == 200
    assert 'class="nav-signout"' in r.text and ">Sign out</button>" in r.text


def test_signout_control_absent_when_auth_disabled(conn, settings):
    client = TestClient(create_app(settings))  # non-loopback host, auth off
    r = client.get("/")
    assert r.status_code == 200
    assert 'class="nav-signout"' not in r.text
    # Local-only mode also keeps HTML cacheable exactly as before.
    assert r.headers.get("Cache-Control") is None


def test_logout_revokes_stolen_cookie_server_side(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    client.post("/login", json={"username": "owner", "password": "pw"})
    stolen = client.cookies.get(auth.COOKIE_NAME)
    assert stolen
    assert client.get("/api/status").status_code == 200
    client.post("/logout")
    # Replaying the pre-logout cookie fails: logout bumped the generation.
    client.cookies.set(auth.COOKIE_NAME, stolen)
    assert client.get("/api/status").status_code == 401
    # A fresh sign-in works under the new generation.
    client.cookies.delete(auth.COOKIE_NAME)  # drop the replayed copy from the jar
    assert client.post("/login", json={"username": "owner", "password": "pw"}).status_code == 200
    assert client.get("/api/status").status_code == 200


def test_logout_revocation_survives_restart(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    client.post("/login", json={"username": "owner", "password": "pw"})
    stolen = client.cookies.get(auth.COOKIE_NAME)
    client.post("/logout")
    # A brand-new app process reads the bumped generation from app_state.
    fresh = TestClient(create_app(settings))
    fresh.cookies.set(auth.COOKIE_NAME, stolen)
    assert fresh.get("/api/status").status_code == 401


def test_unauthenticated_logout_cannot_revoke_sessions(conn, settings):
    # /logout is auth-exempt; a drive-by POST without a valid session must not
    # bump the generation and kill the owner's real sessions.
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    app = create_app(settings)
    owner = TestClient(app)
    owner.post("/login", json={"username": "owner", "password": "pw"})
    assert owner.get("/api/status").status_code == 200
    drive_by = TestClient(app)
    assert drive_by.post("/logout").status_code == 200  # harmless no-op
    assert owner.get("/api/status").status_code == 200  # owner unaffected


def test_favicon_served_without_auth(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    r = client.get("/favicon.ico", follow_redirects=False)
    assert r.status_code == 200  # exempt asset, no /login?next=/favicon.ico junk
