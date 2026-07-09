"""Username/password auth with a stdlib-signed session cookie.

No third-party deps: passwords are PBKDF2-hashed, the session cookie is an
HMAC-signed ``username:generation:expiry`` token. Credentials, the signing
secret, and the session generation live in ``app_state`` (which is inside the
encrypted DB when SQLCipher is enabled). The generation is a monotonic counter
embedded in every cookie: logout bumps it, instantly revoking all outstanding
cookies server-side instead of trusting browsers to forget them.
Loopback clients are exempt so local use and the CLI/menu bar never need a login.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time

from secondbrain.storage import state

_CRED_KEY = "auth_credentials"
_SECRET_KEY = "auth_session_secret"
_GEN_KEY = "auth_session_generation"
_PBKDF2_ROUNDS = 200_000
COOKIE_NAME = "sb_session"
# /favicon.ico is exempt like /static: browsers fetch it unauthenticated and it
# only serves the emoji icon (nothing personal).
EXEMPT_PREFIXES = ("/health", "/login", "/logout", "/static", "/favicon.ico")


# --- password hashing --------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(stored: str, password: str) -> bool:
    try:
        _algo, rounds, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


# --- credential + secret storage (app_state) ---------------------------------


def set_password(conn: sqlite3.Connection, username: str, password: str) -> None:
    creds = json.dumps({"username": username, "hash": hash_password(password)})
    state.set_state(conn, _CRED_KEY, creds)


def get_credentials(conn: sqlite3.Connection) -> dict | None:
    raw = state.get_state(conn, _CRED_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def has_password(conn: sqlite3.Connection) -> bool:
    return get_credentials(conn) is not None


def authenticate(conn: sqlite3.Connection, username: str, password: str) -> bool:
    creds = get_credentials(conn)
    if not creds:
        return False
    return creds.get("username") == username and verify_password(creds.get("hash", ""), password)


def session_secret(conn: sqlite3.Connection) -> bytes:
    raw = state.get_state(conn, _SECRET_KEY)
    if not raw:
        raw = secrets.token_hex(32)
        state.set_state(conn, _SECRET_KEY, raw)
    return raw.encode()


def session_generation(conn: sqlite3.Connection) -> int:
    """Current session generation; cookies minted under an older one are dead."""
    raw = state.get_state(conn, _GEN_KEY)
    try:
        return int(raw) if raw else 0
    except (TypeError, ValueError):
        return 0


def bump_session_generation(conn: sqlite3.Connection) -> int:
    """Revoke every outstanding session cookie (sign out everywhere)."""
    gen = session_generation(conn) + 1
    state.set_state(conn, _GEN_KEY, str(gen))
    return gen


# --- signed session cookie ---------------------------------------------------


def make_cookie(username: str, secret: bytes, max_age_days: int, generation: int = 0) -> str:
    exp = int(time.time()) + max_age_days * 86400
    payload = f"{username}:{generation}:{exp}"
    sig = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{token}.{sig}"


def verify_cookie(cookie: str, secret: bytes, generation: int = 0) -> str | None:
    """Return the username when the cookie is authentic, unexpired, and from the
    current session generation; None otherwise (including pre-generation legacy
    cookies, which simply require one fresh sign-in)."""
    try:
        token, sig = cookie.split(".")
        payload = base64.urlsafe_b64decode(token.encode()).decode()
        expected = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        username, gen, exp = payload.rsplit(":", 2)
        if int(exp) < int(time.time()) or int(gen) != generation:
            return None
    except (ValueError, TypeError):
        return None
    return username


# --- helpers -----------------------------------------------------------------


def is_loopback(host: str | None) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


def is_exempt(path: str) -> bool:
    # Exact match or a proper sub-path only — the bare prefix must NOT match, or a
    # future route like /loginfo would silently bypass auth.
    return any(path == p or path.startswith(p + "/") for p in EXEMPT_PREFIXES)


def env_password() -> str | None:
    """Optional bootstrap password from env (e.g. for headless first-run)."""
    return os.environ.get("SB_AUTH_PASSWORD")


# --- login rate limiting (in-memory, single-process) -------------------------

_MAX_FAILURES = 5
_WINDOW_SECONDS = 300
_login_failures: dict[str, list[float]] = {}


def login_allowed(ip: str) -> bool:
    now = time.time()
    fails = [t for t in _login_failures.get(ip, []) if now - t < _WINDOW_SECONDS]
    if fails:
        _login_failures[ip] = fails
    else:
        _login_failures.pop(ip, None)  # don't retain empty entries (unbounded growth)
    return len(fails) < _MAX_FAILURES


def record_login_failure(ip: str) -> None:
    _login_failures.setdefault(ip, []).append(time.time())


def reset_login_failures(ip: str) -> None:
    _login_failures.pop(ip, None)
