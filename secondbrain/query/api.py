"""FastAPI app exposing search/browse + recording controls. Local-only.

Always bind to 127.0.0.1 (see config.api.host). The same backend serves the web
dashboard, the menu bar app, and the CLI.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlsplit

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from secondbrain import health
from secondbrain.config import Settings, get_settings
from secondbrain.query import service
from secondbrain.search import semantic
from secondbrain.security import auth
from secondbrain.storage import state
from secondbrain.storage.db import db_session, init_db, transaction

log = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# SQLite stores ids as signed 64-bit ints; a larger Python int raises
# OverflowError inside the driver before the query even runs (a 500). Integer
# path/query params that reach SQL are bounded by this instead, so a fuzzed or
# copy-mangled id fails validation cleanly (422) like any other bad input.
_SQLITE_MAX_INT = 2**63 - 1

# Friendly copy for the HTML error page (browser navigations only; API clients
# and non-HTML callers keep the standard JSON error bodies).
_ERROR_PAGE_COPY = {
    404: ("Page not found", "That page doesn't exist — the link may be stale or the item removed."),
    403: ("Not allowed", "You don't have access to that."),
    405: ("Not allowed", "That address doesn't accept this kind of request."),
    410: ("No longer available", "That content has expired and was cleaned up by retention."),
    422: ("That link doesn't look right", "Part of the address has an unexpected value."),
    429: ("Slow down", "Too many attempts — wait a moment and try again."),
    500: ("Something went wrong", "An unexpected error occurred. Details are in the server log."),
}

# Sent on every response (normal and early-return paths alike). Matches the
# offline architecture: pages are self-contained — inline scripts/styles, local
# /static assets, data: favicon — and never talk to another origin.
# frame-ancestors supersedes X-Frame-Options in modern browsers; both are sent.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "media-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


def _safe_next(raw: str | None) -> str | None:
    """Validate a post-login redirect target: same-origin paths only.

    Accepts site-relative paths like ``/day?d=2026-07-02``; rejects absolute
    URLs, scheme-relative ``//host``, and backslash tricks so a crafted login
    link can never bounce a fresh session to another origin.
    """
    if raw and raw.startswith("/") and not raw.startswith(("//", "/\\")):
        return raw
    return None


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _is_cross_origin_write(request: Request) -> bool:
    """True when a state-changing request arrives from another web origin.

    Loopback requests bypass auth entirely, so a drive-by page in the owner's
    browser could otherwise fire ``fetch('http://127.0.0.1:8765/api/resume',
    {method:'POST', mode:'no-cors'})`` and silently flip the microphone back
    on. Browsers always attach an Origin header to cross-origin writes (and
    ``Origin: null`` from sandboxed iframes), so: refuse writes whose Origin
    doesn't match the host we were addressed as. Requests with no Origin at
    all (CLI, menu bar, curl, tests) and same-origin browser requests are
    untouched, and anything carrying the ``X-SecondBrain`` header passes —
    only same-origin scripts can set a custom header without a CORS preflight
    this server never grants.
    """
    if request.method not in _UNSAFE_METHODS:
        return False
    if request.headers.get("x-secondbrain"):
        return False
    origin = request.headers.get("origin")
    if not origin:
        return False
    host = (request.headers.get("host") or "").lower()
    return not host or urlsplit(origin).netloc.lower() != host


def _parse_day(day: str | None) -> datetime | None:
    """Strict YYYY-MM-DD parse; None when missing, malformed, or out of range.

    The year clamp keeps local-timezone math (mktime / timedelta) from
    overflowing on absurd-but-parseable dates like 0001-01-01.
    """
    if not day:
        return None
    try:
        d = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return None
    return d if 1970 <= d.year <= 9000 else None


def _elapsed_s(started_iso: str | None) -> int | None:
    """Whole seconds since a stored UTC timestamp; None when unparseable."""
    if not started_iso:
        return None
    try:
        started = datetime.strptime(started_iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return max(0, int((datetime.now(UTC) - started).total_seconds()))


# Job-type → human label for the health page (raw types stay in the JSON API).
_JOB_TYPE_LABELS = {
    "transcribe": "Transcription",
    "diarize_conversation": "Speaker diarization",
    "cluster_speakers": "Speaker clustering",
    "reattribute_speakers": "Speaker re-attribution",
    "extract_knowledge": "Knowledge extraction",
    "generate_digest": "Brief generation",
}

_GOAL_STATUSES = ("active", "paused", "done", "dropped")
_TASK_STATUSES = ("backlog", "next", "scheduled", "in_progress", "blocked", "done", "dropped")
# Mirrors the planner's default when a task has no estimate (planner._DEFAULT_TASK_MINUTES).
_DEFAULT_TASK_MINUTES = 30


def _fmt_day(day: str | None) -> str | None:
    """'2026-07-15' → 'Jul 15, 2026' for display; raw string when unparseable."""
    d = _parse_day(day)
    if d is None:
        return day or None
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def _local_dt(ts: str | None) -> str | None:
    """Stored UTC ISO timestamp → local wall-clock '2 Jul 2026, 23:24'.

    Registered as the ``localdt`` Jinja filter. None stays None (templates
    chain ``or "—"``); unparseable strings pass through untouched so odd data
    is still visible rather than hidden.
    """
    if not ts:
        return None
    try:
        then = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    local = then.astimezone()
    return f"{local.day} {local.strftime('%b %Y, %H:%M')}"


def _local_day(ts: str | None) -> str | None:
    """Stored UTC ISO timestamp → local calendar day 'YYYY-MM-DD' (for /day links).

    Registered as the ``localday`` Jinja filter; same bucketing as the /day view.
    """
    if not ts:
        return None
    try:
        then = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return then.astimezone().strftime("%Y-%m-%d")


def _local_hms(ts: str | None) -> str:
    """Stored UTC ISO timestamp → local wall-clock 'HH:MM:SS' (24-hour).

    Registered as the ``localhms`` Jinja filter. The server knows the machine's
    timezone, so the /day transcript renders correct local times immediately;
    the client-side pass only refines them if a viewer's timezone differs.
    Unset/unparseable timestamps yield '' (the caller renders an em dash).
    """
    if not ts:
        return ""
    try:
        then = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return then.astimezone().strftime("%H:%M:%S")


def _rel_ago(ts: str | None) -> str | None:
    """Stored UTC timestamp → coarse relative label ('3h ago'), None if unset."""
    if not ts:
        return None
    try:
        then = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    s = (datetime.now(UTC) - then).total_seconds()
    if s < 0:
        return "just now"
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 172800:  # under 2 days: hours read better than '1d'
        return f"{int(s // 3600)}h ago"
    if s < 86400 * 14:
        return f"{int(s // 86400)}d ago"
    return f"on {_fmt_day(then.strftime('%Y-%m-%d'))}"


def _due_info(due_date: str | None, today: str) -> dict:
    """Relative due-date label for task rows ('due tomorrow', '3 days overdue').

    Display-only annotation; raw ``due_date`` stays on the row untouched.
    """
    d, t = _parse_day(due_date), _parse_day(today)
    if d is None or t is None:
        return {"due_label": None, "overdue": False, "due_today": False}
    days = (d - t).days
    if days == 0:
        label = "due today"
    elif days == 1:
        label = "due tomorrow"
    elif days == -1:
        label = "due yesterday"
    elif days < 0:
        label = f"{-days} days overdue"
    elif days <= 13:
        label = f"due in {days} days"
    else:
        label = f"due {_fmt_day(due_date)}"
    return {"due_label": label, "overdue": days < 0, "due_today": days == 0}


def _annotate_dossier_commitments(d: dict) -> dict:
    """Additive due-date annotations on a person dossier's commitments.

    Same display fields the Tasks page uses (``due_label``/``overdue``/
    ``due_today``) so 'send the deck — 3 days overdue' reads identically in
    both places. Raw ``due_date`` stays untouched.
    """
    today = service.local_today()
    commits = d.get("commitments") or {}
    for c in (commits.get("owed_by") or []) + (commits.get("owed_to") or []):
        c.update(_due_info(c.get("due_date"), today))
    return d


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="SecondBrain", version="0.1.0")
    templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
    # Display filters shared by all templates: stored UTC ISO strings →
    # readable local time ('2 Jul 2026, 23:24'), local /day-link days, plain
    # dates ('Jul 15, 2026'), and coarse relative labels ('3h ago').
    templates.env.filters["localdt"] = _local_dt
    templates.env.filters["localday"] = _local_day
    templates.env.filters["localhms"] = _local_hms
    templates.env.filters["prettyday"] = _fmt_day
    templates.env.filters["relago"] = _rel_ago
    static_dir = _WEB_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Session signing secret (created on first use; lives in app_state).
    init_db(settings=settings).close()
    with db_session(settings=settings) as _c:
        secret = auth.session_secret(_c)
        # In-process mirror of the persisted session generation. /logout bumps
        # it to revoke every outstanding cookie at once; the stored value keeps
        # those revocations effective across restarts (single-worker server, so
        # a plain dict is a safe cache).
        session_gen = {"value": auth.session_generation(_c)}

    # Embed any transcripts recorded before vector indexing existed (or while
    # the embedding backend was broken) so natural-language questions can
    # ground against the whole corpus. Background daemon thread; no-op when
    # semantic search is disabled or nothing is missing.
    semantic.start_background_backfill(settings)

    def db():
        return db_session(settings=settings)

    def _harden(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = _CSP
        if (
            settings.security.require_auth
            and response.headers.get("content-type", "").startswith("text/html")
            and "cache-control" not in response.headers
        ):
            # Once remote access (and therefore auth) is on, personal pages must
            # not linger in browser/proxy caches after sign-out.
            response.headers["Cache-Control"] = "no-store"
        return response

    def _cookie_user(request: Request) -> str | None:
        """Username from a valid, current-generation session cookie (else None)."""
        cookie = request.cookies.get(auth.COOKIE_NAME, "")
        return auth.verify_cookie(cookie, secret, generation=session_gen["value"])

    def _authed(request: Request) -> bool:
        """Whether the caller may see privileged detail (mirrors the auth gate)."""
        if not settings.security.require_auth:
            return True
        host = request.client.host if request.client else None
        if auth.is_loopback(host):
            return True
        return _cookie_user(request) is not None

    def _session_context(request: Request) -> dict:
        """Extra context for every template render: whether this visitor holds a
        cookie session. Drives the Sign out control in base.html — loopback
        clients bypass auth entirely and must never see it."""
        host = request.client.host if request.client else None
        via_cookie = bool(
            settings.security.require_auth
            and not auth.is_loopback(host)
            and _cookie_user(request) is not None
        )
        return {"authed_via_cookie": via_cookie}

    templates.context_processors.append(_session_context)

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        # CSRF guard: applies even (especially) to auth-exempt loopback callers
        # — see _is_cross_origin_write. Reads are unaffected; the browser's
        # same-origin policy already walls off their responses.
        if _is_cross_origin_write(request):
            return _harden(
                JSONResponse({"detail": "cross-origin request blocked"}, status_code=403)
            )
        if settings.security.require_auth:
            host = request.client.host if request.client else None
            path = request.url.path
            if (
                not auth.is_loopback(host)
                and not auth.is_exempt(path)
                and _cookie_user(request) is None
            ):
                if path.startswith("/api/"):
                    return _harden(
                        JSONResponse({"detail": "authentication required"}, status_code=401)
                    )
                # Browser navigation: bounce via the login page, remembering the
                # requested page so sign-in lands back on it.
                target = path + (f"?{request.url.query}" if request.url.query else "")
                suffix = f"?next={quote(target, safe='')}" if target != "/" else ""
                return _harden(RedirectResponse(f"/login{suffix}", status_code=303))
        return _harden(await call_next(request))

    # --- error pages ----------------------------------------------------------
    # Browser navigations to page routes get a friendly error.html (shared nav
    # intact); /api/* paths and non-HTML clients (CLI, menu bar, curl) keep the
    # exact JSON error bodies they always had.

    def _wants_html(request: Request) -> bool:
        return not request.url.path.startswith("/api/") and "text/html" in request.headers.get(
            "accept", ""
        )

    def _error_page(request: Request, status_code: int, detail: str | None):
        title, hint = _ERROR_PAGE_COPY.get(status_code, ("Something went wrong", None))
        return templates.TemplateResponse(
            request,
            "error.html",
            {"status_code": status_code, "title": title, "detail": detail, "hint": hint},
            status_code=status_code,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if not _wants_html(request):
            return await http_exception_handler(request, exc)  # default JSON, unchanged
        detail = exc.detail if isinstance(exc.detail, str) else None
        return _error_page(request, exc.status_code, detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        if not _wants_html(request):
            return await request_validation_exception_handler(request, exc)  # default JSON
        return _error_page(request, 422, None)

    @app.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception):
        # Uvicorn still logs the traceback; this only shapes the response body.
        if not _wants_html(request):
            return _harden(JSONResponse({"detail": "internal server error"}, status_code=500))
        return _harden(_error_page(request, 500, None))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        icon = _WEB_DIR / "static" / "favicon.svg"
        if not icon.exists():
            raise HTTPException(404, "Not Found")
        return FileResponse(str(icon), media_type="image/svg+xml")

    @app.get("/health")
    def health_endpoint(request: Request):
        authed = _authed(request)
        with db() as conn:
            full = health.summary(conn, settings)
            # Browsers get an in-shell page (the dashboard's failed-jobs pill
            # lands here); probes/CLI/menu bar keep the exact JSON contract.
            # Auth first: the page carries the same verbose detail as the
            # authed JSON (device names, job errors), never probe-safe output.
            if authed and _wants_html(request):
                over = service.queue_overview(conn)
                failures = over["recent_failures"]
                for f in failures:
                    f["when"] = _rel_ago(f["finished_at"])
                    f["type_label"] = _JOB_TYPE_LABELS.get(f["type"], f["type"])
                return templates.TemplateResponse(
                    request,
                    "health.html",
                    {
                        "health": full,
                        "problems": [c for c in full["checks"] if not c["ok"]],
                        "counts": over["counts"],
                        "failures": failures,
                    },
                )
        # /health is exempt from auth (for liveness probes), so redact the verbose
        # detail (secret names, device names, encryption posture) for unauth callers.
        if authed:
            return full
        return {"status": full["status"], "version": full["version"]}

    def _login_form_response(request, *, status=200, error=None, username="", nxt=None):
        """Render login.html (used by GET /login and the no-JS form fallback)."""
        with db() as conn:
            no_password = not auth.has_password(conn)
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "next": nxt or "",
                "error": error,
                "username": username,
                "no_password": no_password,
                "signed_out": request.query_params.get("signedout") == "1",
            },
            status_code=status,
        )

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        nxt = _safe_next(request.query_params.get("next"))
        if _authed(request):
            # Loopback (auth bypassed) or already signed in — the form would be
            # a dead end, so continue to wherever the visitor was headed.
            return RedirectResponse(nxt or "/", status_code=303)
        return _login_form_response(request, nxt=nxt)

    @app.post("/login")
    async def login(request: Request):
        """Sign in. JSON callers (the login page's JS, tests) get ``{ok:true}``
        plus the session cookie; a native no-JS <form> post (urlencoded) gets a
        303 to its ``next`` target, or the re-rendered form with the error."""
        content_type = (request.headers.get("content-type") or "").lower()
        is_form = "application/x-www-form-urlencoded" in content_type
        if is_form:
            # Parsed with the stdlib (not request.form()) so the no-JS fallback
            # needs no python-multipart dependency — the form is urlencoded.
            raw = (await request.body()).decode("utf-8", errors="replace")
            form = dict(parse_qsl(raw, keep_blank_values=True))
            username = (form.get("username") or "").strip()
            password = form.get("password") or ""
            nxt = _safe_next(form.get("next"))
        else:
            try:
                data = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                raise HTTPException(400, "malformed JSON body") from None
            if not isinstance(data, dict):
                raise HTTPException(422, "username and password are required")
            username = str(data.get("username") or "").strip()
            password = str(data.get("password") or "")
            nxt = None

        def fail(status: int, json_detail: str, form_message: str):
            if is_form:
                return _login_form_response(
                    request, status=status, error=form_message, username=username, nxt=nxt
                )
            raise HTTPException(status, json_detail)

        ip = request.client.host if request.client else "?"
        if not auth.login_allowed(ip):
            return fail(
                429,
                "too many attempts; try again shortly",
                "Too many attempts — wait a few minutes, then try again.",
            )
        if not username or not password:
            return fail(
                422,
                "username and password are required",
                "Enter both a username and a password.",
            )
        with db() as conn:
            ok = auth.authenticate(conn, username, password)
        if not ok:
            auth.record_login_failure(ip)
            return fail(401, "invalid credentials", "Wrong username or password.")
        auth.reset_login_failures(ip)
        resp = (
            RedirectResponse(nxt or "/", status_code=303) if is_form else JSONResponse({"ok": True})
        )
        resp.set_cookie(
            auth.COOKIE_NAME,
            auth.make_cookie(
                username,
                secret,
                settings.security.session_max_age_days,
                generation=session_gen["value"],
            ),
            max_age=settings.security.session_max_age_days * 86400,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",  # set Secure when served over TLS
        )
        return resp

    @app.post("/logout")
    def logout(request: Request):
        """Sign out: clear the cookie and — when the caller actually held a live
        session — bump the stored generation, revoking every outstanding cookie
        on every device (a stolen copy dies too, instead of staying valid for
        session_max_age_days)."""
        if settings.security.require_auth and _cookie_user(request) is not None:
            with db() as conn:
                session_gen["value"] = auth.bump_session_generation(conn)
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(auth.COOKIE_NAME)
        return resp

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        with db() as conn:
            st = service.status(conn, settings)
            stats = service.corpus_stats(conn)
            # Voices for the search speaker filter: only people who actually
            # have transcript lines (and never opted-out voices).
            speakers = [
                s for s in service.list_speakers(conn)
                if not s["opted_out"] and (s["segment_count"] or 0) > 0
            ]
        return templates.TemplateResponse(
            request, "index.html", {"status": st, "stats": stats, "speakers": speakers}
        )

    @app.get("/api/status")
    def api_status():
        with db() as conn:
            return service.status(conn, settings)

    @app.get("/api/stats")
    def api_stats():
        with db() as conn:
            return service.corpus_stats(conn)

    @app.get("/api/search")
    def api_search(
        q: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=200),
        mode: str = Query("auto", pattern="^(auto|fulltext|semantic)$"),
        since: str | None = Query(None),
        until: str | None = Query(None),
        speaker: str | None = Query(None),
    ):
        # Same date contract as /api/day: empty string means "no filter" (a
        # cleared form field), anything else must be a real day — silently
        # matching nothing against a typo'd date would read as "no results".
        since, until = since or None, until or None
        for name, value in (("since", since), ("until", until)):
            if value is not None and _parse_day(value) is None:
                raise HTTPException(422, f"{name} must be a date like 2026-07-02 (YYYY-MM-DD)")
        # Speaker follows the same contract: '' = no filter; anything else must
        # be a real voice — a stale id silently matching nothing would read as
        # "no results".
        speaker_id: int | None = None
        if speaker:
            # Upper bound too: ids beyond SQLite's 64-bit range would raise
            # OverflowError inside the driver (a 500) instead of a clean 422.
            if not speaker.isdigit() or not 1 <= int(speaker) <= _SQLITE_MAX_INT:
                raise HTTPException(422, "speaker must be a numeric speaker id")
            speaker_id = int(speaker)
        with db() as conn:
            if speaker_id is not None:
                if service.speaker_label_for(conn, speaker_id) is None:
                    raise HTTPException(
                        422, "that speaker doesn't exist — it may have been merged or removed"
                    )
                speaker_id = service.resolve(conn, speaker_id)  # merge-safe canonical id
            results = service.search(
                conn, q, limit, mode, settings, since=since, until=until, speaker=speaker_id
            )
            # Lets the UI explain an empty result honestly (e.g. "semantic
            # search isn't available on this machine" instead of "no matches").
            sem_ok = semantic.is_available(conn, settings)
        return {
            "query": q,
            "mode": mode,
            "results": results,
            "count": len(results),
            "limit": limit,
            "semantic_available": sem_ok,
            # Additive: the (merge-resolved) speaker filter that was applied.
            "speaker": speaker_id,
        }

    @app.get("/api/day/{day}")
    def api_day(day: str):
        d = _parse_day(day)
        if d is None:
            raise HTTPException(422, "day must be a date like 2026-07-02 (YYYY-MM-DD)")
        # strptime tolerates non-padded input ("2026-7-2") — echo the canonical
        # form so clients (and <input type=date>) always see YYYY-MM-DD.
        day = d.strftime("%Y-%m-%d")
        with db() as conn:
            return {"day": day, "segments": service.day_segments(conn, day)}

    @app.get("/api/day/{day}/count")
    def api_day_count(day: str):
        """Just the segment count for a day — the /day 'N new lines' poll hits
        this instead of re-fetching every segment, so it can poll more often and
        cheaply. Additive; the full /api/day payload is unchanged."""
        d = _parse_day(day)
        if d is None:
            raise HTTPException(422, "day must be a date like 2026-07-02 (YYYY-MM-DD)")
        day = d.strftime("%Y-%m-%d")
        with db() as conn:
            return {"day": day, "count": service.day_segment_count(conn, day)}

    @app.post("/api/pause")
    def api_pause():
        with db() as conn:
            state.set_paused(conn, True)
        return JSONResponse({"paused": True})

    @app.post("/api/resume")
    def api_resume():
        with db() as conn:
            state.set_paused(conn, False)
        return JSONResponse({"paused": False})

    # --- speakers (Phase 2) --------------------------------------------------

    @app.get("/speakers", response_class=HTMLResponse)
    def speakers_page(request: Request):
        with db() as conn:
            unknowns = service.unknown_speakers(conn)
            speakers = service.list_speakers(conn)
            ignored = service.ignored_speakers(conn)
            quality = service.speaker_quality(conn, settings)
            undo_merge = service.pending_merge_undo(conn)
        return templates.TemplateResponse(
            request,
            "speakers.html",
            {
                "unknowns": unknowns,
                # "known" = named/owner voices only; unknowns already have their
                # own queue above and must not render twice.
                "known": [s for s in speakers if s["kind"] != "unknown"],
                "speakers": speakers,  # every active voice (merge targets)
                "ignored": ignored,  # dismissed "not a person" voices (restorable)
                "owner": next((s for s in speakers if s["is_owner"]), None),
                "quality": quality,
                "undo_merge": undo_merge,  # last merge, while still undoable
                "low_pct": round(settings.diarization.low_confidence_threshold * 100),
            },
        )

    @app.get("/api/speakers")
    def api_speakers():
        with db() as conn:
            return {"speakers": service.list_speakers(conn)}

    @app.get("/api/speakers/unknown")
    def api_unknown():
        with db() as conn:
            return {"unknown": service.unknown_speakers(conn)}

    @app.get("/api/speakers/ignored")
    def api_ignored():
        with db() as conn:
            return {"ignored": service.ignored_speakers(conn)}

    # Speaker/observation ids are 64-bit SQLite ints: bound every id the client
    # can craft (path or body) so an out-of-range value is a clean 422 from
    # validation instead of an OverflowError 500 inside sqlite3.
    @app.get("/api/speakers/{speaker_id}/samples")
    def api_samples(speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        with db() as conn:
            if service.speaker_label_for(conn, speaker_id) is None:
                raise HTTPException(404, "speaker not found")
            if service.is_opted_out(conn, speaker_id, settings):
                raise HTTPException(403, "speaker opted out")
            samples = service.speaker_samples(conn, speaker_id, settings=settings)
        # Absolute filesystem paths stay server-side; clients address clips by id
        # (GET /api/speakers/{id}/clip/{observation_id}), never by path.
        return {"samples": [{k: v for k, v in s.items() if k != "path"} for s in samples]}

    @app.get("/api/speakers/{speaker_id}/clip/{observation_id}")
    def api_clip(
        speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        observation_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
    ):
        with db() as conn:
            if service.is_opted_out(conn, speaker_id, settings):
                raise HTTPException(403, "speaker opted out")
            samples = service.speaker_samples(conn, speaker_id, n=50, settings=settings)
        sample = next((s for s in samples if s["id"] == observation_id), None)
        if sample is None:
            raise HTTPException(404, "observation not found")
        path = _extract_clip(sample, settings)
        if path is None:
            raise HTTPException(410, "audio expired (deleted by retention)")
        # inline: Firefox refuses to play attachment-disposition media in <audio>.
        return FileResponse(
            str(path), media_type="audio/wav", filename=f"clip_{observation_id}.wav",
            content_disposition_type="inline",
        )

    @app.post("/api/speakers/{speaker_id}/name")
    def api_name(
        speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        name: str = Body(..., embed=True),
    ):
        name = name.strip()
        if not name:
            raise HTTPException(400, "name can't be empty")
        if len(name) > 120:
            raise HTTPException(400, "name too long (max 120 characters)")
        with db() as conn:
            if service.speaker_label_for(conn, speaker_id) is None:
                raise HTTPException(404, "speaker not found")
            redacted = service.name_speaker(conn, speaker_id, name, settings)
            sid = service.resolve(conn, speaker_id)
            dup = conn.execute(
                "SELECT id, name FROM speakers WHERE merged_into IS NULL AND id<>? "
                "AND name=? COLLATE NOCASE LIMIT 1",
                (sid, name),
            ).fetchone()
        # "name"/"duplicate_of" are additive; "ok"/"redacted_segments" keep their
        # original shape. duplicate_of warns that another (unmerged) voice already
        # carries this name — the UIs suggest a merge instead of a silent twin.
        return {
            "ok": True,
            "redacted_segments": redacted,
            "name": name,
            "duplicate_of": {"id": dup["id"], "name": dup["name"]} if dup else None,
        }

    @app.post("/api/speakers/merge")
    def api_merge(
        src: int = Body(..., ge=1, le=_SQLITE_MAX_INT),
        dst: int = Body(..., ge=1, le=_SQLITE_MAX_INT),
    ):
        with db() as conn:
            for label, sid in (("src", src), ("dst", dst)):
                if service.speaker_label_for(conn, sid) is None:
                    raise HTTPException(404, f"{label} speaker not found")
            if service.resolve(conn, src) == service.resolve(conn, dst):
                raise HTTPException(400, "those are already the same person")
            src_row = conn.execute(
                "SELECT is_owner, name FROM speakers WHERE id=?", (service.resolve(conn, src),)
            ).fetchone()
            if src_row and src_row["is_owner"]:
                raise HTTPException(
                    400, "can't merge you away — merge the other voice into Me instead"
                )
            dst_row = conn.execute(
                "SELECT name FROM speakers WHERE id=?", (service.resolve(conn, dst),)
            ).fetchone()
            # A named voice merged into an unnamed one keeps its name (see
            # registry.merge_speakers). Surface which name survived so the UI
            # toast can say so instead of the name silently vanishing.
            kept_name = (
                src_row["name"]
                if src_row is not None and dst_row is not None
                and (src_row["name"] or "").strip() and not (dst_row["name"] or "").strip()
                else None
            )
            n, undoable = service.merge_speakers_undoable(conn, src, dst, settings)
        # "undo_available" is additive: the last merge can be reversed for a few
        # minutes via POST /api/speakers/merge/undo (not when dst is opted out —
        # that merge redacts text, which an undo can't bring back).
        # "kept_name" is additive too: the name the merged voice ends up with
        # when an unnamed dst adopts src's name, else null.
        return {
            "ok": True,
            "relabeled_segments": n,
            "undo_available": undoable,
            "kept_name": kept_name,
        }

    @app.post("/api/speakers/merge/undo")
    def api_merge_undo():
        """One-shot undo of the most recent merge, within a short window."""
        with db() as conn:
            res = service.undo_merge(conn)
        if res["status"] == "none":
            raise HTTPException(404, "nothing to undo — no recent merge on record")
        if res["status"] == "expired":
            raise HTTPException(410, "the undo window for that merge has passed")
        if res["status"] == "stale":
            raise HTTPException(
                409, "those voices changed after the merge, so it can no longer be undone"
            )
        return {
            "ok": True,
            "restored_segments": res["restored_segments"],
            "src": res["src"],
            "dst": res["dst"],
        }

    @app.post("/api/speakers/{speaker_id}/owner")
    def api_set_owner(
        speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        name: str | None = Body(None, embed=True),
    ):
        # "name" is additive and optional: sending it names the voice and marks
        # it as the owner in one atomic step (the first-time "This is me" flow);
        # existing body-less callers keep the original mark-owner-only behavior.
        if name is not None:
            name = name.strip()
            if not name:
                raise HTTPException(400, "name can't be empty")
            if len(name) > 120:
                raise HTTPException(400, "name too long (max 120 characters)")
        redacted = 0
        with db() as conn:
            if service.speaker_label_for(conn, speaker_id) is None:
                raise HTTPException(404, "speaker not found")
            with transaction(conn):
                if name is not None:
                    redacted = service.name_speaker(conn, speaker_id, name, settings)
                service.set_owner(conn, speaker_id)
        return {"ok": True, "name": name, "redacted_segments": redacted}

    @app.post("/api/speakers/{speaker_id}/dismiss")
    def api_dismiss_speaker(speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Dismiss a junk unknown voice (TV, delivery person, background video).

        The voice keeps its transcript lines and its profile — future audio from
        the same source still matches it silently — but it leaves the "Who is
        this?" queue and every list/picker. Reversible via /restore.
        """
        with db() as conn:
            s = service.speaker_overview(conn, speaker_id)
            if s is None:
                raise HTTPException(404, "speaker not found")
            if s["is_owner"]:
                raise HTTPException(400, "that's your own voice — it can't be ignored")
            if s["kind"] == "known":
                raise HTTPException(
                    400,
                    f"“{s['name'] or s['display_label']}” is a named person — "
                    "merge or rename that voice instead of ignoring it",
                )
            already = s["kind"] == "ignored"
            if not already:
                service.set_speaker_ignored(conn, s["id"], True)
        return {
            "ok": True,
            "id": s["id"],
            "label": s["display_label"] or f"Speaker #{s['id']}",
            "segment_count": s["segment_count"],
            "already_ignored": already,
        }

    @app.post("/api/speakers/{speaker_id}/restore")
    def api_restore_speaker(speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Put a dismissed voice back in the unknown queue (undo of /dismiss)."""
        with db() as conn:
            s = service.speaker_overview(conn, speaker_id)
            if s is None:
                raise HTTPException(404, "speaker not found")
            if s["kind"] not in ("ignored", "unknown"):
                raise HTTPException(400, "that voice isn't ignored")
            if s["kind"] == "ignored":
                service.set_speaker_ignored(conn, s["id"], False)
        return {
            "ok": True,
            "id": s["id"],
            "label": s["display_label"] or f"Speaker #{s['id']}",
            "already_active": s["kind"] == "unknown",
        }

    # --- speaker quality / corrections (Phase 7) -----------------------------

    @app.get("/day", response_class=HTMLResponse)
    def day_page(request: Request, date: str = Query(None)):
        requested = (date or "").strip()
        today = service.local_today()
        invalid_date = None
        if not requested or requested == "today":
            day = today
        elif _parse_day(requested) is None:
            invalid_date, day = requested, today  # fall back gracefully, tell the user
        else:
            day = requested
        d = _parse_day(day)
        # Canonical YYYY-MM-DD: accepted-but-unpadded input ("2026-7-2") would
        # otherwise reach <input type=date value=…>, which browsers blank out.
        day = d.strftime("%Y-%m-%d")
        with db() as conn:
            segments = service.day_segments(conn, day, settings)
            # Reassign targets are identified people (named voices + the owner)
            # only. Anonymous "Unknown #N" clusters are diarizer groupings, not
            # people to teach a line to — moving a line onto one wouldn't teach a
            # voice profile and contradicts the "who really spoke" framing.
            # Naming an unknown voice still happens on the People page.
            speakers = [
                sp
                for sp in service.list_speakers(conn)
                if not sp["opted_out"] and sp["kind"] != "unknown"
            ]
            nav = service.day_nav(conn, day, settings)
            paused = state.is_paused(conn, default=settings.consent.paused)
            # Baseline for today's "N new lines" poll — same raw count the cheap
            # /api/day/{day}/count poll returns, so both compare like-for-like.
            raw_seg_count = service.day_segment_count(conn, day)
        low = settings.diarization.low_confidence_threshold
        # The owner's display name (usually "Me") lets the dispute affordance
        # phrase a wrongly-attributed owner line in the first person ("Not me…").
        owner = next((sp for sp in speakers if sp["is_owner"]), None)
        return templates.TemplateResponse(
            request,
            "day.html",
            {
                "date": day,  # kept for backward compatibility
                "day": day,
                "today": today,
                "pretty_day": f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}",
                "prev_day": (d - timedelta(days=1)).strftime("%Y-%m-%d"),
                "next_day": (d + timedelta(days=1)).strftime("%Y-%m-%d"),
                "nav": nav,
                "blocks": service.day_blocks(segments),
                "segments": segments,
                "speakers": speakers,
                "owner_name": (owner["name"] or owner["display_label"] or "Me") if owner else "Me",
                # Named people other than the owner: when zero, a wrongly-guessed
                # owner line has no one to reassign to, so the dispute affordance
                # routes to naming the real speaker (or unattributing) instead.
                "n_named_others": sum(
                    1 for sp in speakers if not sp["is_owner"] and sp["kind"] == "known"
                ),
                "invalid_date": invalid_date,
                "recording": settings.consent.recording_enabled and not paused,
                "low_pct": round(low * 100),
                "n_review": sum(
                    1
                    for s in segments
                    if not s.get("speaker_locked")
                    and (s.get("speaker_id") is None or s.get("speaker_low_confidence"))
                ),
                "n_locked": sum(1 for s in segments if s.get("speaker_locked")),
                "raw_seg_count": raw_seg_count,
            },
        )

    @app.post("/api/segments/{segment_id}/speaker")
    def api_reassign_segment(
        segment_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        speaker_id: int = Body(..., embed=True),
    ):
        with db() as conn:
            seg = service.get_segment(conn, segment_id)
            if seg is None:
                raise HTTPException(404, "segment not found")
            label = service.speaker_label_for(conn, speaker_id)
            if label is None:
                raise HTTPException(404, "speaker not found")
            # A correction teaches a voice profile — never for someone who
            # opted out of having their data kept (the pickers hide them,
            # but the API must enforce it too).
            if service.is_opted_out(conn, speaker_id, settings):
                raise HTTPException(403, "speaker opted out")
            ok = service.reassign_segment(conn, segment_id, speaker_id, settings)
            if not ok:
                raise HTTPException(404, "segment not found")
        # Extra fields are additive; "ok" keeps its original shape for the CLI.
        return {
            "ok": True,
            "segment_id": segment_id,
            "speaker": label,
            "locked": True,
            # Whether a confirmed voice exemplar could be fed back into the
            # profile (needs the segment's diarization embedding).
            "learned": bool(seg.get("has_embedding")),
        }

    @app.post("/api/segments/{segment_id}/unassign")
    def api_unassign_segment(segment_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Dispute a wrong attribution when no correct person is named yet.

        Clears the speaker and locks the line so re-attribution won't re-guess
        the rejected voice — the correction path can't otherwise reach a line
        whose only named target is the (wrongly-attributed) owner. Additive: the
        existing reassign endpoint is untouched.
        """
        with db() as conn:
            if service.get_segment(conn, segment_id) is None:
                raise HTTPException(404, "segment not found")
            ok = service.unassign_segment(conn, segment_id, settings)
            if not ok:
                raise HTTPException(404, "segment not found")
        return {"ok": True, "segment_id": segment_id, "speaker": None, "locked": True}

    @app.get("/api/segments/{segment_id}/clip")
    def api_segment_clip(segment_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Audio for one transcript line, so a correction can be made by ear.

        410 once the raw source audio has been swept by retention — transcripts
        outlive their audio by design.
        """
        with db() as conn:
            info = service.segment_clip_info(conn, segment_id)
            if info is None:
                raise HTTPException(404, "segment not found")
            # Opted-out voices are excluded from the day view, but the API must
            # refuse to serve their raw audio no matter how it's addressed.
            if info["speaker_id"] is not None and service.is_opted_out(
                conn, info["speaker_id"], settings
            ):
                raise HTTPException(403, "speaker opted out")
        path = _extract_clip(info, settings, prefix="segclip")
        if path is None:
            raise HTTPException(410, "audio expired (deleted by retention)")
        # inline: Firefox refuses to play attachment-disposition media in <audio>.
        return FileResponse(
            str(path), media_type="audio/wav", filename=f"segment_{segment_id}.wav",
            content_disposition_type="inline",
        )

    @app.post("/api/speakers/reattribute")
    def api_reattribute():
        with db() as conn:
            return {"relabeled": service.reattribute(conn, settings)}

    @app.get("/api/speakers/quality")
    def api_speaker_quality():
        with db() as conn:
            return service.speaker_quality(conn, settings)

    # --- person dossier (Phase 8A) -------------------------------------------

    @app.get("/api/person/{speaker_id}")
    def api_person(speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        with db() as conn:
            d = service.person_dossier(conn, speaker_id, settings)
        if d is None:
            raise HTTPException(404, "person not found")
        return _annotate_dossier_commitments(d)

    @app.get("/person/{speaker_id}", response_class=HTMLResponse)
    def person_page(request: Request, speaker_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        with db() as conn:
            resolved = service.resolve(conn, speaker_id)
            if resolved != speaker_id and service.speaker_label_for(conn, resolved) is not None:
                # A merged voice's old URL (bookmark, stale link) lands on the
                # one canonical profile instead of serving duplicate content.
                # (The JSON route keeps returning the dossier directly — its
                # "speaker_id" field already reports the canonical id.)
                return RedirectResponse(f"/person/{resolved}", status_code=307)
            d = service.person_dossier(conn, speaker_id, settings)
        if d is None:
            if _wants_html(request):
                # Browsers get a specific, navigable page; non-HTML clients
                # keep the exact JSON body below (CLI/menu bar contract).
                return templates.TemplateResponse(
                    request,
                    "error.html",
                    {
                        "status_code": 404,
                        "title": "Person not found",
                        "detail": f"there's no speaker #{speaker_id}",
                        "hint": "That voice may have been merged into another person or "
                        "removed. Everyone SecondBrain knows is on the People page.",
                        "links": [{"href": "/speakers", "label": "Go to People"}],
                    },
                    status_code=404,
                )
            raise HTTPException(404, "person not found")
        return templates.TemplateResponse(
            request, "person.html", {"d": _annotate_dossier_commitments(d)}
        )

    @app.get("/api/relationships")
    def api_relationships():
        with db() as conn:
            return {"relationships": service.relationships(conn, settings)}

    @app.get("/relationships", response_class=HTMLResponse)
    def relationships_page(request: Request):
        with db() as conn:
            rel = service.relationships(conn, settings)
        return templates.TemplateResponse(request, "relationships.html", {"relationships": rel})

    # --- project intelligence (Phase 9) --------------------------------------

    @app.get("/api/projects")
    def api_projects():
        with db() as conn:
            return {"projects": service.list_projects(conn, settings)}

    @app.get("/api/project/{node_id}")
    def api_project(node_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        with db() as conn:
            d = service.project_dossier(conn, node_id, settings)
        if d is None:
            raise HTTPException(404, "project not found")
        return d

    @app.get("/projects", response_class=HTMLResponse)
    def projects_page(request: Request):
        with db() as conn:
            projects = service.list_projects(conn, settings)
        return templates.TemplateResponse(request, "projects.html", {"projects": projects})

    @app.get("/project/{node_id}", response_class=HTMLResponse)
    def project_page(request: Request, node_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        with db() as conn:
            d = service.project_dossier(conn, node_id, settings)
        if d is None:
            if _wants_html(request):
                # Browsers get a specific, navigable page; non-HTML clients
                # keep the exact JSON body below (CLI/menu bar contract).
                return templates.TemplateResponse(
                    request,
                    "error.html",
                    {
                        "status_code": 404,
                        "title": "Project not found",
                        "detail": f"there's no project #{node_id}",
                        "hint": "It may have been merged into another project or removed. "
                        "Everything extracted so far is on the Projects page.",
                        "links": [{"href": "/projects", "label": "Go to Projects"}],
                    },
                    status_code=404,
                )
            raise HTTPException(404, "project not found")
        if d["node_id"] != node_id:
            # A merged node's old URL (bookmark, stale link) lands on the one
            # canonical project page instead of serving duplicate content.
            # (The JSON route keeps returning the dossier directly — its
            # "node_id" field already reports the canonical id.)
            return RedirectResponse(f"/project/{d['node_id']}", status_code=307)
        return templates.TemplateResponse(request, "project.html", {"d": d})

    # --- memory timeline (Phase 8C) ------------------------------------------

    @app.get("/api/timeline/{day}")
    def api_timeline(day: str):
        d = _parse_day(day)
        if d is None:
            raise HTTPException(422, "day must be a date like 2026-07-02 (YYYY-MM-DD)")
        # strptime tolerates non-padded input ("2026-7-2") — echo the canonical
        # form so API consumers see the same day key as the HTML routes.
        day = d.strftime("%Y-%m-%d")
        with db() as conn:
            return {"day": day, "conversations": service.timeline(conn, day, settings)}

    @app.get("/timeline", response_class=HTMLResponse)
    def timeline_today(request: Request):
        return timeline_page(request, None)

    @app.get("/timeline/{day}", response_class=HTMLResponse)
    def timeline_page(request: Request, day: str | None = None):
        requested = (day or "").strip()
        today = service.local_today()
        invalid_date = None
        parsed = _parse_day(requested)
        if not requested or requested == "today":
            day = today
        elif parsed is None:
            invalid_date, day = requested, today  # fall back gracefully, tell the user
        else:
            day = parsed.strftime("%Y-%m-%d")  # normalized (zero-padded)
        d = _parse_day(day)
        with db() as conn:
            blocks = service.timeline(conn, day, settings)
            nav = service.day_nav(conn, day, settings)
            paused = state.is_paused(conn, default=settings.consent.paused)
        is_today = day == today
        now_pct = None
        if is_today:
            now_local = datetime.now().astimezone()
            midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            now_pct = round(min((now_local - midnight).total_seconds() / 864.0, 100.0), 2)
        strip = service.timeline_strip(blocks, day)
        # Mirror the "now" mark onto the zoomed hour axis when it falls inside.
        zoom_now_pct = None
        zoom = strip["zoom"]
        if zoom and now_pct is not None and zoom["start_pct"] <= now_pct <= zoom["end_pct"]:
            zoom_now_pct = round(
                (now_pct - zoom["start_pct"]) / (zoom["end_pct"] - zoom["start_pct"]) * 100.0, 2
            )
        total_seconds = sum(b.get("duration_seconds") or 0.0 for b in blocks)
        last_block = max(blocks, key=lambda b: b.get("ended_at") or "") if blocks else None
        return templates.TemplateResponse(
            request,
            "timeline.html",
            {
                "day": day,
                "today": today,
                "is_today": is_today,
                "pretty_day": f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}",
                "prev_day": (d - timedelta(days=1)).strftime("%Y-%m-%d"),
                "next_day": (d + timedelta(days=1)).strftime("%Y-%m-%d"),
                "nav": nav,
                "blocks": blocks,
                "strip": strip,
                "now_pct": now_pct,
                "zoom_now_pct": zoom_now_pct,
                "total_lines": sum(b.get("segment_count") or 0 for b in blocks),
                "total_talk": service.duration_label(total_seconds),
                "first_time": blocks[0].get("start_time") if blocks else "",
                "last_time": last_block.get("end_time") if last_block else "",
                "tz_label": datetime.now().astimezone().tzname() or "",
                "invalid_date": invalid_date,
                "recording": settings.consent.recording_enabled and not paused,
            },
        )

    # --- knowledge graph + Q&A (Phase 3) -------------------------------------

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page(request: Request):
        with db() as conn:
            seg_count = conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()
            # Suggest a temporal question that can actually be answered: "today"
            # only when today has recordings, else fall back to yesterday/recent.
            today = service.local_today()
            yesterday = (datetime.now().astimezone() - timedelta(days=1)).strftime("%Y-%m-%d")
            if service.day_segment_count(conn, today):
                try_first = "What did I talk about today?"
            elif service.day_segment_count(conn, yesterday):
                try_first = "What did I talk about yesterday?"
            else:
                try_first = "What have I talked about recently?"
        return templates.TemplateResponse(
            request,
            "chat.html",
            {
                "seg_count": seg_count["n"] if seg_count else 0,
                "try_first": try_first,
                "llm_model": settings.llm.model,
                # The client aborts a little after the server would give up, so
                # a wedged model can never leave the page spinning forever.
                "llm_timeout_s": int(settings.llm.request_timeout_s),
            },
        )

    def _clean_history(history: list[dict] | None) -> list[dict]:
        # History is best-effort context: keep only well-formed recent turns so
        # a buggy client can't inflate the prompt or crash the endpoint. We keep
        # a slightly wider window than the prompt spells out (chat._history_block
        # spells out only the last few turns as prose but harvests citation ids
        # from all of them), so a follow-up reaching back to an early turn keeps
        # those sources resolvable. The cap still bounds worst-case work.
        return [
            {"question": str(t.get("question") or ""), "answer": str(t.get("answer") or "")}
            for t in (history or [])[-12:]
            if isinstance(t, dict) and t.get("question") and t.get("answer")
        ]

    def _llm_failure_detail(exc: Exception) -> str | None:
        """Human-readable message for an Ollama transport failure, else None."""
        import httpx

        if isinstance(exc, httpx.ConnectError):
            return (
                "Couldn't reach the local model — is Ollama running? "
                f"(expected at {settings.llm.host})"
            )
        if isinstance(exc, httpx.TimeoutException):
            return (
                f"The local model didn't answer within {int(settings.llm.request_timeout_s)}s. "
                "It may be busy loading — try again, or ask a simpler question."
            )
        if isinstance(exc, httpx.HTTPStatusError):
            hint = (
                f" Model '{settings.llm.model}' may not be pulled — try `ollama pull "
                f"{settings.llm.model}`." if exc.response.status_code == 404 else ""
            )
            return f"The local model returned an error (HTTP {exc.response.status_code}).{hint}"
        return None

    @app.post("/api/ask")
    def api_ask(
        question: str = Body(..., embed=True, min_length=1, max_length=4000),
        history: list[dict] | None = Body(None, embed=True),
    ):
        question = question.strip()
        if not question:
            raise HTTPException(400, "question is empty")
        turns = _clean_history(history)
        try:
            with db() as conn:
                return service.ask(conn, question, settings, history=turns)
        except Exception as e:
            detail = _llm_failure_detail(e)
            if detail is None:
                raise
            raise HTTPException(503, detail) from None

    @app.post("/api/ask/stream")
    async def api_ask_stream(
        question: str = Body(..., embed=True, min_length=1, max_length=4000),
        history: list[dict] | None = Body(None, embed=True),
    ):
        """Streaming variant of /api/ask used by the web chat (NDJSON lines:
        {"event":"delta","text":…}* then {"event":"done","result":<ask payload>},
        or {"event":"error","detail":…}). /api/ask itself is unchanged — the CLI
        and menu bar keep their one-shot JSON contract. Cancelling the request
        (Stop button / closed tab) propagates upstream and aborts the Ollama
        generation instead of letting it run to completion for nobody.
        """
        from secondbrain.knowledge import chat
        from secondbrain.llm.client import get_llm

        q = question.strip()
        if not q:
            raise HTTPException(400, "question is empty")
        turns = _clean_history(history)

        async def gen():
            def line(obj: dict) -> str:
                return json.dumps(obj, ensure_ascii=False) + "\n"

            try:
                # Retrieval is quick (SQLite); the connection is released before
                # the minutes-long generation starts.
                with db() as conn:
                    prep = chat.prepare(conn, q, settings=settings, history=turns)
                llm = get_llm(settings)
                parts: list[str] = []
                async for piece in llm.astream(
                    system=prep.system, prompt=prep.prompt, max_tokens=chat.MAX_ANSWER_TOKENS
                ):
                    parts.append(piece)
                    yield line({"event": "delta", "text": piece})
                yield line({"event": "done", "result": chat.finalize(prep, "".join(parts))})
            except Exception as e:  # noqa: BLE001 - stream already started: report in-band
                detail = _llm_failure_detail(e)
                if detail is None:
                    log.exception("ask stream failed")
                    detail = "Something went wrong while answering — details are in the server log."
                yield line({"event": "error", "detail": detail})

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/graph", response_class=HTMLResponse)
    def graph_page(request: Request):
        return templates.TemplateResponse(request, "graph.html", {})

    @app.get("/api/graph/search")
    def api_graph_search(
        q: str = Query("", max_length=200),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0, le=1_000_000_000),
        node_type: str | None = Query(
            None, alias="type", pattern="^(person|project|organization|topic|place)?$"
        ),
    ):
        """Search nodes by name or alias. An empty/omitted ``q`` returns the
        most connected nodes — the graph page's default browse list. ``offset``
        pages through the ranking (the page's "Show more") and ``type``
        narrows results to one entity type (the page's filter pills; empty =
        all types). ``total`` counts all matches, ``node_total`` the whole
        (non-merged) graph."""
        q = q.strip()
        node_type = node_type or None  # "" ≡ omitted: no type filter
        with db() as conn:
            nodes = service.graph_search(conn, q, limit, offset, node_type=node_type)
            total = service.graph_search_total(conn, q, node_type=node_type)
            unfiltered = not q and node_type is None
            node_total = total if unfiltered else service.graph_search_total(conn, "")
        return {"nodes": nodes, "total": total, "node_total": node_total, "offset": offset}

    @app.get("/api/graph/node/{node_id}")
    def api_graph_node(node_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        with db() as conn:
            node = service.graph_node(conn, node_id, settings)
        if node is None:
            raise HTTPException(
                404,
                "That entity isn't in the graph any more — "
                "it may have been merged into another or forgotten.",
            )
        return node

    # --- proactivity + goals (Phase 4) ---------------------------------------

    def _suggestion_citation_meta(conn, suggestions: list[dict]) -> list[dict]:
        """Resolve every segment id cited by ``suggestions`` for source links."""
        seg_ids: list[int] = []
        for s in suggestions:
            try:
                seg_ids.extend(int(i) for i in json.loads(s.get("citations") or "[]"))
            except (TypeError, ValueError):
                continue  # malformed citations JSON: item still renders, no links
        return service.citation_meta(conn, seg_ids)

    @app.get("/brief", response_class=HTMLResponse)
    def brief_page(request: Request, kind: str = Query("daily"), date: str = Query(None)):
        # HTML page: fall back gracefully on hand-mangled query params (the
        # JSON API validates strictly and 422s instead).
        kind = kind if kind in ("daily", "weekly") else "daily"
        today = service.local_today()  # the user's wall-clock day, like the rest of the UI
        with db() as conn:
            dates = service.list_digest_dates(conn)
            if _parse_day(date) is not None:
                day = date
            elif kind == "weekly" and dates.get("weekly"):
                day = dates["weekly"][0]  # latest weekly review, not empty today
            else:
                day = today
            digest = service.get_digest(conn, day, kind)
            suggestions = service.list_suggestions(conn, day)
            cite_meta = {c["segment_id"]: c for c in _suggestion_citation_meta(conn, suggestions)}
            for c in (digest or {}).get("citations", []):
                cite_meta[c["segment_id"]] = c
            generating = service.digest_generating(conn)
        state_payload = {
            "kind": kind, "date": day, "today": today, "dates": dates,
            "digest": digest, "suggestions": suggestions,
            # per-kind started-at of an in-flight generate run, so the page can
            # resume its progress line after a reload instead of going silent
            "generating": generating,
            "segMeta": {str(k): v for k, v in cite_meta.items()},
        }
        return templates.TemplateResponse(
            request,
            "brief.html",
            {"digest": digest, "suggestions": suggestions, "kind": kind,
             "date": day, "today": today, "state": state_payload},
        )

    @app.get("/api/digest")
    def api_digest(
        kind: str = Query("daily", pattern="^(daily|weekly)$"), date: str = Query(None)
    ):
        if date is not None and _parse_day(date) is None:
            raise HTTPException(422, "date must be a real YYYY-MM-DD date")
        with db() as conn:
            return service.get_digest(conn, date, kind) or {}

    @app.get("/api/digest/dates")
    def api_digest_dates():
        """Dates (newest first) that have a stored digest, per kind."""
        with db() as conn:
            return service.list_digest_dates(conn)

    @app.post("/api/digest/generate")
    def api_digest_generate(
        kind: str = Body("daily", embed=True), force: bool = Body(True, embed=True)
    ):
        from secondbrain.proactive.engine import DigestInFlight

        if kind not in ("daily", "weekly"):
            raise HTTPException(422, "kind must be 'daily' or 'weekly'")
        with db() as conn:
            try:
                return service.generate_digest(conn, settings, kind=kind, force=force) or {}
            except DigestInFlight as exc:
                what = "weekly review" if kind == "weekly" else "daily brief"
                elapsed = _elapsed_s(exc.started_at)
                raise HTTPException(
                    409,
                    f"That {what} is already being written"
                    + (f" (started {elapsed}s ago)" if elapsed is not None else "")
                    + " — it will appear here when it's done.",
                ) from exc

    @app.get("/api/digest/status")
    def api_digest_status(kind: str = Query("daily", pattern="^(daily|weekly)$")):
        """Whether a digest generation is in flight (additive; powers the brief
        page's reload-surviving progress line)."""
        with db() as conn:
            return service.digest_generation_status(conn, kind)

    @app.get("/api/suggestions")
    def api_suggestions(
        date: str = Query(None),
        status: str = Query("open", pattern="^(open|done|dismissed|snoozed)$"),
    ):
        if date is not None and _parse_day(date) is None:
            raise HTTPException(422, "date must be a real YYYY-MM-DD date")
        with db() as conn:
            suggestions = service.list_suggestions(conn, date, status)
            citations = _suggestion_citation_meta(conn, suggestions)
        return {"suggestions": suggestions, "citations": citations}

    _SUGGESTION_ACTIONS = ("done", "dismiss", "snooze", "up", "down", "reopen")

    @app.post("/api/suggestions/{suggestion_id}/action")
    def api_suggestion_action(
        suggestion_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        action: str = Body(..., embed=True),
    ):
        if action not in _SUGGESTION_ACTIONS:
            raise HTTPException(
                422, f"action must be one of: {', '.join(_SUGGESTION_ACTIONS)}"
            )
        with db() as conn:
            found = service.suggestion_action(conn, suggestion_id, action)
        if not found:
            raise HTTPException(404, "suggestion not found — it may have been cleaned up")
        return {"ok": True}

    def _goal_or_404(conn, goal_id: int) -> dict:
        found = service.get_goal(conn, goal_id)
        if found is None:
            raise HTTPException(404, "goal not found — it may have been deleted")
        return found

    def _validated_target_date(target_date: str | None) -> str | None:
        """'' and None clear the date; anything else must be a real day."""
        if not target_date:
            return None
        if _parse_day(target_date) is None:
            raise HTTPException(422, "target_date must be a real YYYY-MM-DD date")
        return target_date

    @app.get("/goals", response_class=HTMLResponse)
    def goals_page(request: Request, status: str = Query(None)):
        # HTML page: fall back gracefully on hand-mangled query params (the
        # JSON API validates strictly and 422s instead).
        status_f = status if status in _GOAL_STATUSES else None
        today = service.local_today()
        with db() as conn:
            goals = service.list_goals(conn, status_f)
            counts = service.goal_status_counts(conn)
        for g in goals:  # display-only annotations
            g["overdue"] = bool(
                g["target_date"] and g["status"] == "active" and g["target_date"] < today
            )
            g["target_label"] = "today" if g["target_date"] == today else _fmt_day(g["target_date"])
            g["progress_label"] = _rel_ago(g["last_progress_at"])
        return templates.TemplateResponse(
            request,
            "goals.html",
            {"goals": goals, "counts": counts, "status_filter": status_f, "today": today},
        )

    @app.get("/api/goals")
    def api_goals(status: str = Query(None)):
        if status is not None and status not in _GOAL_STATUSES:
            raise HTTPException(422, f"status must be one of: {', '.join(_GOAL_STATUSES)}")
        with db() as conn:
            return {"goals": service.list_goals(conn, status)}

    @app.get("/api/goals/{goal_id}")
    def api_get_goal(goal_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """One goal plus its knowledge-graph links (evidence for auto-linking)."""
        with db() as conn:
            return _goal_or_404(conn, goal_id)

    @app.post("/api/goals")
    def api_create_goal(
        title: str = Body(..., max_length=500),
        description: str = Body(None, max_length=4000),
        target_date: str = Body(None),
        priority: int = Body(2, ge=1, le=3),
    ):
        title = title.strip()
        if not title:
            raise HTTPException(422, "title can't be empty")
        target_date = _validated_target_date(target_date)
        with db() as conn:
            gid = service.create_goal(
                conn, title=title, description=(description or "").strip() or None,
                target_date=target_date, priority=priority, settings=settings,
            )
        return {"id": gid}

    @app.patch("/api/goals/{goal_id}")
    def api_update_goal(
        goal_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        title: str = Body(None, max_length=500),
        description: str = Body(None, max_length=4000),
        target_date: str = Body(None),
        priority: int = Body(None, ge=1, le=3),
        status: str = Body(None),
    ):
        """Edit a goal in place (re-embeds + re-links, keeping goal_links
        history — no delete/recreate needed to fix a typo). Omitted fields are
        unchanged; empty-string description/target_date clear the value."""
        fields: dict = {}
        if title is not None:
            title = title.strip()
            if not title:
                raise HTTPException(422, "title can't be empty")
            fields["title"] = title
        if description is not None:
            fields["description"] = description.strip() or None
        if target_date is not None:
            fields["target_date"] = _validated_target_date(target_date)
        if priority is not None:
            fields["priority"] = priority
        if status is not None:
            if status not in _GOAL_STATUSES:
                raise HTTPException(422, f"status must be one of: {', '.join(_GOAL_STATUSES)}")
            fields["status"] = status
        if not fields:
            raise HTTPException(422, "nothing to update — send at least one field")
        with db() as conn:
            _goal_or_404(conn, goal_id)
            service.update_goal(conn, goal_id, settings=settings, **fields)
            return {"ok": True, "goal": _goal_or_404(conn, goal_id)["goal"]}

    @app.post("/api/goals/{goal_id}/status")
    def api_goal_status(
        goal_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        status: str = Body(..., embed=True),
    ):
        if status not in _GOAL_STATUSES:
            raise HTTPException(422, f"status must be one of: {', '.join(_GOAL_STATUSES)}")
        with db() as conn:
            _goal_or_404(conn, goal_id)
            service.set_goal_status(conn, goal_id, status)
        return {"ok": True}

    @app.delete("/api/goals/{goal_id}")
    def api_delete_goal(goal_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        with db() as conn:
            _goal_or_404(conn, goal_id)
            service.delete_goal(conn, goal_id)
        return {"ok": True}

    # --- tasks + daily planning (Phase 6) ------------------------------------

    def _task_or_404(conn, task_id: int) -> dict:
        found = service.get_task(conn, task_id)
        if found is None:
            raise HTTPException(404, "task not found — it may have been removed")
        return found

    def _validated_due_date(due_date: str | None) -> str | None:
        """'' and None clear the date; anything else must be a real day."""
        if not due_date:
            return None
        if _parse_day(due_date) is None:
            raise HTTPException(422, "due_date must be a real YYYY-MM-DD date")
        return due_date

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks_page(request: Request):
        today = service.local_today()
        with db() as conn:
            # Self-heal before rendering: tasks accepted into a past day's plan
            # but never finished would keep a stale 'scheduled' pill that
            # contradicts a Today section with no plan yet.
            service.release_stale_scheduled(conn, today)
            plan = service.get_day(conn)
            tasks = service.annotate_task_priorities(conn, service.list_tasks(conn), settings)
            note_counts = service.task_research_note_counts(conn)
            actions = service.list_action_items(conn)
            goal_titles = {g["id"]: g["title"] for g in service.list_goals(conn)}

        for t in tasks:  # display-only annotations
            t.update(_due_info(t["due_date"], today))
            t["notes_count"] = note_counts.get(t["id"], 0)
            t["closed_label"] = _rel_ago(t["completed_at"] or t["updated_at"])
            t["goal_title"] = goal_titles.get(t["goal_id"])  # link rows to their goal
        by_id = {t["id"]: t for t in tasks}
        # A task appears exactly once on the page: planned tasks render in the
        # Today section, everything else in Backlog / Completed.
        plan_ids = set(plan["task_ids"]) if plan else set()
        plan_tasks = [by_id[tid] for tid in plan["task_ids"] if tid in by_id] if plan else []
        open_tasks = [
            t for t in tasks
            if t["status"] not in ("done", "dropped") and t["id"] not in plan_ids
        ]
        open_tasks.sort(key=lambda t: t["priority_score"], reverse=True)
        completed = [
            t for t in tasks
            if t["status"] in ("done", "dropped") and t["id"] not in plan_ids
        ]
        completed.sort(key=lambda t: t["completed_at"] or t["updated_at"] or "", reverse=True)
        for a in actions:
            a["detected_label"] = _rel_ago(a["first_seen"])
            a.update(_due_info(a["due_date"], today))

        d = _parse_day(today)
        n_open_total = sum(1 for t in tasks if t["status"] not in ("done", "dropped"))
        return templates.TemplateResponse(
            request,
            "tasks.html",
            {
                "plan": plan,
                "plan_tasks": plan_tasks,
                "plan_done": sum(1 for t in plan_tasks if t["status"] == "done"),
                "planned_minutes": sum(
                    t["estimate_minutes"] or _DEFAULT_TASK_MINUTES for t in plan_tasks
                ),
                "open_tasks": open_tasks,
                "completed": completed,
                "actions": actions,
                "n_open_total": n_open_total,
                "default_capacity": settings.tasks.daily_capacity_minutes,
                "today_label": f"{d.strftime('%A')} {d.day} {d.strftime('%B')}" if d else today,
                # Config-gated: when on, task rows also offer "Research (web)".
                "web_research_enabled": settings.tasks.web_research_enabled,
            },
        )

    @app.get("/api/tasks")
    def api_tasks(goal_id: int = Query(None, ge=1, le=_SQLITE_MAX_INT), status: str = Query(None)):
        if status is not None and status not in _TASK_STATUSES:
            raise HTTPException(422, f"status must be one of: {', '.join(_TASK_STATUSES)}")
        with db() as conn:
            tasks = service.list_tasks(conn, goal_id=goal_id, status=status)
            # quadrant/priority_score are additive fields (Eisenhower view).
            return {"tasks": service.annotate_task_priorities(conn, tasks, settings)}

    @app.post("/api/tasks")
    def api_create_task(title: str = Body(..., max_length=500),
                        goal_id: int = Body(None, ge=1, le=_SQLITE_MAX_INT),
                        detail: str = Body(None, max_length=4000),
                        estimate_minutes: int = Body(None, ge=1, le=1440),
                        value: int = Body(3, ge=1, le=5),
                        effort: int = Body(3, ge=1, le=5), due_date: str = Body(None)):
        title = title.strip()
        if not title:
            raise HTTPException(422, "title can't be empty")
        due_date = _validated_due_date(due_date)
        with db() as conn:
            if goal_id is not None and service.get_goal(conn, goal_id) is None:
                raise HTTPException(404, "goal not found — it may have been deleted")
            tid = service.create_task(conn, title=title, goal_id=goal_id,
                                      detail=(detail or "").strip() or None,
                                      estimate_minutes=estimate_minutes, value=value,
                                      effort=effort, due_date=due_date)
        return {"id": tid}

    @app.patch("/api/tasks/{task_id}")
    def api_update_task(
        task_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        title: str = Body(None, max_length=500),
        detail: str = Body(None, max_length=4000),
        estimate_minutes: int = Body(None, ge=0, le=1440),
        due_date: str = Body(None),
        value: int = Body(None, ge=1, le=5),
        effort: int = Body(None, ge=1, le=5),
    ):
        """Edit a task in place (fix a typo, set a due date/estimate). Omitted
        fields are unchanged; empty-string due_date and estimate 0 clear them."""
        fields: dict = {}
        if title is not None:
            title = title.strip()
            if not title:
                raise HTTPException(422, "title can't be empty")
            fields["title"] = title
        if detail is not None:
            fields["detail"] = detail.strip() or None
        if estimate_minutes is not None:
            fields["estimate_minutes"] = estimate_minutes or None  # 0 clears
        if due_date is not None:
            fields["due_date"] = _validated_due_date(due_date)
        if value is not None:
            fields["value"] = value
        if effort is not None:
            fields["effort"] = effort
        if not fields:
            raise HTTPException(422, "nothing to update — send at least one field")
        with db() as conn:
            _task_or_404(conn, task_id)
            service.update_task(conn, task_id, **fields)
            task = service.annotate_task_priorities(
                conn, [service.get_task(conn, task_id)], settings
            )[0]
        return {"ok": True, "task": task}

    @app.post("/api/tasks/{task_id}/status")
    def api_task_status(
        task_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        status: str = Body(..., embed=True),
    ):
        if status not in _TASK_STATUSES:
            raise HTTPException(422, f"status must be one of: {', '.join(_TASK_STATUSES)}")
        with db() as conn:
            _task_or_404(conn, task_id)
            service.task_set_status(conn, task_id, status)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/research")
    def api_task_research(
        task_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        web: bool = Body(False, embed=True),
    ):
        with db() as conn:
            _task_or_404(conn, task_id)  # before the slow LLM call, not after
            try:
                note_id = service.task_research(conn, task_id, web=web, settings=settings)
            except RuntimeError as exc:  # web research disabled / not configured
                raise HTTPException(400, str(exc)) from None
            except Exception as e:
                detail = _llm_failure_detail(e)
                if detail is None:
                    raise
                raise HTTPException(503, detail) from None
            notes = service.task_research_notes(conn, task_id)
        return {"id": note_id, "notes": notes}

    def _seg_ref_id(source: dict) -> int | None:
        """seg:<id> source ref → segment id (None for web/malformed refs)."""
        ref = str(source.get("ref") or "")
        return int(ref[4:]) if ref.startswith("seg:") and ref[4:].isdigit() else None

    @app.get("/api/tasks/{task_id}/research")
    def api_task_research_notes(task_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Stored research notes for a task, newest first (the POST endpoint
        keeps its original response shape; this read view parses sources).
        Each seg:<id> source carries the local ``day`` it was heard on so the
        UI can link it to /day?date=<day>#seg-<id> — backfilled here for notes
        stored before the day was recorded at research time."""
        with db() as conn:
            _task_or_404(conn, task_id)
            notes = service.task_research_notes(conn, task_id)
            for n in notes:
                try:
                    n["sources"] = json.loads(n["sources"] or "[]")
                except (TypeError, ValueError):
                    n["sources"] = []
                n["sources"] = [s for s in n["sources"] if isinstance(s, dict)]
                n["created_label"] = _rel_ago(n["created_at"])
            missing = {sid for n in notes for s in n["sources"]
                       if not s.get("day") and (sid := _seg_ref_id(s)) is not None}
            days = service.segment_local_days(conn, list(missing)) if missing else {}
        for n in notes:
            for s in n["sources"]:
                sid = _seg_ref_id(s)
                if sid is not None and not s.get("day") and sid in days:
                    s["day"] = days[sid]
        return {"task_id": task_id, "notes": notes}

    @app.post("/api/actions/{edge_id}/promote")
    def api_promote_action(edge_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Turn a detected conversation action item into a backlog task
        (idempotent — promoting twice returns the same task)."""
        with db() as conn:
            tid = service.promote_action_item(conn, edge_id)
            if tid is None:
                raise HTTPException(
                    404, "action item not found — it may have been superseded or forgotten"
                )
            task = service.get_task(conn, tid)
        return {"ok": True, "task_id": tid, "title": task["title"] if task else None}

    @app.post("/api/actions/{edge_id}/dismiss")
    def api_dismiss_action(edge_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Drop a detected action item that isn't a real to-do (the extractor
        over-triggers on casual phrasing). The edge is marked invalid — never
        deleted — so it leaves the Tasks page for good; already-promoted tasks
        are unaffected. Idempotent."""
        with db() as conn:
            if not service.dismiss_action_item(conn, edge_id):
                raise HTTPException(
                    404, "action item not found — it may have been superseded or forgotten"
                )
        return {"ok": True}

    def _plan_envelope(plan: dict | None) -> dict:
        """Consistent plan responses: ``plan`` is always present (null when no
        plan exists). Plan fields are also mirrored at the top level so older
        consumers that read them there keep working — never removed."""
        return {"plan": plan, **(plan or {})}

    @app.get("/api/plan/today")
    def api_plan_today():
        """Read-only view of today's plan. Never creates one — a bare GET (a
        poller, a prefetch) must not mutate state; proposing lives in POST."""
        with db() as conn:
            return _plan_envelope(service.get_day(conn))

    @app.post("/api/plan/today")
    def api_plan_action(
        action: str = Body("propose", embed=True),
        capacity_minutes: int = Body(None, embed=True, ge=15, le=1440),
        task_id: int = Body(None, embed=True, ge=1, le=_SQLITE_MAX_INT),
    ):
        if action not in ("propose", "accept", "remove_task"):
            raise HTTPException(422, "action must be 'propose', 'accept' or 'remove_task'")
        with db() as conn:
            if action == "accept":
                plan = service.get_day(conn)
                if plan is None:
                    raise HTTPException(409, "no plan to accept yet — propose one first")
                if not plan["task_ids"]:
                    raise HTTPException(
                        409, "the proposed plan is empty — add a task, then re-propose"
                    )
                return _plan_envelope(service.accept_day(conn))
            if action == "remove_task":
                # "Not today": pull one task out of the plan without re-proposing.
                if task_id is None:
                    raise HTTPException(422, "task_id is required to remove a task")
                plan = service.get_day(conn)
                if plan is None:
                    raise HTTPException(409, "no plan for today yet — nothing to remove from")
                if task_id not in plan["task_ids"]:
                    raise HTTPException(404, "that task isn't in today's plan")
                return _plan_envelope(service.remove_from_day(conn, task_id))
            return _plan_envelope(service.propose_day(conn, capacity_minutes=capacity_minutes,
                                                      settings=settings))

    @app.post("/api/goals/{goal_id}/decompose")
    def api_decompose(goal_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT)):
        """Ask the local model for a milestones→tasks plan. Proposes only —
        nothing is persisted until /plan/accept."""
        from secondbrain.llm.jsonout import LLMJSONError

        with db() as conn:
            _goal_or_404(conn, goal_id)
            try:
                return service.decompose_goal(conn, goal_id, settings)
            except (LLMJSONError, ValidationError) as exc:
                raise HTTPException(
                    502,
                    "The local model returned an unusable plan — try again "
                    "(small models sometimes need a second attempt).",
                ) from exc
            except Exception as e:
                detail = _llm_failure_detail(e)
                if detail is None:
                    raise
                raise HTTPException(503, detail) from None

    @app.post("/api/goals/{goal_id}/plan/accept")
    def api_accept_plan(
        goal_id: int = PathParam(ge=1, le=_SQLITE_MAX_INT),
        plan: dict = Body(...),
    ):
        with db() as conn:
            _goal_or_404(conn, goal_id)
            try:
                ids = service.accept_plan(conn, goal_id, plan)
            except ValidationError as exc:
                raise HTTPException(
                    422, "plan doesn't match the expected shape (milestones → tasks)"
                ) from exc
        return {"task_ids": ids}

    return app


def _load_soundfile():
    """The optional audio slicing backend, or None when the extra is absent."""
    try:
        import soundfile as sf  # lazy: `audio` extra
    except ImportError:
        return None
    return sf


def _extract_clip(sample: dict, settings: Settings, prefix: str = "sample"):
    """Slice [start,end] from the source audio into a cached WAV, or None if gone.

    The extracted WAV is derived raw audio, so it follows the same retention
    policy as its source: once the source chunk was swept, the cached clip is
    removed here too (and by the retention sweeper) instead of being served.
    ``prefix`` namespaces the cache file: "sample" for speaker-observation
    clips, "segclip" for per-transcript-line clips (ids come from different
    tables, so they must not share cache filenames). The cache name also
    carries the window (``{prefix}_{id}_{start}-{end}.wav``, centiseconds), so
    when a clip's serve window improves later — e.g. segment attribution
    catches up and a sub-second exemplar blip is upgraded to a full spoken
    line — the stale cached slice is replaced instead of being served forever.

    Raises HTTPException 503 when the audio can't be sliced because the
    optional soundfile backend isn't installed: that's a fixable install gap,
    not "audio expired", and callers must not report it as a 410.
    """
    start = float(sample["start_offset_s"] or 0.0)
    end = float(sample["end_offset_s"] or 0.0)
    window = f"{int(round(max(0.0, start) * 100))}-{int(round(max(start, end) * 100))}"
    out = settings.audio_processed_dir / f"{prefix}_{sample['id']}_{window}.wav"

    def _drop_cached(keep: Path | None = None) -> None:
        """Remove this clip's cache files (legacy unwindowed name included)."""
        legacy = settings.audio_processed_dir / f"{prefix}_{sample['id']}.wav"
        stale = settings.audio_processed_dir.glob(f"{prefix}_{sample['id']}_*.wav")
        for p in (legacy, *stale):
            if p != keep:
                with contextlib.suppress(OSError):
                    p.unlink(missing_ok=True)

    src = Path(sample["path"])
    if sample.get("audio_status") == "deleted" or not src.exists():
        _drop_cached()  # a cached clip must not outlive its source
        return None
    if out.exists():  # already sliced for an earlier listen — reuse, don't re-read
        return out
    sf = _load_soundfile()
    if sf is None:
        raise HTTPException(
            503,
            "audio playback needs the optional audio backend (soundfile) — "
            "install it with: pip install 'secondbrain[audio]'",
        )
    audio, sr = sf.read(str(src))
    a = int(max(0, start) * sr)
    b = int(max(start, end) * sr) or len(audio)
    settings.audio_processed_dir.mkdir(parents=True, exist_ok=True)
    _drop_cached(keep=out)  # superseded windows for this id go before the new slice
    sf.write(str(out), audio[a:b], sr)
    return out
