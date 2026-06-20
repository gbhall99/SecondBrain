"""FastAPI app exposing search/browse + recording controls. Local-only.

Always bind to 127.0.0.1 (see config.api.host). The same backend serves the web
dashboard, the menu bar app, and the CLI.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from secondbrain import health
from secondbrain.config import Settings, get_settings
from secondbrain.query import service
from secondbrain.security import auth
from secondbrain.storage import state
from secondbrain.storage.db import db_session, init_db

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="SecondBrain", version="0.1.0")
    templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
    static_dir = _WEB_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Session signing secret (created on first use; lives in app_state).
    init_db(settings=settings).close()
    with db_session(settings=settings) as _c:
        secret = auth.session_secret(_c)

    def db():
        return db_session(settings=settings)

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        if settings.security.require_auth:
            host = request.client.host if request.client else None
            path = request.url.path
            if not auth.is_loopback(host) and not auth.is_exempt(path):
                cookie = request.cookies.get(auth.COOKIE_NAME, "")
                if auth.verify_cookie(cookie, secret) is None:
                    if path.startswith("/api/"):
                        return JSONResponse({"detail": "authentication required"}, status_code=401)
                    return RedirectResponse("/login", status_code=303)
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    def health_endpoint():
        with db() as conn:
            return health.summary(conn, settings)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html", {})

    @app.post("/login")
    def login(request: Request, username: str = Body(...), password: str = Body(...)):
        ip = request.client.host if request.client else "?"
        if not auth.login_allowed(ip):
            raise HTTPException(429, "too many attempts; try again shortly")
        with db() as conn:
            ok = auth.authenticate(conn, username, password)
        if not ok:
            auth.record_login_failure(ip)
            raise HTTPException(401, "invalid credentials")
        auth.reset_login_failures(ip)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            auth.COOKIE_NAME,
            auth.make_cookie(username, secret, settings.security.session_max_age_days),
            max_age=settings.security.session_max_age_days * 86400,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",  # set Secure when served over TLS
        )
        return resp

    @app.post("/logout")
    def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(auth.COOKIE_NAME)
        return resp

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        with db() as conn:
            st = service.status(conn, settings)
        return templates.TemplateResponse(request, "index.html", {"status": st})

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
    ):
        with db() as conn:
            results = service.search(conn, q, limit, mode, settings, since=since, until=until)
        return {"query": q, "mode": mode, "results": results}

    @app.get("/api/day/{day}")
    def api_day(day: str):
        with db() as conn:
            return {"day": day, "segments": service.day_segments(conn, day)}

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
            known = service.list_speakers(conn)
        return templates.TemplateResponse(
            request, "speakers.html", {"unknowns": unknowns, "known": known}
        )

    @app.get("/api/speakers")
    def api_speakers():
        with db() as conn:
            return {"speakers": service.list_speakers(conn)}

    @app.get("/api/speakers/unknown")
    def api_unknown():
        with db() as conn:
            return {"unknown": service.unknown_speakers(conn)}

    @app.get("/api/speakers/{speaker_id}/samples")
    def api_samples(speaker_id: int):
        with db() as conn:
            return {"samples": service.speaker_samples(conn, speaker_id)}

    @app.get("/api/speakers/{speaker_id}/clip/{observation_id}")
    def api_clip(speaker_id: int, observation_id: int):
        with db() as conn:
            samples = service.speaker_samples(conn, speaker_id, n=50)
        sample = next((s for s in samples if s["id"] == observation_id), None)
        if sample is None:
            raise HTTPException(404, "observation not found")
        path = _extract_clip(sample, settings)
        if path is None:
            raise HTTPException(410, "audio expired (deleted by retention)")
        return FileResponse(
            str(path), media_type="audio/wav", filename=f"clip_{observation_id}.wav"
        )

    @app.post("/api/speakers/{speaker_id}/name")
    def api_name(speaker_id: int, name: str = Body(..., embed=True)):
        with db() as conn:
            redacted = service.name_speaker(conn, speaker_id, name, settings)
        return {"ok": True, "redacted_segments": redacted}

    @app.post("/api/speakers/merge")
    def api_merge(src: int = Body(...), dst: int = Body(...)):
        with db() as conn:
            n = service.merge_speakers(conn, src, dst, settings)
        return {"ok": True, "relabeled_segments": n}

    @app.post("/api/speakers/{speaker_id}/owner")
    def api_set_owner(speaker_id: int):
        with db() as conn:
            service.set_owner(conn, speaker_id)
        return {"ok": True}

    # --- speaker quality / corrections (Phase 7) -----------------------------

    @app.get("/day", response_class=HTMLResponse)
    def day_page(request: Request, date: str = Query(None)):
        with db() as conn:
            segments = service.day_segments(conn, date)
            speakers = service.list_speakers(conn)
        return templates.TemplateResponse(
            request, "day.html", {"segments": segments, "speakers": speakers, "date": date or ""}
        )

    @app.post("/api/segments/{segment_id}/speaker")
    def api_reassign_segment(segment_id: int, speaker_id: int = Body(..., embed=True)):
        with db() as conn:
            ok = service.reassign_segment(conn, segment_id, speaker_id, settings)
        return {"ok": ok}

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
    def api_person(speaker_id: int):
        with db() as conn:
            d = service.person_dossier(conn, speaker_id, settings)
        if d is None:
            raise HTTPException(404, "person not found")
        return d

    @app.get("/person/{speaker_id}", response_class=HTMLResponse)
    def person_page(request: Request, speaker_id: int):
        with db() as conn:
            d = service.person_dossier(conn, speaker_id, settings)
        if d is None:
            raise HTTPException(404, "person not found")
        return templates.TemplateResponse(request, "person.html", {"d": d})

    @app.get("/api/relationships")
    def api_relationships():
        with db() as conn:
            return {"relationships": service.relationships(conn, settings)}

    @app.get("/relationships", response_class=HTMLResponse)
    def relationships_page(request: Request):
        with db() as conn:
            rel = service.relationships(conn, settings)
        return templates.TemplateResponse(request, "relationships.html", {"relationships": rel})

    # --- knowledge graph + Q&A (Phase 3) -------------------------------------

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page(request: Request):
        return templates.TemplateResponse(request, "chat.html", {})

    @app.post("/api/ask")
    def api_ask(question: str = Body(..., embed=True)):
        with db() as conn:
            return service.ask(conn, question, settings)

    @app.get("/graph", response_class=HTMLResponse)
    def graph_page(request: Request):
        return templates.TemplateResponse(request, "graph.html", {})

    @app.get("/api/graph/search")
    def api_graph_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
        with db() as conn:
            return {"nodes": service.graph_search(conn, q, limit)}

    @app.get("/api/graph/node/{node_id}")
    def api_graph_node(node_id: int):
        with db() as conn:
            node = service.graph_node(conn, node_id)
        if node is None:
            raise HTTPException(404, "node not found")
        return node

    # --- proactivity + goals (Phase 4) ---------------------------------------

    @app.get("/brief", response_class=HTMLResponse)
    def brief_page(request: Request):
        with db() as conn:
            daily = service.get_digest(conn, kind="daily")
            suggestions = service.list_suggestions(conn)
        return templates.TemplateResponse(
            request, "brief.html", {"digest": daily, "suggestions": suggestions}
        )

    @app.get("/api/digest")
    def api_digest(kind: str = Query("daily"), date: str = Query(None)):
        with db() as conn:
            return service.get_digest(conn, date, kind) or {}

    @app.post("/api/digest/generate")
    def api_digest_generate(
        kind: str = Body("daily", embed=True), force: bool = Body(True, embed=True)
    ):
        with db() as conn:
            return service.generate_digest(conn, settings, kind=kind, force=force) or {}

    @app.get("/api/suggestions")
    def api_suggestions(date: str = Query(None), status: str = Query("open")):
        with db() as conn:
            return {"suggestions": service.list_suggestions(conn, date, status)}

    @app.post("/api/suggestions/{suggestion_id}/action")
    def api_suggestion_action(suggestion_id: int, action: str = Body(..., embed=True)):
        with db() as conn:
            service.suggestion_action(conn, suggestion_id, action)
        return {"ok": True}

    @app.get("/goals", response_class=HTMLResponse)
    def goals_page(request: Request):
        with db() as conn:
            goals = service.list_goals(conn)
        return templates.TemplateResponse(request, "goals.html", {"goals": goals})

    @app.get("/api/goals")
    def api_goals():
        with db() as conn:
            return {"goals": service.list_goals(conn)}

    @app.post("/api/goals")
    def api_create_goal(
        title: str = Body(...),
        description: str = Body(None),
        target_date: str = Body(None),
        priority: int = Body(2),
    ):
        with db() as conn:
            gid = service.create_goal(
                conn, title=title, description=description,
                target_date=target_date, priority=priority, settings=settings,
            )
        return {"id": gid}

    @app.post("/api/goals/{goal_id}/status")
    def api_goal_status(goal_id: int, status: str = Body(..., embed=True)):
        with db() as conn:
            service.set_goal_status(conn, goal_id, status)
        return {"ok": True}

    @app.delete("/api/goals/{goal_id}")
    def api_delete_goal(goal_id: int):
        with db() as conn:
            service.delete_goal(conn, goal_id)
        return {"ok": True}

    # --- tasks + daily planning (Phase 6) ------------------------------------

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks_page(request: Request):
        with db() as conn:
            today = service.get_day(conn)
            backlog = service.list_tasks(conn)
        return templates.TemplateResponse(
            request, "tasks.html", {"today": today, "backlog": backlog}
        )

    @app.get("/api/tasks")
    def api_tasks(goal_id: int = Query(None), status: str = Query(None)):
        with db() as conn:
            return {"tasks": service.list_tasks(conn, goal_id=goal_id, status=status)}

    @app.post("/api/tasks")
    def api_create_task(title: str = Body(...), goal_id: int = Body(None),
                        estimate_minutes: int = Body(None), value: int = Body(3),
                        effort: int = Body(3), due_date: str = Body(None)):
        with db() as conn:
            tid = service.create_task(conn, title=title, goal_id=goal_id,
                                      estimate_minutes=estimate_minutes, value=value,
                                      effort=effort, due_date=due_date)
        return {"id": tid}

    @app.post("/api/tasks/{task_id}/status")
    def api_task_status(task_id: int, status: str = Body(..., embed=True)):
        with db() as conn:
            service.task_set_status(conn, task_id, status)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/research")
    def api_task_research(task_id: int, web: bool = Body(False, embed=True)):
        with db() as conn:
            note_id = service.task_research(conn, task_id, web=web, settings=settings)
            notes = service.task_research_notes(conn, task_id)
        return {"id": note_id, "notes": notes}

    @app.get("/api/plan/today")
    def api_plan_today():
        with db() as conn:
            return service.get_day(conn) or service.propose_day(conn, settings=settings)

    @app.post("/api/plan/today")
    def api_plan_action(action: str = Body("propose", embed=True)):
        with db() as conn:
            if action == "accept":
                return service.accept_day(conn) or {}
            return service.propose_day(conn, settings=settings)

    @app.post("/api/goals/{goal_id}/decompose")
    def api_decompose(goal_id: int):
        with db() as conn:
            return service.decompose_goal(conn, goal_id, settings)

    @app.post("/api/goals/{goal_id}/plan/accept")
    def api_accept_plan(goal_id: int, plan: dict = Body(...)):
        with db() as conn:
            ids = service.accept_plan(conn, goal_id, plan)
        return {"task_ids": ids}

    return app


def _extract_clip(sample: dict, settings: Settings):
    """Slice [start,end] from the source audio into a temp WAV, or None if gone."""
    src = Path(sample["path"])
    if sample.get("audio_status") == "deleted" or not src.exists():
        return None
    try:
        import soundfile as sf  # lazy: `audio` extra
    except ImportError:
        return None
    start = float(sample["start_offset_s"] or 0.0)
    end = float(sample["end_offset_s"] or 0.0)
    audio, sr = sf.read(str(src))
    a = int(max(0, start) * sr)
    b = int(max(start, end) * sr) or len(audio)
    out = settings.audio_processed_dir / f"sample_{sample['id']}.wav"
    settings.audio_processed_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio[a:b], sr)
    return out
