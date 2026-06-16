"""FastAPI app exposing search/browse + recording controls. Local-only.

Always bind to 127.0.0.1 (see config.api.host). The same backend serves the web
dashboard, the menu bar app, and the CLI.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from secondbrain.config import Settings, get_settings
from secondbrain.query import service
from secondbrain.storage import state
from secondbrain.storage.db import db_session

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="SecondBrain", version="0.1.0")
    templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
    static_dir = _WEB_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def db():
        return db_session(settings=settings)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        with db() as conn:
            st = service.status(conn, settings)
        return templates.TemplateResponse(request, "index.html", {"status": st})

    @app.get("/api/status")
    def api_status():
        with db() as conn:
            return service.status(conn, settings)

    @app.get("/api/search")
    def api_search(
        q: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=200),
        mode: str = Query("auto", pattern="^(auto|fulltext|semantic)$"),
    ):
        with db() as conn:
            results = service.search(conn, q, limit, mode, settings)
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

    return app
