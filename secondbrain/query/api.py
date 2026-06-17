"""FastAPI app exposing search/browse + recording controls. Local-only.

Always bind to 127.0.0.1 (see config.api.host). The same backend serves the web
dashboard, the menu bar app, and the CLI.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
