"""SecondBrain command-line interface (Typer).

Examples:
    sb init                      # create data dirs + database
    sb devices                   # list audio input devices
    sb start                     # run the always-on capture+transcribe daemon
    sb serve                     # run the local web UI / API
    sb status                    # recording status, queue, disk
    sb search "what did we decide about pricing"
    sb show 2026-06-16           # all segments for a day
    sb pause / sb resume         # toggle recording live
    sb drain                     # process pending transcription jobs now
    sb sweep                     # delete expired raw audio per retention policy
"""

from __future__ import annotations

import json

import typer

from secondbrain.config import get_settings
from secondbrain.query import service
from secondbrain.storage import retention, state
from secondbrain.storage.db import db_session, init_db

app = typer.Typer(no_args_is_help=True, add_completion=False, help="SecondBrain CLI")


@app.command()
def init() -> None:
    """Create data directories and initialise the database."""
    settings = get_settings()
    settings.ensure_dirs()
    init_db(settings=settings).close()
    typer.echo(f"Initialised SecondBrain at {settings.data_path}")


@app.command()
def devices() -> None:
    """List available audio input devices."""
    from secondbrain.capture.devices import list_input_devices

    for d in list_input_devices():
        mark = " (default)" if d.default else ""
        typer.echo(f"[{d.index}] {d.name} — {d.channels}ch{mark}")


@app.command()
def start() -> None:
    """Run the always-on capture + transcription daemon (foreground)."""
    from secondbrain.daemon import main as daemon_main

    daemon_main()


@app.command()
def serve(
    host: str = typer.Option(None, help="Override bind host (default from config)."),
    port: int = typer.Option(None, help="Override bind port (default from config)."),
) -> None:
    """Run the local web UI / API."""
    import uvicorn

    from secondbrain.query.api import create_app

    settings = get_settings()
    settings.ensure_dirs()
    init_db(settings=settings).close()
    uvicorn.run(
        create_app(settings),
        host=host or settings.api.host,
        port=port or settings.api.port,
        log_level="info",
    )


@app.command()
def status() -> None:
    """Show recording status, queue depth, and disk."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        typer.echo(json.dumps(service.status(conn, settings), indent=2))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search phrase."),
    limit: int = typer.Option(20, "--limit", "-n"),
    mode: str = typer.Option("auto", help="auto | fulltext | semantic"),
) -> None:
    """Search transcripts (full-text + semantic)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        hits = service.search(conn, query, limit, mode, settings)
    if not hits:
        typer.echo("No matches.")
        raise typer.Exit()
    for h in hits:
        when = h.get("start_at") or "?"
        snippet = h.get("snippet") or h["text"]
        typer.echo(f"{when}  (#{h['audio_file_id']} @ {h['start_offset_s']:.1f}s)\n  {snippet}\n")


@app.command()
def show(day: str = typer.Argument(None, help="YYYY-MM-DD (default: today).")) -> None:
    """Print all transcript segments for a day."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        segs = service.day_segments(conn, day)
    if not segs:
        typer.echo("Nothing recorded for that day.")
        raise typer.Exit()
    for s in segs:
        typer.echo(f"{s.get('start_at') or s['start_offset_s']}: {s['text']}")


@app.command()
def pause() -> None:
    """Pause recording (live)."""
    with db_session(settings=get_settings()) as conn:
        state.set_paused(conn, True)
    typer.echo("Recording paused.")


@app.command()
def resume() -> None:
    """Resume recording (live)."""
    with db_session(settings=get_settings()) as conn:
        state.set_paused(conn, False)
    typer.echo("Recording resumed.")


@app.command()
def drain(max_jobs: int = typer.Option(None, help="Max jobs to process.")) -> None:
    """Process pending transcription jobs now (useful for off-peak batching)."""
    from secondbrain.pipeline import worker

    settings = get_settings()
    with db_session(settings=settings) as conn:
        n = worker.drain(conn, settings=settings, max_jobs=max_jobs)
    typer.echo(f"Processed {n} job(s).")


@app.command()
def sweep() -> None:
    """Delete expired raw audio per the retention policy."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        n = retention.sweep_expired_audio(conn, settings)
    typer.echo(f"Deleted {n} expired raw-audio file(s).")


if __name__ == "__main__":
    app()
