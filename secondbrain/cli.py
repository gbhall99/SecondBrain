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
import os
from pathlib import Path

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
        spk = _speaker_prefix(h)
        loc = f"(#{h['audio_file_id']} @ {h['start_offset_s']:.1f}s)"
        typer.echo(f"{when}  {loc}\n  {spk}{snippet}\n")


@app.command()
def ask(question: str = typer.Argument(..., help="Question to answer from your data.")) -> None:
    """Ask your second brain a question (grounded in your captured knowledge)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        result = service.ask(conn, question, settings)
    typer.echo(result["answer"].strip() + "\n")
    if result["citations"]:
        typer.echo("Sources:")
        for c in result["citations"]:
            when = (c.get("start_at") or "")[:19]
            typer.echo(f"  [{c['segment_id']}] {when} — {c['speaker']}: {c['text'][:100]}")


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
        typer.echo(f"{s.get('start_at') or s['start_offset_s']}: {_speaker_prefix(s)}{s['text']}")


def _speaker_prefix(seg: dict) -> str:
    spk = seg.get("speaker")
    if not spk:
        return ""
    flag = " (?)" if seg.get("speaker_low_confidence") else ""
    return f"{spk}{flag}: "


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


# --- speaker management (Phase 2) --------------------------------------------

speaker_app = typer.Typer(no_args_is_help=True, help="Voice profiles & diarization.")
app.add_typer(speaker_app, name="speaker")


@speaker_app.command("setup")
def speaker_setup() -> None:
    """Download/authorize the pyannote diarization models (one-time)."""
    from secondbrain.pipeline.diarize import get_diarizer

    settings = get_settings()
    if not settings.diarization.hf_token and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    ):
        typer.echo(
            "No HuggingFace token found. Create a read token at "
            "https://huggingface.co/settings/tokens and accept the model terms at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "Then set it in config.local.toml ([diarization].hf_token) or HF_TOKEN env."
        )
        raise typer.Exit(1)
    try:
        get_diarizer(settings)._ensure()  # type: ignore[attr-defined]
        typer.echo("Diarization models ready.")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Setup failed: {exc}")
        raise typer.Exit(1) from exc


@speaker_app.command("enroll-owner")
def speaker_enroll_owner(
    name: str = typer.Option("Me", help="Your display name."),
    clips: int = typer.Option(3, help="Number of guided clips to record."),
    seconds: float = typer.Option(10.0, help="Seconds per clip."),
    from_files: list[str] = typer.Option(None, "--from", help="Use existing audio files instead."),
) -> None:
    """Enroll your own voice (guided recording, or --from existing clips)."""
    from secondbrain.speaker.enroll import enroll_owner_from_files, record_clip

    settings = get_settings()
    settings.ensure_dirs()
    paths: list[Path] = []
    if from_files:
        paths = [Path(f) for f in from_files]
    else:
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "I usually start my mornings reviewing the day's priorities.",
            "Recording a few sentences helps the system learn my voice.",
        ]
        for i in range(clips):
            typer.echo(f"\nClip {i + 1}/{clips} — read aloud: \"{sentences[i % len(sentences)]}\"")
            typer.confirm("Ready to record?", default=True)
            p = settings.audio_processed_dir / f"enroll_{i}.flac"
            record_clip(p, seconds, settings)
            paths.append(p)
    with db_session(settings=settings) as conn:
        owner_id = enroll_owner_from_files(conn, paths, settings=settings, name=name)
    typer.echo(f"Enrolled owner '{name}' (speaker #{owner_id}) from {len(paths)} clip(s).")


@speaker_app.command("list")
def speaker_list() -> None:
    """List known and unknown speakers."""
    with db_session(settings=get_settings()) as conn:
        for s in service.list_speakers(conn):
            label = s["name"] or s["display_label"] or f"#{s['id']}"
            owner = " (you)" if s["is_owner"] else ""
            opt = " [opted-out]" if s["opted_out"] else ""
            typer.echo(
                f"#{s['id']:<4} {label}{owner}{opt} — {s['kind']}, {s['segment_count']} segs"
            )


@speaker_app.command("unknowns")
def speaker_unknowns() -> None:
    """List unknown voices awaiting a name."""
    with db_session(settings=get_settings()) as conn:
        rows = service.unknown_speakers(conn)
    if not rows:
        typer.echo("No unknown voices. 🎉")
        raise typer.Exit()
    for s in rows:
        typer.echo(f"#{s['id']:<4} {s['display_label']} — {s['segment_count']} segs, "
                   f"last {s['last_seen_at'] or '—'}")


@speaker_app.command("name")
def speaker_name(speaker_id: int, name: str) -> None:
    """Name a speaker (relabels their history)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        redacted = service.name_speaker(conn, speaker_id, name, settings)
    msg = f"Named #{speaker_id} → '{name}'."
    if redacted:
        msg += f" Redacted {redacted} segments (opted-out)."
    typer.echo(msg)


@speaker_app.command("merge")
def speaker_merge(src: int, dst: int) -> None:
    """Merge speaker SRC into DST (relabels SRC's history)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        n = service.merge_speakers(conn, src, dst, settings)
    typer.echo(f"Merged #{src} → #{dst}; relabeled {n} segments.")


@speaker_app.command("opt-out")
def speaker_opt_out(speaker_id: int) -> None:
    """Opt a speaker out: redact their past + future words."""
    from secondbrain.speaker import registry

    with db_session(settings=get_settings()) as conn:
        n = registry.redact_speaker_segments(conn, speaker_id)
    typer.echo(f"Opted out #{speaker_id}; redacted {n} segments.")


@speaker_app.command("cluster")
def speaker_cluster() -> None:
    """Run unknown-speaker clustering now (merges recurring voices)."""
    from secondbrain.speaker import cluster

    settings = get_settings()
    with db_session(settings=settings) as conn:
        n = cluster.run_clustering(conn, settings)
    typer.echo(f"Performed {n} merge(s).")


if __name__ == "__main__":
    app()
