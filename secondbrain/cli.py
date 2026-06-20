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
    sb backup                    # consistent DB snapshot (safe with WAL)
    sb export --format md        # portable transcript/graph/goals dump
    sb forget day 2026-06-16     # permanently delete a day (right to be forgotten)
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
    from secondbrain.security import auth

    settings = get_settings()
    settings.ensure_dirs()
    init_db(settings=settings).close()
    bind_host = host or settings.api.host
    # Fail closed: refuse to expose beyond loopback without auth + a password.
    if not auth.is_loopback(bind_host) and not settings.security.require_auth:
        typer.echo(
            f"Refusing to bind to non-loopback host {bind_host!r} without auth. "
            "Set [security].require_auth=true and run `sb auth set-password`, or "
            "keep host=127.0.0.1 and use `tailscale serve`."
        )
        raise typer.Exit(1)
    if settings.security.require_auth:
        with db_session(settings=settings) as conn:
            if not auth.has_password(conn):
                typer.echo("require_auth is set but no password. Run `sb auth set-password`.")
                raise typer.Exit(1)
    uvicorn.run(
        create_app(settings),
        host=bind_host,
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
def stats() -> None:
    """Show a high-level overview of the captured corpus (graph, goals, tasks)."""
    with db_session(settings=get_settings()) as conn:
        typer.echo(json.dumps(service.corpus_stats(conn), indent=2))


@app.command()
def person(
    speaker_id: int = typer.Argument(None, help="Speaker id (omit with --list)."),
    list_: bool = typer.Option(False, "--list", help="List people instead."),
) -> None:
    """Show a person dossier (identity, interactions, facts, commitments, quotes)."""
    with db_session(settings=get_settings()) as conn:
        if list_ or speaker_id is None:
            typer.echo(json.dumps(service.list_speakers(conn), indent=2))
            return
        d = service.person_dossier(conn, speaker_id, get_settings())
        if d is None:
            typer.echo(f"No such speaker: {speaker_id}")
            raise typer.Exit(1)
        typer.echo(json.dumps(d, indent=2))


@app.command()
def relationships() -> None:
    """List people you interact with, ranked (opted-out excluded)."""
    with db_session(settings=get_settings()) as conn:
        typer.echo(json.dumps(service.relationships(conn, get_settings()), indent=2))


@app.command()
def timeline(day: str = typer.Argument(None, help="YYYY-MM-DD (default: today).")) -> None:
    """Show a day as conversations with inline extracted knowledge."""
    with db_session(settings=get_settings()) as conn:
        typer.echo(json.dumps(service.timeline(conn, day, get_settings()), indent=2))


@app.command()
def queue(
    reclaim: bool = typer.Option(False, help="Re-queue jobs stuck in 'running'."),
) -> None:
    """Show job-queue counts and recent failures (optionally reclaim stuck jobs)."""
    with db_session(settings=get_settings()) as conn:
        if reclaim:
            n = service.reclaim_stale_jobs(conn)
            typer.echo(f"Reclaimed {n} stuck job(s).")
        typer.echo(json.dumps(service.queue_overview(conn), indent=2))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search phrase."),
    limit: int = typer.Option(20, "--limit", "-n"),
    mode: str = typer.Option("auto", help="auto | fulltext | semantic"),
    since: str = typer.Option(None, "--since", help="On/after this day (YYYY-MM-DD)."),
    until: str = typer.Option(None, "--until", help="On/before this day (YYYY-MM-DD)."),
) -> None:
    """Search transcripts (full-text + semantic)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        hits = service.search(conn, query, limit, mode, settings, since=since, until=until)
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


@app.command()
def backup(
    dest: str = typer.Option(None, help="Destination .db path."),
    keep: int = typer.Option(
        0, help="Prune to the newest N snapshots afterwards (0 = keep all)."
    ),
) -> None:
    """Write a consistent snapshot of the database (safe with WAL)."""
    settings = get_settings()
    path = service.backup_database(settings=settings, dest=dest)
    typer.echo(f"Database backed up to {path}")
    if keep > 0:
        removed = service.prune_backups(settings=settings, keep=keep)
        if removed:
            typer.echo(f"Pruned {removed} old snapshot(s), kept newest {keep}.")


@app.command()
def backups() -> None:
    """List available backup snapshots (newest first)."""
    rows = service.list_backups(settings=get_settings())
    if not rows:
        typer.echo("No backups found.")
        return
    for r in rows:
        mb = r["size_bytes"] / (1024 * 1024)
        typer.echo(f"{r['name']}  ({mb:.1f} MB, {r['modified'][:19]})")


@app.command()
def restore(
    src: str = typer.Argument(..., help="Path to a backup .db snapshot."),
    backup_current: bool = typer.Option(
        True, help="Snapshot the current DB before replacing it."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Replace the live database with a backup snapshot (stop the daemon first)."""
    from secondbrain.storage.backup import RestoreError

    if not yes:
        typer.confirm("This replaces the current database. Continue?", abort=True)
    try:
        path = service.restore_database(
            settings=get_settings(), src=src, backup_current=backup_current
        )
    except RestoreError as exc:
        typer.echo(f"Restore failed: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"Database restored to {path}")


@app.command()
def export(
    fmt: str = typer.Option("both", "--format", help="json | md | both."),
    out: str = typer.Option(None, "--out", help="Output directory."),
    since: str = typer.Option(None, "--since", help="Segments on/after this day (YYYY-MM-DD)."),
    until: str = typer.Option(None, "--until", help="Segments on/before this day (YYYY-MM-DD)."),
) -> None:
    """Export transcripts, graph, goals, and tasks (opted-out speakers excluded)."""
    settings = get_settings()
    out_dir = Path(out) if out else settings.data_path / "exports"
    with db_session(settings=settings) as conn:
        paths = service.export_data(
            conn, out_dir, fmt=fmt, settings=settings, since=since, until=until
        )
    for p in paths:
        typer.echo(f"Exported {p}")


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


@speaker_app.command("reassign")
def speaker_reassign(segment_id: int, speaker_id: int) -> None:
    """Correct a segment's speaker (locks it + teaches the profile)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        ok = service.reassign_segment(conn, segment_id, speaker_id, settings)
    typer.echo("Reassigned." if ok else "Segment not found.")


@speaker_app.command("reattribute")
def speaker_reattribute() -> None:
    """Re-label past low-confidence/unknown segments against improved profiles."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        n = service.reattribute(conn, settings)
    typer.echo(f"Relabeled {n} segment(s).")


@speaker_app.command("recompute")
def speaker_recompute() -> None:
    """Recompute all profile centroids from their kept exemplars."""
    with db_session(settings=get_settings()) as conn:
        n = service.recompute_profiles(conn)
    typer.echo(f"Recomputed {n} profile(s).")


@speaker_app.command("prune")
def speaker_prune() -> None:
    """Prune low-quality / excess voice exemplars."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        n = service.prune_profiles(conn, settings)
    typer.echo(f"Pruned {n} exemplar(s).")


@speaker_app.command("quality")
def speaker_quality() -> None:
    """Show speaker-profile quality metrics."""
    with db_session(settings=get_settings()) as conn:
        typer.echo(json.dumps(service.speaker_quality(conn), indent=2))


# --- proactivity + goals (Phase 4) -------------------------------------------


@app.command()
def digest(
    weekly: bool = typer.Option(False, help="Show the weekly review instead of the daily brief."),
    regenerate: bool = typer.Option(False, help="Force regeneration."),
) -> None:
    """Show today's morning brief (or weekly review)."""
    settings = get_settings()
    kind = "weekly" if weekly else "daily"
    with db_session(settings=settings) as conn:
        d = service.generate_digest(conn, settings, kind=kind, force=regenerate)
        suggestions = service.list_suggestions(conn)
    if d:
        typer.echo(d["summary_md"].strip() + "\n")
    for s in suggestions:
        typer.echo(f"  #{s['id']} [{s['kind']}] {s['title']} — {s['detail']}")


@app.command("digest-action")
def digest_action(
    suggestion_id: int,
    action: str = typer.Argument(..., help="dismiss | snooze | done | up | down"),
) -> None:
    """Act on a suggestion (dismiss/snooze/done/up/down)."""
    with db_session(settings=get_settings()) as conn:
        service.suggestion_action(conn, suggestion_id, action)
    typer.echo(f"Applied '{action}' to suggestion #{suggestion_id}.")


goals_app = typer.Typer(no_args_is_help=True, help="Manage goals.")
app.add_typer(goals_app, name="goals")


@goals_app.command("add")
def goals_add(
    title: str,
    description: str = typer.Option(None, "--description", "-d"),
    target_date: str = typer.Option(None, "--target-date"),
    priority: int = typer.Option(2, "--priority", "-p", help="1 high, 2 med, 3 low"),
) -> None:
    """Add a goal."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        gid = service.create_goal(conn, title=title, description=description,
                                  target_date=target_date, priority=priority, settings=settings)
    typer.echo(f"Added goal #{gid}.")


@goals_app.command("list")
def goals_list() -> None:
    """List goals."""
    with db_session(settings=get_settings()) as conn:
        for g in service.list_goals(conn):
            due = f" by {g['target_date']}" if g["target_date"] else ""
            typer.echo(f"#{g['id']:<4} [{g['status']}] P{g['priority']} {g['title']}{due}")


@goals_app.command("set-status")
def goals_set_status(goal_id: int, status: str) -> None:
    """Set goal status (active|paused|done|dropped)."""
    with db_session(settings=get_settings()) as conn:
        service.set_goal_status(conn, goal_id, status)
    typer.echo(f"Goal #{goal_id} → {status}.")


@goals_app.command("rm")
def goals_rm(goal_id: int) -> None:
    """Delete a goal."""
    with db_session(settings=get_settings()) as conn:
        service.delete_goal(conn, goal_id)
    typer.echo(f"Deleted goal #{goal_id}.")


# --- tasks + daily planning (Phase 6) ----------------------------------------

task_app = typer.Typer(no_args_is_help=True, help="Tasks & daily planning.")
app.add_typer(task_app, name="task")


@task_app.command("add")
def task_add(
    title: str,
    goal: int = typer.Option(None, "--goal", help="Link to goal id."),
    minutes: int = typer.Option(None, "--minutes", help="Estimate."),
    value: int = typer.Option(3, "--value"),
    effort: int = typer.Option(3, "--effort"),
    due: str = typer.Option(None, "--due", help="YYYY-MM-DD"),
) -> None:
    """Add a task."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        tid = service.create_task(conn, title=title, goal_id=goal, estimate_minutes=minutes,
                                  value=value, effort=effort, due_date=due)
    typer.echo(f"Added task #{tid}.")


@task_app.command("list")
def task_list(status: str = typer.Option(None), goal: int = typer.Option(None)) -> None:
    """List tasks."""
    with db_session(settings=get_settings()) as conn:
        for t in service.list_tasks(conn, goal_id=goal, status=status):
            due = f" due {t['due_date']}" if t["due_date"] else ""
            typer.echo(f"#{t['id']:<4} [{t['status']}] {t['title']}{due}")


@task_app.command("done")
def task_done(task_id: int) -> None:
    """Mark a task done."""
    with db_session(settings=get_settings()) as conn:
        service.task_set_status(conn, task_id, "done")
    typer.echo(f"Task #{task_id} done.")


@task_app.command("research")
def task_research(task_id: int, web: bool = typer.Option(False, "--web")) -> None:
    """Research a task (local graph-RAG by default; --web for opt-in web)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        service.task_research(conn, task_id, web=web, settings=settings)
        notes = service.task_research_notes(conn, task_id)
    if notes:
        typer.echo(notes[0]["summary_md"])


@app.command()
def plan(
    accept: bool = typer.Option(False, "--accept", help="Accept the proposed plan."),
    capacity: int = typer.Option(None, "--capacity", help="Minutes available today."),
) -> None:
    """Propose (or accept) today's prioritised plan."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        day = service.accept_day(conn) if accept else service.propose_day(
            conn, capacity_minutes=capacity, settings=settings
        )
    if not day or not day.get("tasks"):
        typer.echo("No ready tasks to plan.")
        raise typer.Exit()
    typer.echo(f"Today ({day['status']}, {day['capacity_minutes']}m):")
    for t in day["tasks"]:
        if t:
            typer.echo(f"  #{t['id']:<4} {t['title']} ({t.get('estimate_minutes') or '–'}m)")


@app.command("decompose")
def decompose(goal_id: int, accept: bool = typer.Option(False, "--accept")) -> None:
    """Propose an AI plan for a goal (use --accept to create the tasks)."""
    settings = get_settings()
    with db_session(settings=settings) as conn:
        proposal = service.decompose_goal(conn, goal_id, settings)
        for ms in proposal.get("milestones", []):
            typer.echo(f"• {ms['title']}")
            for t in ms.get("tasks", []):
                typer.echo(f"    - {t['title']} ({t.get('estimate_minutes') or '–'}m)")
        if accept:
            ids = service.accept_plan(conn, goal_id, proposal)
            typer.echo(f"\nCreated {len(ids)} task(s).")


# --- hardening: health + auth (Phase 5) --------------------------------------


@app.command()
def doctor() -> None:
    """Run health/preflight checks (config, disk, migrations, backends)."""
    from secondbrain import health

    settings = get_settings()
    with db_session(settings=settings) as conn:
        checks = health.run_checks(conn, settings)
    failed = 0
    for c in checks:
        mark = "✓" if c.ok else "✗"
        if not c.ok:
            failed += 1
        typer.echo(f"  {mark} {c.name}: {c.detail}")
    typer.echo(f"\n{'All checks passed.' if not failed else f'{failed} check(s) failed.'}")
    if failed:
        raise typer.Exit(1)


deploy_app = typer.Typer(no_args_is_help=True, help="Install always-on launchd agents (macOS).")
app.add_typer(deploy_app, name="deploy")

# Checks that must pass before installing always-on agents (others are warnings).
_DEPLOY_CRITICAL = {"migrations", "disk", "database"}


@deploy_app.command("launchd")
def deploy_launchd(
    load: bool = typer.Option(False, "--load", help="Load each agent via launchctl after writing."),
    include_menubar: bool = typer.Option(
        False, "--include-menubar", help="Also install the menu bar agent (needs the `mac` extra)."
    ),
    unload: bool = typer.Option(
        False, "--unload", help="Unload the agents via launchctl instead of installing."
    ),
) -> None:
    """Render deploy/*.plist into ~/Library/LaunchAgents with this venv's Python and
    repo path, then optionally (un)load them via launchctl."""
    from secondbrain import deploy as deploy_mod
    from secondbrain import health

    if not unload:
        # Preflight: refuse to install agents on top of a broken setup.
        settings = get_settings()
        try:
            with db_session(settings=settings) as conn:
                checks = health.run_checks(conn, settings)
        except Exception as exc:  # noqa: BLE001 - surface a clear setup hint
            typer.echo(f"Could not open the database ({exc}). Run `sb init` first.")
            raise typer.Exit(1) from exc
        critical = [c for c in checks if not c.ok and c.name in _DEPLOY_CRITICAL]
        warnings = [c for c in checks if not c.ok and c.name not in _DEPLOY_CRITICAL]
        for c in warnings:
            typer.echo(f"  ! {c.name}: {c.detail}")
        if critical:
            typer.echo("Preflight failed — fix these before installing agents:")
            for c in critical:
                typer.echo(f"  ✗ {c.name}: {c.detail}")
            raise typer.Exit(1)

    written = deploy_mod.install_launchd(load=load, include_menubar=include_menubar, unload=unload)
    if unload:
        typer.echo("Unloaded SecondBrain launchd agents.")
        return
    for p in written:
        typer.echo(f"Wrote {p}")
    if load:
        typer.echo("Loaded via launchctl. macOS will prompt for Microphone permission first run.")
    else:
        typer.echo("Re-run with --load to start them now (or `launchctl load <plist>`).")


auth_app = typer.Typer(no_args_is_help=True, help="Authentication for remote access.")
app.add_typer(auth_app, name="auth")


@auth_app.command("set-password")
def auth_set_password(
    username: str = typer.Option(None, help="Username (default from config)."),
    password: str = typer.Option(
        None, prompt=True, hide_input=True, confirmation_prompt=True,
        help="Password (prompted if omitted).",
    ),
) -> None:
    """Set the web UI username/password (stored hashed in the database)."""
    from secondbrain.security import auth

    settings = get_settings()
    user = username or settings.security.username
    with db_session(settings=settings) as conn:
        auth.set_password(conn, user, password)
    typer.echo(f"Password set for '{user}'. Set [security].require_auth=true to enforce remotely.")


@auth_app.command("status")
def auth_status() -> None:
    """Show whether auth is configured."""
    from secondbrain.security import auth

    settings = get_settings()
    with db_session(settings=settings) as conn:
        has = auth.has_password(conn)
    typer.echo(f"require_auth={settings.security.require_auth} · password_set={has}")


config_app = typer.Typer(no_args_is_help=True, help="Inspect configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Print the effective configuration as JSON (secrets redacted)."""
    from secondbrain.config import redacted_dict

    typer.echo(json.dumps(redacted_dict(get_settings()), indent=2, default=str))


@config_app.command("check")
def config_check() -> None:
    """Report any secrets found in the committed config.toml."""
    from secondbrain.config import committed_secrets

    leaked = committed_secrets()
    if leaked:
        typer.echo("Secrets in committed config.toml: " + ", ".join(leaked))
        typer.echo("Move them to config.local.toml or environment variables.")
        raise typer.Exit(1)
    typer.echo("No secrets in committed config.toml.")


forget_app = typer.Typer(no_args_is_help=True, help="Permanently delete captured data.")
app.add_typer(forget_app, name="forget")


def _confirm_forget(yes: bool) -> None:
    if not yes:
        typer.confirm("This permanently deletes data and cannot be undone. Continue?", abort=True)


@forget_app.command("day")
def forget_day(
    date: str = typer.Argument(..., help="Day to forget (YYYY-MM-DD)."),
    vacuum: bool = typer.Option(False, help="Reclaim freed disk space afterwards."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Forget everything captured on a day (transcripts, vectors, raw audio)."""
    _confirm_forget(yes)
    with db_session(settings=get_settings()) as conn:
        res = service.forget_day(conn, date, vacuum=vacuum)
    typer.echo(f"Forgot {res['segments']} segment(s), {res['audio_files']} audio file(s).")


@forget_app.command("range")
def forget_range(
    start: str = typer.Argument(..., help="Start day (YYYY-MM-DD)."),
    end: str = typer.Argument(..., help="End day, inclusive (YYYY-MM-DD)."),
    vacuum: bool = typer.Option(False, help="Reclaim freed disk space afterwards."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Forget everything captured across a date range (inclusive)."""
    _confirm_forget(yes)
    with db_session(settings=get_settings()) as conn:
        res = service.forget_range(conn, start, end, vacuum=vacuum)
    typer.echo(f"Forgot {res['segments']} segment(s), {res['audio_files']} audio file(s).")


@forget_app.command("person")
def forget_person(
    speaker_id: int = typer.Argument(..., help="Speaker id to forget."),
    vacuum: bool = typer.Option(False, help="Reclaim freed disk space afterwards."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Forget a person: their segments, voice profile, and knowledge-graph nodes."""
    _confirm_forget(yes)
    with db_session(settings=get_settings()) as conn:
        res = service.forget_person(conn, speaker_id, vacuum=vacuum)
    typer.echo(
        f"Forgot speaker {speaker_id}: {res['segments']} segment(s), "
        f"{res['speakers']} profile(s), {res['kg_nodes']} graph node(s)."
    )


if __name__ == "__main__":
    app()
