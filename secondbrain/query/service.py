"""Backend-agnostic query helpers shared by the API, CLI, and (later) chat."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.search import combined
from secondbrain.speaker import registry
from secondbrain.storage import retention, state
from secondbrain.storage.models import segments_for_day


def _speaker_label(row: sqlite3.Row | None) -> str | None:
    if row is None:
        return None
    return row["name"] or row["display_label"] or f"Speaker {row['id']}"


def _speaker_labels(conn: sqlite3.Connection, segment_ids: list[int]) -> dict[int, dict]:
    """Resolve speaker name + low-confidence flag for the given segment ids."""
    if not segment_ids:
        return {}
    placeholders = ",".join("?" * len(segment_ids))
    rows = conn.execute(
        f"""
        SELECT ts.id AS seg_id, ts.speaker_confidence AS conf,
               sp.id, sp.name, sp.display_label
        FROM transcript_segments ts
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE ts.id IN ({placeholders})
        """,
        segment_ids,
    ).fetchall()
    low = get_settings().diarization.low_confidence_threshold
    out: dict[int, dict] = {}
    for r in rows:
        conf = r["conf"]
        out[r["seg_id"]] = {
            "speaker": _speaker_label(r) if r["id"] is not None else None,
            "speaker_confidence": conf,
            "speaker_low_confidence": conf is not None and conf < low,
        }
    return out


def search(conn: sqlite3.Connection, query: str, limit: int = 20, mode: str = "auto",
           settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    hits = combined.search(conn, query, limit, settings=settings, mode=mode)
    results = [asdict(h) for h in hits]
    labels = _speaker_labels(conn, [h["segment_id"] for h in results])
    for h in results:
        h.update(labels.get(h["segment_id"], {}))
    return results


def day_segments(
    conn: sqlite3.Connection, day: str | None = None, settings: Settings | None = None
) -> list[dict]:
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    opted = registry.opted_out_speaker_ids(conn, settings or get_settings())
    rows = [
        dict(r) for r in segments_for_day(conn, day) if r["speaker_id"] not in opted
    ]
    labels = _speaker_labels(conn, [r["id"] for r in rows])
    for r in rows:
        r.update(labels.get(r["id"], {}))
    return rows


def status(conn: sqlite3.Connection, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    seg_total = conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"]
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    today_segs = len(segments_for_day(conn, today))
    paused = state.is_paused(conn, default=settings.consent.paused)
    speakers_known = conn.execute(
        "SELECT COUNT(*) AS n FROM speakers WHERE kind IN ('owner','known') AND merged_into IS NULL"
    ).fetchone()["n"]
    unknown_pending = conn.execute(
        "SELECT COUNT(*) AS n FROM speakers WHERE kind='unknown' AND merged_into IS NULL"
    ).fetchone()["n"]
    return {
        "recording_enabled": settings.consent.recording_enabled,
        "paused": paused,
        "recording": settings.consent.recording_enabled and not paused,
        "disk_free_gb": round(retention.free_disk_gb(settings.data_path), 2),
        "disk_ok": retention.disk_ok(settings),
        "jobs": q.counts(conn),
        "segments_total": seg_total,
        "segments_today": today_segs,
        "retention_hours": settings.consent.raw_audio_retention_hours,
        "diarization_enabled": settings.diarization.enabled,
        "speakers_known": speakers_known,
        "unknown_clusters_pending": unknown_pending,
        "proactive_enabled": settings.proactive.enabled,
        "digest_count_today": conn.execute(
            "SELECT COUNT(*) AS n FROM suggestions WHERE digest_date=? AND status='open'",
            (today,),
        ).fetchone()["n"],
    }


# --- speaker management (shared by CLI / API / web) --------------------------


def list_speakers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, display_label, kind, is_owner, opted_out, segment_count, "
        "last_seen_at FROM speakers WHERE merged_into IS NULL ORDER BY is_owner DESC, "
        "kind, segment_count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def unknown_speakers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, display_label, segment_count, last_seen_at FROM speakers "
        "WHERE kind='unknown' AND merged_into IS NULL ORDER BY segment_count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def speaker_samples(conn: sqlite3.Connection, speaker_id: int, n: int = 3) -> list[dict]:
    """Top observations (audio still on disk preferred) for clip playback."""
    rows = conn.execute(
        """
        SELECT so.id, so.audio_file_id, so.start_offset_s, so.end_offset_s, so.start_at,
               af.path, af.status AS audio_status
        FROM speaker_observations so
        JOIN audio_files af ON af.id = so.audio_file_id
        WHERE so.speaker_id = ?
        ORDER BY (af.status != 'deleted') DESC, so.confidence DESC
        LIMIT ?
        """,
        (resolve(conn, speaker_id), n),
    ).fetchall()
    return [dict(r) for r in rows]


def resolve(conn: sqlite3.Connection, speaker_id: int) -> int:
    return registry.resolve_speaker_id(conn, speaker_id)


def name_speaker(conn: sqlite3.Connection, speaker_id: int, name: str,
                 settings: Settings | None = None) -> int:
    return registry.name_speaker(conn, speaker_id, name, settings)


def merge_speakers(conn: sqlite3.Connection, src: int, dst: int,
                   settings: Settings | None = None) -> int:
    return registry.merge_speakers(conn, src, dst, settings)


def set_owner(conn: sqlite3.Connection, speaker_id: int) -> None:
    """Mark an existing (history-discovered) speaker as the owner."""
    sid = registry.resolve_speaker_id(conn, speaker_id)
    conn.execute("UPDATE speakers SET is_owner=0 WHERE is_owner=1 AND id<>?", (sid,))
    conn.execute("UPDATE speakers SET is_owner=1, kind='owner' WHERE id=?", (sid,))


# --- speaker quality / self-correction (Phase 7) -----------------------------


def reassign_segment(
    conn, segment_id: int, speaker_id: int, settings: Settings | None = None
) -> bool:
    from secondbrain.speaker import correct

    return correct.reassign_segment(conn, segment_id, speaker_id, settings or get_settings())


def reattribute(conn, settings: Settings | None = None) -> int:
    from secondbrain.speaker import reattribute as ra

    return ra.run_reattribution(conn, settings or get_settings())


def recompute_profiles(conn) -> int:
    rows = conn.execute("SELECT id FROM speakers WHERE merged_into IS NULL").fetchall()
    for r in rows:
        registry.recompute_centroid(conn, r["id"])
    return len(rows)


def prune_profiles(conn, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    rows = conn.execute("SELECT id FROM speakers WHERE merged_into IS NULL").fetchall()
    return sum(registry.prune_exemplars(conn, r["id"], settings) for r in rows)


def speaker_quality(conn, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    low = settings.diarization.low_confidence_threshold
    return {
        "speakers": conn.execute(
            "SELECT COUNT(*) AS n FROM speakers WHERE merged_into IS NULL"
        ).fetchone()["n"],
        "exemplars": conn.execute(
            "SELECT COUNT(*) AS n FROM speaker_observations WHERE pruned=0"
        ).fetchone()["n"],
        "locked_segments": conn.execute(
            "SELECT COUNT(*) AS n FROM transcript_segments WHERE speaker_locked=1"
        ).fetchone()["n"],
        "low_confidence_segments": conn.execute(
            "SELECT COUNT(*) AS n FROM transcript_segments "
            "WHERE speaker_id IS NOT NULL AND speaker_confidence < ?",
            (low,),
        ).fetchone()["n"],
    }


# --- tasks + daily planning (Phase 6) ----------------------------------------


def create_task(conn, **kw) -> int:
    from secondbrain.tasks import store

    return store.create_task(conn, **kw)


def list_tasks(conn, *, goal_id=None, status=None) -> list[dict]:
    from secondbrain.tasks import store

    return store.list_tasks(conn, goal_id=goal_id, status=status)


def task_set_status(conn, task_id: int, status: str) -> None:
    from secondbrain.tasks import store

    store.set_status(conn, task_id, status)


def promote_action_item(conn, edge_id: int, goal_id: int | None = None) -> int | None:
    from secondbrain.tasks import store

    return store.promote_action_item(conn, edge_id, goal_id)


def decompose_goal(conn, goal_id: int, settings: Settings | None = None) -> dict:
    from secondbrain.tasks import decompose

    return decompose.propose_plan(conn, goal_id, settings=settings or get_settings())


def accept_plan(conn, goal_id: int, plan: dict) -> list[int]:
    from secondbrain.tasks import decompose

    return decompose.accept_plan(conn, goal_id, plan)


def propose_day(conn, date=None, capacity_minutes=None, settings: Settings | None = None) -> dict:
    from secondbrain.tasks import planner

    return planner.propose_day(conn, date, capacity_minutes, settings or get_settings())


def accept_day(conn, date=None) -> dict | None:
    from secondbrain.tasks import planner

    return planner.accept_day(conn, date)


def get_day(conn, date=None) -> dict | None:
    from secondbrain.tasks import planner

    return planner.get_day(conn, date)


def task_research(
    conn, task_id: int, query=None, *, web=False, settings: Settings | None = None
) -> int:
    from secondbrain.tasks import research

    return research.run_research(conn, task_id, query, web=web, settings=settings or get_settings())


def task_research_notes(conn, task_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM task_research WHERE task_id=? ORDER BY created_at DESC", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# --- knowledge graph + Q&A (Phase 3) -----------------------------------------


def ask(conn: sqlite3.Connection, question: str, settings: Settings | None = None) -> dict:
    from secondbrain.knowledge import chat

    return chat.answer(conn, question, settings=settings or get_settings())


def graph_search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    from secondbrain.knowledge.graph import normalize_name

    like = f"%{normalize_name(query)}%"
    rows = conn.execute(
        """
        SELECT id, type, name, display_label,
               (SELECT COUNT(*) FROM kg_edges e
                WHERE e.valid=1 AND (e.src_node_id=kg_nodes.id OR e.dst_node_id=kg_nodes.id))
               AS edge_count
        FROM kg_nodes
        WHERE merged_into IS NULL AND normalized_name LIKE ?
        ORDER BY edge_count DESC LIMIT ?
        """,
        (like, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# --- goals + proactive (Phase 4) ---------------------------------------------


def create_goal(conn, *, title, description=None, target_date=None, priority=2,
                settings: Settings | None = None) -> int:
    from secondbrain.goals import link, store

    settings = settings or get_settings()
    gid = store.create_goal(conn, title=title, description=description,
                            target_date=target_date, priority=priority, settings=settings)
    link.relink_goal(conn, gid, settings)
    return gid


def update_goal(conn, goal_id: int, settings: Settings | None = None, **fields) -> None:
    from secondbrain.goals import link, store

    settings = settings or get_settings()
    store.update_goal(conn, goal_id, settings=settings, **fields)
    link.relink_goal(conn, goal_id, settings)


def list_goals(conn, status: str | None = None) -> list[dict]:
    from secondbrain.goals import store

    return store.list_goals(conn, status)


def get_goal(conn, goal_id: int) -> dict | None:
    from secondbrain.goals import store

    return store.get_goal(conn, goal_id)


def set_goal_status(conn, goal_id: int, status: str) -> None:
    from secondbrain.goals import store

    store.set_status(conn, goal_id, status)


def delete_goal(conn, goal_id: int) -> None:
    from secondbrain.goals import store

    store.delete_goal(conn, goal_id)


def generate_digest(conn, settings: Settings | None = None, kind: str = "daily",
                    force: bool = False, date: str | None = None) -> dict | None:
    from secondbrain.proactive import engine, store

    settings = settings or get_settings()
    d = date or _today()
    existing = store.get_digest(conn, d, kind)
    if existing and not force:
        return existing
    return engine.run_digest(conn, settings=settings, kind=kind, date=d)


def get_digest(conn, date: str | None = None, kind: str = "daily") -> dict | None:
    from secondbrain.proactive import store

    return store.get_digest(conn, date or _today(), kind)


def list_suggestions(conn, date: str | None = None, status: str = "open") -> list[dict]:
    from secondbrain.proactive import store

    return store.list_suggestions(conn, date, status)


def suggestion_action(conn, suggestion_id: int, action: str) -> None:
    from secondbrain.proactive import store

    store.suggestion_action(conn, suggestion_id, action)


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


# --- backup & export (roadmap) -----------------------------------------------


def backup_database(settings: Settings | None = None, dest=None):
    """Write a consistent snapshot of the database; returns the destination path."""
    from secondbrain.storage import backup

    return backup.backup_database(settings=settings or get_settings(), dest=dest)


def prune_backups(settings: Settings | None = None, keep: int = 10) -> int:
    """Keep the newest ``keep`` backup snapshots; delete older. Returns count removed."""
    from secondbrain.storage import backup

    return backup.prune_backups(settings=settings or get_settings(), keep=keep)


def list_backups(settings: Settings | None = None) -> list:
    """List available backup snapshots, newest first."""
    from secondbrain.storage import backup

    return backup.list_backups(settings=settings or get_settings())


def restore_database(settings: Settings | None = None, src=None, *, backup_current: bool = True):
    """Replace the live database with a snapshot; returns the restored path."""
    from secondbrain.storage import backup

    return backup.restore_database(
        settings=settings or get_settings(), src=src, backup_current=backup_current
    )


def export_data(conn, out_dir, fmt: str = "both", settings: Settings | None = None) -> list:
    """Export transcripts/graph/goals/tasks as JSON and/or Markdown. Returns paths."""
    from secondbrain.storage import backup

    settings = settings or get_settings()
    paths = []
    if fmt in ("json", "both"):
        paths.append(backup.export_json(conn, out_dir, settings))
    if fmt in ("md", "markdown", "both"):
        paths.append(backup.export_markdown(conn, out_dir, settings))
    return paths


# --- data "forget" (right to be forgotten) -----------------------------------


def forget_day(conn, date: str, settings: Settings | None = None, *, vacuum: bool = False) -> dict:
    from secondbrain.storage import forget

    return forget.forget_day(conn, date, settings or get_settings(), vacuum=vacuum)


def forget_range(conn, start_date: str, end_date: str, settings: Settings | None = None,
                 *, vacuum: bool = False) -> dict:
    from secondbrain.storage import forget

    return forget.forget_range(
        conn, start_date, end_date, settings or get_settings(), vacuum=vacuum
    )


def forget_person(conn, speaker_id: int, settings: Settings | None = None,
                  *, vacuum: bool = False) -> dict:
    from secondbrain.storage import forget

    return forget.forget_person(conn, speaker_id, settings or get_settings(), vacuum=vacuum)


def graph_node(conn: sqlite3.Connection, node_id: int) -> dict | None:
    from secondbrain.knowledge.graph import resolve_node_id

    nid = resolve_node_id(conn, node_id)
    node = conn.execute("SELECT * FROM kg_nodes WHERE id=?", (nid,)).fetchone()
    if node is None:
        return None
    edges = conn.execute(
        """
        SELECT e.id, e.predicate, e.kind, e.object_text, e.due_date, e.confidence,
               e.conversation_id, e.source_segment_ids,
               s.name AS src_name, d.name AS dst_name, d.id AS dst_id
        FROM kg_edges e
        JOIN kg_nodes s ON s.id = e.src_node_id
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        WHERE e.valid=1 AND (e.src_node_id=? OR e.dst_node_id=?)
        ORDER BY e.confidence DESC
        """,
        (nid, nid),
    ).fetchall()
    return {"node": dict(node), "edges": [dict(e) for e in edges]}
