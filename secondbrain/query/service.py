"""Backend-agnostic query helpers shared by the API, CLI, and (later) chat."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.search import combined
from secondbrain.storage import retention, state
from secondbrain.storage.models import segments_for_day


def search(conn: sqlite3.Connection, query: str, limit: int = 20, mode: str = "auto",
           settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    hits = combined.search(conn, query, limit, settings=settings, mode=mode)
    return [asdict(h) for h in hits]


def day_segments(conn: sqlite3.Connection, day: str | None = None) -> list[dict]:
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    rows = segments_for_day(conn, day)
    return [dict(r) for r in rows]


def status(conn: sqlite3.Connection, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    seg_total = conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"]
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    today_segs = len(segments_for_day(conn, today))
    paused = state.is_paused(conn, default=settings.consent.paused)
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
    }
