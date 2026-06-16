"""Lightweight dataclasses + persistence helpers over the SQLite schema.

Deliberately thin (no ORM): the schema is small and SQL keeps the FTS5/sqlite-vec
integration straightforward. These helpers centralise the SQL so callers don't
hand-write it everywhere.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Matches SQLite's strftime('%Y-%m-%dT%H:%M:%fZ','now') exactly: seconds with a
# 3-digit (millisecond) fraction. Keeping the two in lockstep is essential —
# timestamps are compared as strings (e.g. the job queue's scheduled_at <= now).
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def iso_from_dt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, ISO_FORMAT).replace(tzinfo=UTC)


def utcnow_iso() -> str:
    return iso_from_dt(datetime.now(UTC))


@dataclass
class AudioFile:
    path: str
    started_at: str
    sample_rate: int
    channels: int = 1
    ended_at: str | None = None
    duration_s: float | None = None
    has_speech: bool | None = None
    status: str = "recorded"
    retention_delete_after: str | None = None
    id: int | None = None


@dataclass
class Segment:
    transcript_id: int
    audio_file_id: int
    start_offset_s: float
    end_offset_s: float
    text: str
    start_at: str | None = None
    confidence: float | None = None
    speaker_id: int | None = None
    id: int | None = None


@dataclass
class SearchHit:
    segment_id: int
    audio_file_id: int
    text: str
    start_offset_s: float
    end_offset_s: float
    start_at: str | None
    score: float
    snippet: str = ""
    extra: dict = field(default_factory=dict)


# --- audio_files -------------------------------------------------------------


def insert_audio_file(conn: sqlite3.Connection, af: AudioFile) -> int:
    cur = conn.execute(
        """
        INSERT INTO audio_files
            (path, started_at, ended_at, sample_rate, channels, duration_s,
             has_speech, status, retention_delete_after)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            af.path,
            af.started_at,
            af.ended_at,
            af.sample_rate,
            af.channels,
            af.duration_s,
            None if af.has_speech is None else int(af.has_speech),
            af.status,
            af.retention_delete_after,
        ),
    )
    af.id = int(cur.lastrowid)
    return af.id


def set_audio_status(conn: sqlite3.Connection, audio_file_id: int, status: str) -> None:
    conn.execute("UPDATE audio_files SET status = ? WHERE id = ?", (status, audio_file_id))


def get_audio_file(conn: sqlite3.Connection, audio_file_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM audio_files WHERE id = ?", (audio_file_id,)).fetchone()


# --- transcripts + segments --------------------------------------------------


def insert_transcript(
    conn: sqlite3.Connection,
    audio_file_id: int,
    backend: str,
    model: str | None,
    language: str | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO transcripts (audio_file_id, backend, model, language) VALUES (?, ?, ?, ?)",
        (audio_file_id, backend, model, language),
    )
    return int(cur.lastrowid)


def insert_segments(conn: sqlite3.Connection, segments: list[Segment]) -> int:
    rows = [
        (
            s.transcript_id,
            s.audio_file_id,
            s.start_offset_s,
            s.end_offset_s,
            s.start_at,
            s.text,
            s.confidence,
            s.speaker_id,
        )
        for s in segments
    ]
    conn.executemany(
        """
        INSERT INTO transcript_segments
            (transcript_id, audio_file_id, start_offset_s, end_offset_s,
             start_at, text, confidence, speaker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def segments_for_day(conn: sqlite3.Connection, day: str) -> list[sqlite3.Row]:
    """All segments whose absolute start falls on ``day`` (YYYY-MM-DD), ordered."""
    return conn.execute(
        """
        SELECT * FROM transcript_segments
        WHERE substr(start_at, 1, 10) = ?
        ORDER BY start_at, start_offset_s
        """,
        (day,),
    ).fetchall()
