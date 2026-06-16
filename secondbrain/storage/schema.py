"""Canonical SQLite schema (single source of truth).

This DDL is applied both by the Alembic initial migration (production) and
directly in tests via :func:`apply_base_schema`, so the two can never drift.

Design notes:
- Every transcript segment is traceable back to ``audio_file_id`` + a time
  offset, so we can re-transcribe with better models later without losing
  provenance. This substrate is what the Phase 3 knowledge graph builds on.
- ``transcript_segments`` carries a nullable ``speaker_id`` now so Phase 2
  diarization can populate it without a schema migration of the hot table.
- Full-text search uses an external-content FTS5 table kept in sync by triggers.
  The semantic (sqlite-vec) index is created lazily at runtime (see
  ``storage.db.try_load_sqlite_vec``) because the extension may be unavailable.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "0001_initial"

# Ordered DDL statements. Each is executed individually so this list can also be
# reused by an Alembic migration via op.execute().
STATEMENTS: list[str] = [
    # --- core capture/transcription tables -----------------------------------
    """
    CREATE TABLE IF NOT EXISTS audio_files (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        path                 TEXT NOT NULL UNIQUE,
        started_at           TEXT NOT NULL,            -- ISO-8601 UTC
        ended_at             TEXT,
        sample_rate          INTEGER NOT NULL,
        channels             INTEGER NOT NULL DEFAULT 1,
        duration_s           REAL,
        has_speech           INTEGER,                  -- VAD result: 0/1/NULL(unknown)
        status               TEXT NOT NULL DEFAULT 'recorded',
                             -- recorded | transcribing | transcribed | failed | deleted
        retention_delete_after TEXT,                   -- raw-audio auto-delete deadline
        created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audio_files_status ON audio_files(status)",
    "CREATE INDEX IF NOT EXISTS idx_audio_files_started ON audio_files(started_at)",
    """
    CREATE TABLE IF NOT EXISTS transcripts (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
        backend       TEXT NOT NULL,
        model         TEXT,
        language      TEXT,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_transcripts_audio ON transcripts(audio_file_id)",
    """
    CREATE TABLE IF NOT EXISTS transcript_segments (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id  INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
        audio_file_id  INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
        start_offset_s REAL NOT NULL,                  -- seconds from audio file start
        end_offset_s   REAL NOT NULL,
        start_at       TEXT,                            -- absolute wall-clock (UTC)
        text           TEXT NOT NULL,
        confidence     REAL,
        speaker_id     INTEGER,                         -- populated in Phase 2
        created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_segments_transcript ON transcript_segments(transcript_id)",
    "CREATE INDEX IF NOT EXISTS idx_segments_audio ON transcript_segments(audio_file_id)",
    "CREATE INDEX IF NOT EXISTS idx_segments_start_at ON transcript_segments(start_at)",
    # --- durable job queue ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        type          TEXT NOT NULL,
        payload       TEXT NOT NULL DEFAULT '{}',       -- JSON
        state         TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
        attempts      INTEGER NOT NULL DEFAULT 0,
        max_attempts  INTEGER NOT NULL DEFAULT 3,
        priority      INTEGER NOT NULL DEFAULT 0,       -- higher runs first
        error         TEXT,
        scheduled_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        started_at    TEXT,
        finished_at   TEXT,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(state, priority DESC, scheduled_at)",
    # --- speaker stub (Phase 2 fills this in) ---------------------------------
    """
    CREATE TABLE IF NOT EXISTS speakers (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT,
        is_owner    INTEGER NOT NULL DEFAULT 0,
        opted_out   INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    # --- full-text search (external-content FTS5 + sync triggers) -------------
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS transcript_segments_fts USING fts5(
        text,
        content='transcript_segments',
        content_rowid='id',
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_segments_ai AFTER INSERT ON transcript_segments BEGIN
        INSERT INTO transcript_segments_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_segments_ad AFTER DELETE ON transcript_segments BEGIN
        INSERT INTO transcript_segments_fts(transcript_segments_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_segments_au AFTER UPDATE ON transcript_segments BEGIN
        INSERT INTO transcript_segments_fts(transcript_segments_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
        INSERT INTO transcript_segments_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    # --- runtime key/value state (pause toggle, etc.) ------------------------
    """
    CREATE TABLE IF NOT EXISTS app_state (
        key        TEXT PRIMARY KEY,
        value      TEXT,
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
]


def apply_base_schema(conn: sqlite3.Connection) -> None:
    """Create all base tables/indices/triggers idempotently (non-Alembic path).

    Used directly by tests and ``init_db``. Production deployments may instead run
    ``alembic upgrade head`` (the initial migration runs the same STATEMENTS).
    Both paths stamp ``alembic_version`` so they interoperate.
    """
    for stmt in STATEMENTS:
        conn.execute(stmt)
    conn.execute("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)")
    row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO alembic_version(version_num) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
