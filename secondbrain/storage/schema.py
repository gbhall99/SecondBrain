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

SCHEMA_VERSION = "0004_proactive"

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


# --- Phase 2: diarization + speakers (migration 0002_speakers) ----------------
# New tables/indices (idempotent CREATEs). Speaker embeddings are stored as
# struct-packed float32 BLOBs (centroid on `speakers`, per-observation on
# `speaker_observations`) and compared with cosine in Python — at single-user
# scale this needs no sqlite-vec/ANN index and keeps the logic CI-testable.
STATEMENTS_0002_CREATE: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at  TEXT,
        ended_at    TEXT,
        status      TEXT NOT NULL DEFAULT 'open',
                    -- open | closed | diarizing | diarized | failed
        chunk_count INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status)",
    """
    CREATE TABLE IF NOT EXISTS speaker_observations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        speaker_id      INTEGER REFERENCES speakers(id),
        conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
        audio_file_id   INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
        start_offset_s  REAL,
        end_offset_s    REAL,
        start_at        TEXT,
        confidence      REAL,
        embedding       BLOB,            -- struct-packed float32 speaker embedding
        created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_speaker_obs_speaker ON speaker_observations(speaker_id)",
    "CREATE INDEX IF NOT EXISTS idx_segments_speaker ON transcript_segments(speaker_id)",
    "CREATE INDEX IF NOT EXISTS idx_audio_conversation ON audio_files(conversation_id)",
]

# Columns added to existing tables: (table, column_name, column_def). SQLite's
# ADD COLUMN is not idempotent, so apply_base_schema guards via _safe_add_column;
# the Alembic migration runs the raw ALTERs (clean DB at revision 0001).
COLUMNS_0002: list[tuple[str, str, str]] = [
    ("speakers", "kind", "TEXT NOT NULL DEFAULT 'unknown'"),  # owner|known|unknown
    ("speakers", "display_label", "TEXT"),
    ("speakers", "exemplar_count", "INTEGER NOT NULL DEFAULT 0"),
    ("speakers", "last_seen_at", "TEXT"),
    ("speakers", "segment_count", "INTEGER NOT NULL DEFAULT 0"),
    ("speakers", "merged_into", "INTEGER"),
    ("speakers", "centroid", "BLOB"),  # struct-packed float32 profile centroid
    ("audio_files", "conversation_id", "INTEGER"),
    ("transcript_segments", "speaker_confidence", "REAL"),
]

ALTERS_0002: list[str] = [
    f"ALTER TABLE {t} ADD COLUMN {name} {ddl}" for t, name, ddl in COLUMNS_0002
]


# --- Phase 3: knowledge graph (migration 0003_knowledge) ----------------------
STATEMENTS_0003_CREATE: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS knowledge_extractions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
        model           TEXT,
        backend         TEXT,
        chunk_index     INTEGER,
        segment_id_low  INTEGER,
        segment_id_high INTEGER,
        raw_json        TEXT,
        created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kg_nodes (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        type                 TEXT NOT NULL,   -- person|project|organization|topic|place
        name                 TEXT NOT NULL,
        normalized_name      TEXT,
        display_label        TEXT,
        speaker_id           INTEGER REFERENCES speakers(id),  -- Person ↔ voice link
        embedding            BLOB,            -- struct-packed float32
        confidence           REAL,
        source_extraction_id INTEGER REFERENCES knowledge_extractions(id),
        merged_into          INTEGER,
        first_seen           TEXT,
        last_seen            TEXT,
        created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_type_name ON kg_nodes(type, normalized_name)",
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_speaker ON kg_nodes(speaker_id)",
    """
    CREATE TABLE IF NOT EXISTS kg_aliases (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id          INTEGER NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        alias            TEXT NOT NULL,
        normalized_alias TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kg_aliases_norm ON kg_aliases(normalized_alias)",
    """
    CREATE TABLE IF NOT EXISTS kg_edges (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        src_node_id          INTEGER NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        dst_node_id          INTEGER REFERENCES kg_nodes(id) ON DELETE CASCADE,
        predicate            TEXT,
        kind                 TEXT NOT NULL,   -- fact|action_item|decision|idea|mention
        object_text          TEXT,            -- literal object (date, free text)
        due_date             TEXT,
        confidence           REAL,
        source_extraction_id INTEGER REFERENCES knowledge_extractions(id),
        conversation_id      INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
        source_segment_ids   TEXT,            -- JSON array of segment ids (citations)
        superseded_by        INTEGER REFERENCES kg_edges(id),
        valid                INTEGER NOT NULL DEFAULT 1,
        first_seen           TEXT,
        last_seen            TEXT,
        created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(src_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_dst ON kg_edges(dst_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_kind_valid ON kg_edges(kind, valid)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_conversation ON kg_edges(conversation_id)",
]

COLUMNS_0003: list[tuple[str, str, str]] = [
    ("conversations", "knowledge_status", "TEXT NOT NULL DEFAULT 'pending'"),
]

ALTERS_0003: list[str] = [
    f"ALTER TABLE {t} ADD COLUMN {name} {ddl}" for t, name, ddl in COLUMNS_0003
]


# --- Phase 4: proactivity + goals (migration 0004_proactive) ------------------
STATEMENTS_0004_CREATE: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS goals (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        title            TEXT NOT NULL,
        description      TEXT,
        target_date      TEXT,
        priority         INTEGER NOT NULL DEFAULT 2,   -- 1 high, 2 med, 3 low
        status           TEXT NOT NULL DEFAULT 'active',  -- active|paused|done|dropped
        embedding        BLOB,
        last_progress_at TEXT,
        created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        updated_at       TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)",
    """
    CREATE TABLE IF NOT EXISTS goal_links (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id    INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
        kind       TEXT NOT NULL,           -- 'node' | 'edge'
        ref_id     INTEGER NOT NULL,
        relation   TEXT NOT NULL,           -- related|advances|contradicts
        score      REAL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        UNIQUE(goal_id, kind, ref_id, relation)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_goal_links_goal ON goal_links(goal_id)",
    """
    CREATE TABLE IF NOT EXISTS suggestions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        digest_date   TEXT NOT NULL,
        kind          TEXT NOT NULL,
        title         TEXT NOT NULL,
        detail        TEXT,
        payload       TEXT NOT NULL DEFAULT '{}',
        citations     TEXT NOT NULL DEFAULT '[]',
        importance    REAL NOT NULL DEFAULT 0,
        confidence    REAL NOT NULL DEFAULT 0,
        status        TEXT NOT NULL DEFAULT 'open',  -- open|dismissed|snoozed|done
        snoozed_until TEXT,
        goal_id       INTEGER REFERENCES goals(id) ON DELETE SET NULL,
        dedupe_hash   TEXT,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_suggestions_date_status ON suggestions(digest_date, status)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_suggestions_dedupe "
    "ON suggestions(dedupe_hash, digest_date)",
    """
    CREATE TABLE IF NOT EXISTS suggestion_feedback (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        suggestion_id INTEGER REFERENCES suggestions(id) ON DELETE SET NULL,
        dedupe_hash   TEXT,
        kind          TEXT,
        vote          TEXT,                  -- up|down
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_feedback_dedupe ON suggestion_feedback(dedupe_hash)",
    """
    CREATE TABLE IF NOT EXISTS digests (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        digest_date    TEXT NOT NULL,
        kind           TEXT NOT NULL DEFAULT 'daily',  -- daily|weekly
        summary_md     TEXT NOT NULL,
        suggestion_ids TEXT NOT NULL DEFAULT '[]',
        model          TEXT,
        backend        TEXT,
        created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        UNIQUE(digest_date, kind)
    )
    """,
]

COLUMNS_0004: list[tuple[str, str, str]] = []  # all-new tables; kept for parity

ALTERS_0004: list[str] = [
    f"ALTER TABLE {t} ADD COLUMN {name} {ddl}" for t, name, ddl in COLUMNS_0004
]


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any((r[1] if not isinstance(r, sqlite3.Row) else r["name"]) == column for r in rows)


def _safe_add_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if not _column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def apply_phase2_schema(conn: sqlite3.Connection) -> None:
    """Apply the 0002 additions idempotently (ADD COLUMNs then dependent creates)."""
    for table, column, ddl in COLUMNS_0002:
        _safe_add_column(conn, table, column, ddl)
    for stmt in STATEMENTS_0002_CREATE:  # some indices reference the new columns
        conn.execute(stmt)


def apply_phase3_schema(conn: sqlite3.Connection) -> None:
    """Apply the 0003 additions idempotently (ADD COLUMNs then creates)."""
    for table, column, ddl in COLUMNS_0003:
        _safe_add_column(conn, table, column, ddl)
    for stmt in STATEMENTS_0003_CREATE:
        conn.execute(stmt)


def apply_phase4_schema(conn: sqlite3.Connection) -> None:
    """Apply the 0004 additions idempotently (all-new tables)."""
    for table, column, ddl in COLUMNS_0004:
        _safe_add_column(conn, table, column, ddl)
    for stmt in STATEMENTS_0004_CREATE:
        conn.execute(stmt)


def apply_base_schema(conn: sqlite3.Connection) -> None:
    """Create all base tables/indices/triggers idempotently (non-Alembic path).

    Used directly by tests and ``init_db``. Production deployments may instead run
    ``alembic upgrade head`` (the migrations run the same STATEMENTS).
    Both paths stamp ``alembic_version`` so they interoperate.
    """
    for stmt in STATEMENTS:
        conn.execute(stmt)
    apply_phase2_schema(conn)
    apply_phase3_schema(conn)
    apply_phase4_schema(conn)
    conn.execute("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)")
    row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO alembic_version(version_num) VALUES (?)", (SCHEMA_VERSION,))
    else:
        conn.execute("UPDATE alembic_version SET version_num=?", (SCHEMA_VERSION,))
    conn.commit()
