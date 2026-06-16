"""Full-text search over transcript segments via FTS5 (always available)."""

from __future__ import annotations

import sqlite3

from secondbrain.storage.models import SearchHit


def _fts_query(raw: str) -> str:
    """Turn a user phrase into a safe FTS5 MATCH query.

    Each whitespace token is double-quoted (so punctuation/operators in user
    input can't break the query) and combined with implicit AND.
    """
    tokens = [t for t in raw.replace('"', " ").split() if t]
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[SearchHit]:
    match = _fts_query(query)
    rows = conn.execute(
        """
        SELECT s.id, s.audio_file_id, s.text, s.start_offset_s, s.end_offset_s,
               s.start_at,
               bm25(transcript_segments_fts) AS score,
               snippet(transcript_segments_fts, 0, '[', ']', ' … ', 12) AS snip
        FROM transcript_segments_fts
        JOIN transcript_segments s ON s.id = transcript_segments_fts.rowid
        WHERE transcript_segments_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match, limit),
    ).fetchall()
    return [
        SearchHit(
            segment_id=r["id"],
            audio_file_id=r["audio_file_id"],
            text=r["text"],
            start_offset_s=r["start_offset_s"],
            end_offset_s=r["end_offset_s"],
            start_at=r["start_at"],
            score=float(r["score"]),  # bm25: lower is better
            snippet=r["snip"] or "",
        )
        for r in rows
    ]
