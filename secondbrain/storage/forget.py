"""Data "forget" — purge a person, a day, or a date range, then reclaim space.

The user's right to be forgotten, enforced across every store that holds their
words: transcript segments (and their FTS index, kept in sync by triggers),
semantic search vectors, speaker profiles/observations, and the knowledge graph
nodes/edges derived from them. Raw audio files on disk are removed too once no
segment references them. ``vacuum`` reclaims the freed pages so deleted data
doesn't linger in the file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from secondbrain.config import Settings, get_settings


def _delete_segment_vectors(conn: sqlite3.Connection, seg_ids: list[int]) -> None:
    """Best-effort purge of semantic vectors (the vec0 table may not exist)."""
    if not seg_ids:
        return
    placeholders = ",".join("?" * len(seg_ids))
    try:
        conn.execute(
            f"DELETE FROM segment_vectors WHERE segment_id IN ({placeholders})", seg_ids
        )
    except sqlite3.OperationalError:
        pass


def _delete_orphan_audio(conn: sqlite3.Connection, audio_ids: list[int]) -> int:
    """Delete audio_files (and their raw file on disk) that have no segments left.

    Cascades to transcripts via ``ON DELETE CASCADE``. Returns files removed.
    """
    removed = 0
    for aid in audio_ids:
        still = conn.execute(
            "SELECT 1 FROM transcript_segments WHERE audio_file_id=? LIMIT 1", (aid,)
        ).fetchone()
        if still:
            continue
        row = conn.execute("SELECT path FROM audio_files WHERE id=?", (aid,)).fetchone()
        if row and row["path"]:
            p = Path(row["path"])
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        conn.execute("DELETE FROM audio_files WHERE id=?", (aid,))
        removed += 1
    return removed


def _purge_segments(conn: sqlite3.Connection, seg_ids: list[int]) -> dict:
    """Delete the given segments + their vectors; drop now-orphaned audio files.

    The FTS index is kept in sync by the AFTER DELETE trigger on the table.
    """
    if not seg_ids:
        return {"segments": 0, "audio_files": 0}
    audio_ids = [
        r["audio_file_id"]
        for r in conn.execute(
            f"SELECT DISTINCT audio_file_id FROM transcript_segments "
            f"WHERE id IN ({','.join('?' * len(seg_ids))})",
            seg_ids,
        ).fetchall()
    ]
    _delete_segment_vectors(conn, seg_ids)
    conn.execute(
        f"DELETE FROM transcript_segments WHERE id IN ({','.join('?' * len(seg_ids))})",
        seg_ids,
    )
    audio_removed = _delete_orphan_audio(conn, audio_ids)
    return {"segments": len(seg_ids), "audio_files": audio_removed}


def forget_day(
    conn: sqlite3.Connection, date: str, settings: Settings | None = None, *, vacuum: bool = False
) -> dict:
    """Forget everything captured on ``date`` (YYYY-MM-DD)."""
    return forget_range(conn, date, date, settings, vacuum=vacuum)


def forget_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    settings: Settings | None = None,
    *,
    vacuum: bool = False,
) -> dict:
    """Forget everything captured between ``start_date`` and ``end_date`` (inclusive)."""
    seg_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM transcript_segments "
            "WHERE substr(start_at, 1, 10) BETWEEN ? AND ?",
            (start_date, end_date),
        ).fetchall()
    ]
    result = _purge_segments(conn, seg_ids)
    if vacuum:
        _vacuum(conn)
    return result


def forget_person(
    conn: sqlite3.Connection,
    speaker_id: int,
    settings: Settings | None = None,
    *,
    vacuum: bool = False,
) -> dict:
    """Forget a person: their segments, voice profile/observations, and graph nodes.

    Includes any speakers soft-merged into this one. The owner cannot be forgotten
    this way (refuse, to avoid wiping the whole self-record by accident).
    """
    settings = settings or get_settings()
    row = conn.execute(
        "SELECT is_owner FROM speakers WHERE id=?", (speaker_id,)
    ).fetchone()
    if row is None:
        return {"segments": 0, "audio_files": 0, "speakers": 0, "kg_nodes": 0}
    if row["is_owner"]:
        raise ValueError("refusing to forget the owner; use day/range forget instead")

    ids = {speaker_id}
    for r in conn.execute(
        "SELECT id FROM speakers WHERE merged_into=?", (speaker_id,)
    ).fetchall():
        ids.add(int(r["id"]))
    id_list = list(ids)
    ph = ",".join("?" * len(id_list))

    seg_ids = [
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM transcript_segments WHERE speaker_id IN ({ph})", id_list
        ).fetchall()
    ]
    result = _purge_segments(conn, seg_ids)

    conn.execute(f"DELETE FROM speaker_observations WHERE speaker_id IN ({ph})", id_list)
    node_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM kg_nodes WHERE speaker_id IN ({ph})", id_list
    ).fetchone()["n"]
    # kg_edges + kg_aliases cascade via ON DELETE CASCADE.
    conn.execute(f"DELETE FROM kg_nodes WHERE speaker_id IN ({ph})", id_list)
    conn.execute(f"DELETE FROM speakers WHERE id IN ({ph})", id_list)

    result["speakers"] = len(id_list)
    result["kg_nodes"] = node_count
    if vacuum:
        _vacuum(conn)
    return result


def _vacuum(conn: sqlite3.Connection) -> None:
    """Reclaim freed pages. Requires autocommit (no open transaction)."""
    conn.execute("VACUUM")


def vacuum(conn: sqlite3.Connection) -> None:
    _vacuum(conn)
