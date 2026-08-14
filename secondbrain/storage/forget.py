"""Data "forget" — purge a person, a day, or a date range, then reclaim space.

The user's right to be forgotten, enforced across every store that holds their
words: transcript segments (and their FTS index, kept in sync by triggers),
semantic search vectors, speaker profiles/observations, and the knowledge graph
nodes/edges derived from them. Knowledge-graph edges have the forgotten segments
removed from their citations, and any edge left ungrounded (no remaining
citation) is deleted — a forgotten statement must not survive as an asserted
fact. Raw audio files on disk are removed too once no segment references them.
``vacuum`` reclaims the freed pages so deleted data doesn't linger in the file.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.storage.db import transaction


def _local_day_utc_bounds(day: str) -> tuple[str, str]:
    """UTC ISO bounds [start, end) covering the *local* calendar day ``day``.

    Mirrors ``query.service._local_day_utc_bounds`` (storage must not import
    query): every read surface buckets days in the owner's local timezone, so
    "forget Tuesday" must mean the same Tuesday the day view shows — not the
    UTC one, which can differ by several hours at the edges.
    """
    start_local = datetime.strptime(day, "%Y-%m-%d")  # naive == system local time
    fmt = "%Y-%m-%dT%H:%M:%S"
    start = start_local.astimezone(UTC).strftime(fmt)
    end = (start_local + timedelta(days=1)).astimezone(UTC).strftime(fmt)
    return start, end


def _delete_segment_vectors(conn: sqlite3.Connection, seg_ids: list[int]) -> None:
    """Best-effort purge of semantic vectors (the vec0 table may not exist)."""
    if not seg_ids:
        return
    placeholders = ",".join("?" * len(seg_ids))
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute(
            f"DELETE FROM segment_vectors WHERE segment_id IN ({placeholders})", seg_ids
        )


def _prune_graph_citations(conn: sqlite3.Connection, seg_ids: list[int]) -> int:
    """Remove forgotten segments from edge citations; drop now-ungrounded edges.

    A knowledge-graph edge cites the transcript segment(s) it was extracted from.
    When those segments are forgotten, the citation is removed; an edge left with
    no citations is no longer grounded in anything the user retains, so it is
    deleted (a forgotten statement must not survive as an asserted fact). Returns
    the number of edges deleted.
    """
    if not seg_ids:
        return 0
    gone = set(seg_ids)
    deleted = 0
    rows = conn.execute(
        "SELECT id, source_segment_ids FROM kg_edges "
        "WHERE source_segment_ids IS NOT NULL AND source_segment_ids != '[]'"
    ).fetchall()
    for r in rows:
        try:
            cites = json.loads(r["source_segment_ids"] or "[]")
        except (TypeError, ValueError):
            continue
        kept = [c for c in cites if c not in gone]
        if len(kept) == len(cites):
            continue  # this edge didn't cite any forgotten segment
        if kept:
            conn.execute(
                "UPDATE kg_edges SET source_segment_ids=? WHERE id=?",
                (json.dumps(kept), r["id"]),
            )
        else:
            conn.execute("DELETE FROM kg_edges WHERE id=?", (r["id"],))
            deleted += 1
    return deleted


def _delete_orphan_audio(
    conn: sqlite3.Connection, audio_ids: list[int]
) -> tuple[int, set[int]]:
    """Delete audio_files (and their raw file on disk) that have no segments left.

    Cascades to transcripts and speaker_observations via ``ON DELETE CASCADE``.
    Returns (files removed, speaker ids whose observations were cascaded away) —
    the latter so callers can rebuild those voiceprints without the forgotten
    audio.
    """
    removed = 0
    affected_speakers: set[int] = set()
    for aid in audio_ids:
        still = conn.execute(
            "SELECT 1 FROM transcript_segments WHERE audio_file_id=? LIMIT 1", (aid,)
        ).fetchone()
        if still:
            continue
        affected_speakers.update(
            int(r["speaker_id"])
            for r in conn.execute(
                "SELECT DISTINCT speaker_id FROM speaker_observations "
                "WHERE audio_file_id=? AND speaker_id IS NOT NULL",
                (aid,),
            ).fetchall()
        )
        row = conn.execute("SELECT path FROM audio_files WHERE id=?", (aid,)).fetchone()
        if row and row["path"]:
            p = Path(row["path"])
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
        conn.execute("DELETE FROM audio_files WHERE id=?", (aid,))
        removed += 1
    return removed, affected_speakers


def _purge_forgotten_conversations(conn: sqlite3.Connection, conv_ids: set[int]) -> int:
    """Delete conversations that have no audio left, plus their extraction rows.

    A ``knowledge_extractions`` row carries the LLM's raw extraction JSON for a
    conversation — which echoes the transcript text. Once every chunk of a
    conversation has been forgotten, that provenance text must go too. Deleting
    the conversation cascades its extractions; first null the
    ``source_extraction_id`` back-references on any surviving nodes/edges so the
    cascade doesn't trip a foreign-key constraint. Returns conversations removed.
    """
    removed = 0
    for cid in conv_ids:
        still = conn.execute(
            "SELECT 1 FROM audio_files WHERE conversation_id=? LIMIT 1", (cid,)
        ).fetchone()
        if still:
            continue
        ext_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM knowledge_extractions WHERE conversation_id=?", (cid,)
            ).fetchall()
        ]
        if ext_ids:
            ph = ",".join("?" * len(ext_ids))
            conn.execute(
                f"UPDATE kg_nodes SET source_extraction_id=NULL "
                f"WHERE source_extraction_id IN ({ph})",
                ext_ids,
            )
            conn.execute(
                f"UPDATE kg_edges SET source_extraction_id=NULL "
                f"WHERE source_extraction_id IN ({ph})",
                ext_ids,
            )
        conn.execute("DELETE FROM conversations WHERE id=?", (cid,))  # cascades extractions
        removed += 1
    return removed


def _purge_segments(conn: sqlite3.Connection, seg_ids: list[int]) -> tuple[dict, set[int]]:
    """Delete the given segments + their vectors; drop now-orphaned audio files.

    The FTS index is kept in sync by the AFTER DELETE trigger on the table.
    Returns (result counts, speaker ids whose segments/observations were
    affected) so callers can refresh those profiles.
    """
    if not seg_ids:
        return {"segments": 0, "audio_files": 0, "kg_edges": 0, "conversations": 0}, set()
    ph = ",".join("?" * len(seg_ids))
    rows = conn.execute(
        f"SELECT DISTINCT af.id AS aid, af.conversation_id AS cid "
        f"FROM transcript_segments ts JOIN audio_files af ON af.id = ts.audio_file_id "
        f"WHERE ts.id IN ({ph})",
        seg_ids,
    ).fetchall()
    audio_ids = [r["aid"] for r in rows]
    conv_ids = {r["cid"] for r in rows if r["cid"] is not None}
    affected_speakers = {
        int(r["speaker_id"])
        for r in conn.execute(
            f"SELECT DISTINCT speaker_id FROM transcript_segments "
            f"WHERE id IN ({ph}) AND speaker_id IS NOT NULL",
            seg_ids,
        ).fetchall()
    }
    _delete_segment_vectors(conn, seg_ids)
    edges_removed = _prune_graph_citations(conn, seg_ids)
    conn.execute(f"DELETE FROM transcript_segments WHERE id IN ({ph})", seg_ids)
    audio_removed, obs_speakers = _delete_orphan_audio(conn, audio_ids)
    affected_speakers |= obs_speakers
    convs_removed = _purge_forgotten_conversations(conn, conv_ids)
    return {
        "segments": len(seg_ids),
        "audio_files": audio_removed,
        "kg_edges": edges_removed,
        "conversations": convs_removed,
    }, affected_speakers


def _refresh_speaker_profiles(conn: sqlite3.Connection, speaker_ids: set[int]) -> int:
    """Rebuild centroid/exemplar counts + segment stats for surviving speakers.

    A forgotten day/person removes observations other voiceprints were built
    from; recomputing (or clearing, when nothing remains) makes sure no profile
    still encodes forgotten audio. Speakers deleted by the purge are skipped.
    Returns the number of profiles refreshed.
    """
    from secondbrain.speaker import registry  # lazy: keep storage import-light

    refreshed = 0
    for sid in speaker_ids:
        if conn.execute("SELECT 1 FROM speakers WHERE id=?", (sid,)).fetchone() is None:
            continue
        registry.refresh_profile(conn, sid)
        registry._recount_segments(conn, sid)
        refreshed += 1
    return refreshed


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
    """Forget everything captured between ``start_date`` and ``end_date`` (inclusive).

    Dates are *local* calendar days (matching every read surface). Segments
    with no ``start_at`` of their own are pulled in via their audio file's
    capture timestamp, so forgotten-day content can't survive as undated rows.
    """
    start_utc, _ = _local_day_utc_bounds(start_date)
    _, end_utc = _local_day_utc_bounds(end_date)
    with transaction(conn):
        seg_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM transcript_segments WHERE start_at >= ? AND start_at < ?",
                (start_utc, end_utc),
            ).fetchall()
        ]
        seg_ids += [
            r["id"]
            for r in conn.execute(
                "SELECT ts.id FROM transcript_segments ts "
                "JOIN audio_files af ON af.id = ts.audio_file_id "
                "WHERE ts.start_at IS NULL AND af.started_at >= ? AND af.started_at < ?",
                (start_utc, end_utc),
            ).fetchall()
        ]
        result, affected = _purge_segments(conn, seg_ids)
        result["speakers_refreshed"] = _refresh_speaker_profiles(conn, affected)
    if vacuum:  # VACUUM cannot run inside a transaction
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
        return {
            "segments": 0, "audio_files": 0, "kg_edges": 0, "conversations": 0,
            "speakers": 0, "kg_nodes": 0,
        }
    if row["is_owner"]:
        raise ValueError("refusing to forget the owner; use day/range forget instead")

    ids = {speaker_id}
    for r in conn.execute(
        "SELECT id FROM speakers WHERE merged_into=?", (speaker_id,)
    ).fetchall():
        ids.add(int(r["id"]))
    id_list = list(ids)
    ph = ",".join("?" * len(id_list))

    with transaction(conn):
        seg_ids = [
            r["id"]
            for r in conn.execute(
                f"SELECT id FROM transcript_segments WHERE speaker_id IN ({ph})", id_list
            ).fetchall()
        ]
        result, affected = _purge_segments(conn, seg_ids)

        conn.execute(f"DELETE FROM speaker_observations WHERE speaker_id IN ({ph})", id_list)
        node_count = conn.execute(
            f"SELECT COUNT(*) AS n FROM kg_nodes WHERE speaker_id IN ({ph})", id_list
        ).fetchone()["n"]
        # kg_edges + kg_aliases cascade via ON DELETE CASCADE.
        conn.execute(f"DELETE FROM kg_nodes WHERE speaker_id IN ({ph})", id_list)
        conn.execute(f"DELETE FROM speakers WHERE id IN ({ph})", id_list)
        # OTHER speakers may have had observations on the deleted audio (shared
        # conversations) — their voiceprints must forget it too.
        result["speakers_refreshed"] = _refresh_speaker_profiles(conn, affected - ids)

    result["speakers"] = len(id_list)
    result["kg_nodes"] = node_count
    if vacuum:  # VACUUM cannot run inside a transaction
        _vacuum(conn)
    return result


def _vacuum(conn: sqlite3.Connection) -> None:
    """Reclaim freed pages. Requires autocommit (no open transaction)."""
    conn.execute("VACUUM")


def vacuum(conn: sqlite3.Connection) -> None:
    _vacuum(conn)
