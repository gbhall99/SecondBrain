"""Backend-agnostic query helpers shared by the API, CLI, and (later) chat."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.search import combined
from secondbrain.speaker import registry
from secondbrain.storage import retention, state
from secondbrain.storage.models import parse_iso, utcnow_iso


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
               sp.id, sp.name, sp.display_label, sp.kind, sp.is_owner
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
        attributed = r["id"] is not None
        kind = r["kind"] if attributed else None
        out[r["seg_id"]] = {
            "speaker": _speaker_label(r) if attributed else None,
            "speaker_confidence": conf,
            "speaker_low_confidence": conf is not None and conf < low,
            # Kind/named/owner ride along so callers can tell a real identified
            # person from an anonymous "Unknown #N" diarizer cluster: confirming
            # a guess onto a placeholder identity is not a teaching action, so
            # the day view suppresses the confirm affordance for those.
            "speaker_kind": kind,
            "speaker_is_named": attributed and kind not in (None, "unknown"),
            "speaker_is_owner": bool(attributed and r["is_owner"]),
        }
    return out


def search(conn: sqlite3.Connection, query: str, limit: int = 20, mode: str = "auto",
           settings: Settings | None = None, since: str | None = None,
           until: str | None = None, speaker: int | None = None) -> list[dict]:
    settings = settings or get_settings()
    # Date filters are *local* calendar days (YYYY-MM-DD) — the same bucketing
    # the /day view and the result "day" field use — converted to UTC bounds
    # and pushed into the search SQL itself. Filtering before LIMIT keeps a
    # filtered page exact at any corpus size (a post-filter over a capped
    # candidate pool would silently drop matches while looking exhaustive).
    since_utc = _local_day_utc_bounds(since)[0] if since else None
    until_utc = _local_day_utc_bounds(until)[1] if until else None
    # Merge-safe: a stale id (bookmarked URL from before a merge) still
    # finds the canonical voice's lines.
    sid = registry.resolve_speaker_id(conn, speaker) if speaker is not None else None
    hits = combined.search(conn, query, limit, settings=settings, mode=mode,
                           since_utc=since_utc, until_utc=until_utc, speaker_id=sid)
    results = [asdict(h) for h in hits]
    labels = _speaker_labels(conn, [h["segment_id"] for h in results])
    for h in results:
        h.update(labels.get(h["segment_id"], {}))
        # Local calendar day (same bucketing as the /day view), so clients can
        # group hits by day and link straight into that day's transcript.
        h["day"] = _local_day_of(h.get("start_at"))
    return results


def local_today() -> str:
    """Today's date (YYYY-MM-DD) in the machine's local timezone."""
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _parse_utc_ts(ts: str | None) -> datetime | None:
    """Parse a stored UTC timestamp ('…Z', with or without milliseconds)."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _local_day_of(ts: str | None) -> str | None:
    """Local calendar date (YYYY-MM-DD) a stored UTC timestamp falls on."""
    dt = _parse_utc_ts(ts)
    return dt.astimezone().strftime("%Y-%m-%d") if dt else None


def _local_day_utc_bounds(day: str) -> tuple[str, str]:
    """UTC ISO bounds [start, end) covering the *local* calendar day ``day``.

    Timestamps are stored as UTC strings; comparing against these bounds keeps
    the index usable while bucketing days the way the owner experiences them
    (a 23:30 local conversation belongs to that evening, not to tomorrow).
    """
    start_local = datetime.strptime(day, "%Y-%m-%d")  # naive == system local time
    fmt = "%Y-%m-%dT%H:%M:%S"
    start = start_local.astimezone(UTC).strftime(fmt)
    end = (start_local + timedelta(days=1)).astimezone(UTC).strftime(fmt)
    return start, end


def day_segments(
    conn: sqlite3.Connection, day: str | None = None, settings: Settings | None = None
) -> list[dict]:
    """All segments on the *local* calendar day, ordered, with speaker labels.

    Each row carries the transcript_segments columns plus ``conversation_id``
    and ``audio_status`` (via the segment's audio file) so callers can group
    lines into blocks and know whether the source audio is still playable.
    """
    day = day or local_today()
    start, end = _local_day_utc_bounds(day)
    opted = registry.opted_out_speaker_ids(conn, settings or get_settings())
    rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT ts.*, af.conversation_id AS conversation_id,
                   af.status AS audio_status
            FROM transcript_segments ts
            LEFT JOIN audio_files af ON af.id = ts.audio_file_id
            WHERE ts.start_at >= ? AND ts.start_at < ?
            ORDER BY ts.start_at, ts.start_offset_s
            """,
            (start, end),
        )
        if r["speaker_id"] not in opted
    ]
    labels = _speaker_labels(conn, [r["id"] for r in rows])
    for r in rows:
        r.update(labels.get(r["id"], {}))
    return rows


def day_segment_count(conn: sqlite3.Connection, day: str) -> int:
    """Cheap COUNT of transcript segments on a *local* calendar day."""
    start, end = _local_day_utc_bounds(day)
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM transcript_segments WHERE start_at >= ? AND start_at < ?",
        (start, end),
    ).fetchone()
    return int(row["n"]) if row else 0


def day_nav(
    conn: sqlite3.Connection, day: str, settings: Settings | None = None
) -> dict:
    """Nearest local days that actually have recorded segments, around ``day``.

    Applies the same opt-out filter as the day/timeline views themselves, so a
    "Last/Next recorded day" jump can never land on a day whose only speech is
    from opted-out speakers (which those pages would render as empty).
    Unattributed lines (speaker NULL) count as recorded speech.
    """
    start, end = _local_day_utc_bounds(day)
    opted = sorted(registry.opted_out_speaker_ids(conn, settings or get_settings()))
    visible = ""
    if opted:
        ph = ",".join("?" * len(opted))
        visible = f" AND (speaker_id IS NULL OR speaker_id NOT IN ({ph}))"
    prev = conn.execute(
        f"SELECT MAX(start_at) AS ts FROM transcript_segments WHERE start_at < ?{visible}",
        (start, *opted),
    ).fetchone()["ts"]
    nxt = conn.execute(
        f"SELECT MIN(start_at) AS ts FROM transcript_segments WHERE start_at >= ?{visible}",
        (end, *opted),
    ).fetchone()["ts"]
    return {"prev_day_with_data": _local_day_of(prev), "next_day_with_data": _local_day_of(nxt)}


def day_blocks(segments: list[dict], gap_minutes: int = 5) -> list[dict]:
    """Group a day's ordered segments into conversation blocks for display.

    Segments sharing a conversation_id stay together; where conversation ids
    are missing (still transcribing / legacy rows) a silence longer than
    ``gap_minutes`` starts a new block.
    """
    blocks: list[dict] = []
    prev_ts: datetime | None = None
    for s in segments:
        ts = _parse_utc_ts(s.get("start_at"))
        conv = s.get("conversation_id")
        block = blocks[-1] if blocks else None
        if block is not None:
            if conv is not None and block["conversation_id"] is not None:
                fresh = conv != block["conversation_id"]
            else:
                gap = (ts - prev_ts).total_seconds() if ts and prev_ts else 0.0
                fresh = gap > gap_minutes * 60
        else:
            fresh = True
        if fresh:
            block = {
                "conversation_id": conv,
                "started_at": s.get("start_at"),
                "ended_at": s.get("start_at"),
                "segments": [],
            }
            blocks.append(block)
        if block["conversation_id"] is None:
            block["conversation_id"] = conv
        block["ended_at"] = s.get("start_at") or block["ended_at"]
        block["segments"].append(s)
        prev_ts = ts or prev_ts
    return blocks


def segment_clip_info(conn: sqlite3.Connection, segment_id: int) -> dict | None:
    """Slice-source info for one transcript line's audio, or None if no segment.

    Shape matches what the clip extractor needs: the source file ``path`` and
    ``audio_status`` plus the segment's own in-file offsets. ``speaker_id``
    rides along so the API can refuse to serve opted-out voices.
    """
    row = conn.execute(
        """
        SELECT ts.id, ts.speaker_id, ts.start_offset_s, ts.end_offset_s,
               af.path, af.status AS audio_status
        FROM transcript_segments ts
        JOIN audio_files af ON af.id = ts.audio_file_id
        WHERE ts.id = ?
        """,
        (segment_id,),
    ).fetchone()
    return dict(row) if row else None


def get_segment(conn: sqlite3.Connection, segment_id: int) -> dict | None:
    """One segment's correction-relevant fields, or None if it doesn't exist."""
    row = conn.execute(
        """
        SELECT ts.id, ts.speaker_id, ts.observation_id, ts.speaker_locked,
               (o.embedding IS NOT NULL) AS has_embedding
        FROM transcript_segments ts
        LEFT JOIN speaker_observations o ON o.id = ts.observation_id
        WHERE ts.id = ?
        """,
        (segment_id,),
    ).fetchone()
    return dict(row) if row else None


def speaker_label_for(conn: sqlite3.Connection, speaker_id: int) -> str | None:
    """Display label for a (merge-resolved) speaker id; None if it doesn't exist."""
    sid = registry.resolve_speaker_id(conn, speaker_id)
    row = conn.execute(
        "SELECT id, name, display_label FROM speakers WHERE id=?", (sid,)
    ).fetchone()
    return _speaker_label(row) if row else None


def _compact_span(seconds: float) -> str:
    """Compact duration label for 'no audio for …' copy: '4m', '25h', '3d'."""
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m"
    if seconds < 48 * 3600:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _capture_freshness(conn: sqlite3.Connection, settings: Settings, *, recording: bool) -> dict:
    """Whether the recorder is *actually* producing audio, not just switched on.

    The capture loop registers an audio_files row roughly every
    ``capture.chunk_seconds`` whenever it is unpaused — even in a silent room —
    so a long gap while ``recording`` claims true means capture is broken (mic
    unplugged/missing, capture thread dead). ``last_capture_at`` is the newest
    chunk's start (indexed; rows land one chunk-length later, which the 3x
    threshold absorbs). A grace window after the latest pause→resume flip
    avoids flagging the ordinary "just resumed after a long pause" minute, and
    an empty corpus is never flagged (a fresh install isn't a failure).
    """
    row = conn.execute("SELECT MAX(started_at) AS ts FROM audio_files").fetchone()
    last_iso = row["ts"] if row else None
    out = {"last_capture_at": last_iso, "capture_stale": False, "capture_stale_for": None}
    last = _parse_utc_ts(last_iso)
    if not recording or last is None:
        return out
    now = datetime.now(UTC)
    threshold = max(3 * settings.capture.chunk_seconds, 120)
    gap = (now - last).total_seconds()
    flipped = _parse_utc_ts(state.pause_changed_at(conn))
    since_flip = (now - flipped).total_seconds() if flipped else gap
    if gap > threshold and since_flip > threshold:
        out["capture_stale"] = True
        out["capture_stale_for"] = _compact_span(gap)
    return out


def status(conn: sqlite3.Connection, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    seg_total = conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"]
    # Local day: suggestions are filed under a local digest_date, so the
    # digest_count_today lookup below must use the same calendar. (For
    # segments_today it also matches the user's wall-clock sense of "today".)
    today = local_today()
    # Same local-day bucketing as the /day view the pill links to — a 00:30
    # conversation must count toward the day the owner experienced, not the
    # UTC date prefix it happens to be stored under.
    today_segs = day_segment_count(conn, today)
    paused = state.is_paused(conn, default=settings.consent.paused)
    speakers_known = conn.execute(
        "SELECT COUNT(*) AS n FROM speakers WHERE kind IN ('owner','known') AND merged_into IS NULL"
    ).fetchone()["n"]
    unknown_pending = conn.execute(
        "SELECT COUNT(*) AS n FROM speakers WHERE kind='unknown' AND merged_into IS NULL"
    ).fetchone()["n"]
    recording = settings.consent.recording_enabled and not paused
    return {
        "recording_enabled": settings.consent.recording_enabled,
        "paused": paused,
        "recording": recording,
        # Additive capture-health fields: recording=true only says the recorder
        # is *switched on*; these say whether audio is actually arriving.
        **_capture_freshness(conn, settings, recording=recording),
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


def queue_overview(conn: sqlite3.Connection, failures: int = 10) -> dict:
    """Job-queue counts plus the most recent dead-lettered failures."""
    return {"counts": q.counts(conn), "recent_failures": q.recent_failures(conn, failures)}


def reclaim_stale_jobs(conn: sqlite3.Connection, older_than_minutes: int = 30) -> int:
    """Re-queue jobs stuck in 'running' (e.g. a worker died mid-job)."""
    return q.reclaim_stale(conn, older_than_minutes)


def corpus_stats(conn: sqlite3.Connection) -> dict:
    """A high-level overview of what the second brain has captured."""

    def _count(sql: str) -> int:
        return conn.execute(sql).fetchone()["n"]

    # Local calendar days (same bucketing as segments_today, search day groups,
    # and the /day view) — a 23:30Z capture must not read as tomorrow's date.
    span = conn.execute(
        "SELECT MIN(start_at) AS first, MAX(start_at) AS last "
        "FROM transcript_segments WHERE start_at IS NOT NULL"
    ).fetchone()
    return {
        "segments": _count("SELECT COUNT(*) AS n FROM transcript_segments"),
        "conversations": _count("SELECT COUNT(*) AS n FROM conversations"),
        "speakers": _count("SELECT COUNT(*) AS n FROM speakers WHERE merged_into IS NULL"),
        "kg_nodes": _count("SELECT COUNT(*) AS n FROM kg_nodes WHERE merged_into IS NULL"),
        "kg_edges": _count("SELECT COUNT(*) AS n FROM kg_edges WHERE valid=1"),
        "projects": _count(
            "SELECT COUNT(*) AS n FROM kg_nodes WHERE type='project' AND merged_into IS NULL"
        ),
        "goals": _count("SELECT COUNT(*) AS n FROM goals"),
        "goals_active": _count("SELECT COUNT(*) AS n FROM goals WHERE status='active'"),
        "tasks": _count("SELECT COUNT(*) AS n FROM tasks"),
        "tasks_open": _count(
            "SELECT COUNT(*) AS n FROM tasks WHERE status NOT IN ('done','dropped')"
        ),
        "first_day": _local_day_of(span["first"]),
        "last_day": _local_day_of(span["last"]),
    }


# --- speaker management (shared by CLI / API / web) --------------------------


def list_speakers(conn: sqlite3.Connection) -> list[dict]:
    # Ignored voices (dismissed as "not a person": TVs, one-off strangers) are
    # excluded everywhere a voice can be picked or listed — see ignored_speakers().
    rows = conn.execute(
        "SELECT id, name, display_label, kind, is_owner, opted_out, segment_count, "
        "last_seen_at FROM speakers WHERE merged_into IS NULL AND kind<>'ignored' "
        "ORDER BY is_owner DESC, kind, segment_count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def unknown_speakers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, display_label, segment_count, last_seen_at FROM speakers "
        "WHERE kind='unknown' AND merged_into IS NULL ORDER BY segment_count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def ignored_speakers(conn: sqlite3.Connection) -> list[dict]:
    """Voices dismissed from the unknown queue ("not a person") — restorable."""
    rows = conn.execute(
        "SELECT id, display_label, segment_count, last_seen_at FROM speakers "
        "WHERE kind='ignored' AND merged_into IS NULL "
        "ORDER BY COALESCE(last_seen_at, '') DESC, id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def speaker_overview(conn: sqlite3.Connection, speaker_id: int) -> dict | None:
    """Core row for one (merge-resolved) speaker; None when it doesn't exist."""
    sid = registry.resolve_speaker_id(conn, speaker_id)
    row = conn.execute(
        "SELECT id, name, display_label, kind, is_owner, opted_out, segment_count, "
        "last_seen_at FROM speakers WHERE id=?",
        (sid,),
    ).fetchone()
    return dict(row) if row else None


def set_speaker_ignored(conn: sqlite3.Connection, speaker_id: int, ignored: bool) -> None:
    """Flip a voice between the unknown queue and the ignored list.

    Ignored voices keep their id, centroid, and transcript segments — future
    audio from the same source still matches them silently (so a dismissed TV
    never re-enters the queue) — but they leave ``list_speakers()``, the merge
    targets, and every picker. Fully reversible.
    """
    sid = registry.resolve_speaker_id(conn, speaker_id)
    conn.execute(
        "UPDATE speakers SET kind=? WHERE id=?", ("ignored" if ignored else "unknown", sid)
    )


def is_opted_out(
    conn: sqlite3.Connection, speaker_id: int, settings: Settings | None = None
) -> bool:
    """True if the (resolved) speaker has opted out of data collection."""
    settings = settings or get_settings()
    return registry.resolve_speaker_id(conn, speaker_id) in registry.opted_out_speaker_ids(
        conn, settings
    )


# Voice clips: recognising a voice by ear needs a few seconds of continuous
# speech. Diarization exemplar windows are optimised for embeddings, not ears —
# many are sub-second blips — so clip selection prefers longer windows.
MAX_CLIP_S = 10.0  # cap served clips: enough to recognise anyone, quick to load
MIN_USABLE_CLIP_S = 1.5  # blips shorter than this rank last (near-useless by ear)


def speaker_samples(
    conn: sqlite3.Connection, speaker_id: int, n: int = 3, settings: Settings | None = None
) -> list[dict]:
    """Best playable voice clips for "who is this?" recognition.

    For each candidate observation the served window is upgraded, in order:
    the longest transcript segment attributed to the same voice that overlaps
    the exemplar (same speech turn, more of it); else — when the raw exemplar
    is a sub-``MIN_USABLE_CLIP_S`` blip — the longest segment by this voice
    anywhere in the same audio file; else the raw window. Windows are clamped
    to ``MAX_CLIP_S``. Ranking prefers audio still on disk, then usable
    length, then longer windows, with observation confidence as the tiebreak;
    identical windows (several blips inside one line) collapse into one clip.

    Returns nothing for opted-out speakers — their raw audio must never be served.
    """
    if is_opted_out(conn, speaker_id, settings):
        return []
    sid = resolve(conn, speaker_id)
    # Bounded candidate pool, deterministically ordered so a clip id listed at
    # n=3 is always found again by the clip endpoint's wider n=50 lookup.
    obs = conn.execute(
        """
        SELECT so.id, so.audio_file_id, so.start_offset_s, so.end_offset_s, so.start_at,
               so.confidence, af.path, af.status AS audio_status
        FROM speaker_observations so
        JOIN audio_files af ON af.id = so.audio_file_id
        WHERE so.speaker_id = ?
        ORDER BY (af.status != 'deleted') DESC,
                 (so.end_offset_s - so.start_offset_s) DESC,
                 so.confidence DESC, so.id DESC
        LIMIT 400
        """,
        (sid,),
    ).fetchall()
    if not obs:
        return []
    file_ids = sorted({r["audio_file_id"] for r in obs})
    marks = ",".join("?" * len(file_ids))
    segs_by_file: dict[int, list[sqlite3.Row]] = {}
    for s in conn.execute(
        "SELECT audio_file_id, start_offset_s, end_offset_s, start_at "
        f"FROM transcript_segments WHERE speaker_id = ? AND audio_file_id IN ({marks})",
        (sid, *file_ids),
    ):
        segs_by_file.setdefault(s["audio_file_id"], []).append(s)

    ranked: list[tuple] = []
    for r in obs:
        start = float(r["start_offset_s"] or 0.0)
        end = max(start, float(r["end_offset_s"] or 0.0))
        start_at = r["start_at"]
        best = None  # longest overlapping segment (same turn), else longest in file
        fallback = None
        for s in segs_by_file.get(r["audio_file_id"], ()):
            length = s["end_offset_s"] - s["start_offset_s"]
            if (
                s["end_offset_s"] > start
                and s["start_offset_s"] < end
                and (best is None or length > (best["end_offset_s"] - best["start_offset_s"]))
            ):
                best = s
            if fallback is None or length > (
                fallback["end_offset_s"] - fallback["start_offset_s"]
            ):
                fallback = s
        if best is None and end - start < MIN_USABLE_CLIP_S:
            # The exemplar is a blip too short to recognise anyone; any line
            # attributed to this voice in the same recording beats it.
            best = fallback
        if best is not None:
            start = float(best["start_offset_s"])
            end = max(start, float(best["end_offset_s"]))
            start_at = best["start_at"] or start_at
        end = min(end, start + MAX_CLIP_S)
        dur = end - start
        rank = (
            r["audio_status"] == "deleted",  # playable audio first
            dur < MIN_USABLE_CLIP_S,  # windows long enough to recognise first
            -dur,  # then the longest window
            -(r["confidence"] or 0.0),  # then the strongest voice match
            -r["id"],  # then the most recent observation
        )
        ranked.append(
            (
                rank,
                (r["audio_file_id"], round(start, 2), round(end, 2)),  # dedupe window
                {
                    "id": r["id"],
                    "audio_file_id": r["audio_file_id"],
                    "start_offset_s": start,
                    "end_offset_s": end,
                    "start_at": start_at,
                    "path": r["path"],
                    "audio_status": r["audio_status"],
                    # Additive: lets clients set expectations before pressing play.
                    "duration_s": round(dur, 2),
                },
            )
        )
    ranked.sort(key=lambda entry: entry[0])
    # Dedupe after ranking so the best-ranked observation represents a window:
    # several blips inside one spoken line become one clip, not three copies.
    out: list[dict] = []
    seen_windows: set[tuple] = set()
    for _, window, sample in ranked:
        if window in seen_windows:
            continue
        seen_windows.add(window)
        out.append(sample)
        if len(out) >= n:
            break
    return out


def resolve(conn: sqlite3.Connection, speaker_id: int) -> int:
    return registry.resolve_speaker_id(conn, speaker_id)


# --- person dossier (Phase 8A) -----------------------------------------------


def _person_connections(conn, sid: int, settings: Settings, limit: int = 8) -> list[dict]:
    """Other speakers who shared conversations with this person, ranked."""
    opted = registry.opted_out_speaker_ids(conn, settings)
    rows = conn.execute(
        """
        SELECT other.sid AS sid, COUNT(DISTINCT other.conv) AS shared
        FROM (
            SELECT af.conversation_id AS conv FROM transcript_segments ts
            JOIN audio_files af ON af.id = ts.audio_file_id
            WHERE ts.speaker_id = ? AND af.conversation_id IS NOT NULL
        ) mine
        JOIN (
            SELECT af.conversation_id AS conv, ts.speaker_id AS sid
            FROM transcript_segments ts JOIN audio_files af ON af.id = ts.audio_file_id
            WHERE af.conversation_id IS NOT NULL AND ts.speaker_id IS NOT NULL
        ) other ON other.conv = mine.conv
        WHERE other.sid != ?
        GROUP BY other.sid
        """,
        (sid, sid),
    ).fetchall()
    agg: dict[int, int] = {}
    for r in rows:
        osid = registry.resolve_speaker_id(conn, r["sid"])
        if osid == sid or osid in opted:
            continue
        agg[osid] = agg.get(osid, 0) + r["shared"]
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    if not top:
        return []
    ph = ",".join("?" * len(top))
    names = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT id, name, display_label, is_owner FROM speakers WHERE id IN ({ph})",
            [osid for osid, _ in top],
        ).fetchall()
    }
    out = []
    for osid, shared in top:
        s = names.get(osid)
        if s is None:
            continue
        if s["is_owner"]:  # a renamed owner shows their stored name everywhere
            lbl = s["name"] or "Me"
        else:
            lbl = s["name"] or s["display_label"] or f"Speaker {osid}"
        out.append({"speaker_id": osid, "label": lbl, "shared_conversations": shared})
    return out


def _local_hhmm(ts: str | None) -> str:
    """A stored UTC timestamp as local wall-clock HH:MM ('' if unparseable)."""
    dt = _parse_utc_ts(ts)
    return dt.astimezone().strftime("%H:%M") if dt else ""


def _utc_iso(dt: datetime) -> str:
    """Format an aware datetime in the storage format (UTC, trailing 'Z')."""
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def duration_label(seconds: float) -> str:
    """Human duration for overviews: 'under 1 min', '21 min', '2 h 05 min'."""
    if seconds < 60:
        return "under 1 min"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    h, m = divmod(minutes, 60)
    return f"{h} h {m:02d} min" if m else f"{h} h"


def timeline(conn: sqlite3.Connection, day: str | None = None,
             settings: Settings | None = None) -> list[dict]:
    """A *local* calendar day as a chronological list of conversations, each
    with attributed segments (opt-out filtered), start/end + duration, and the
    knowledge extracted from it. Malformed days just come back empty (the API
    and page validate upstream; the CLI shouldn't traceback on a typo)."""
    settings = settings or get_settings()
    day = day or local_today()
    try:
        start, end = _local_day_utc_bounds(day)
    except (ValueError, OverflowError, OSError):
        return []
    opted = registry.opted_out_speaker_ids(conn, settings)
    rows = conn.execute(
        """
        SELECT ts.id, ts.start_at, ts.text, ts.speaker_id,
               ts.start_offset_s, ts.end_offset_s, af.conversation_id AS conv,
               COALESCE(sp.name, sp.display_label) AS speaker,
               CASE WHEN sp.is_owner THEN 1 ELSE 0 END AS is_owner
        FROM transcript_segments ts
        JOIN audio_files af ON af.id = ts.audio_file_id
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE ts.start_at >= ? AND ts.start_at < ?
        ORDER BY ts.start_at, ts.id
        """,
        (start, end),
    ).fetchall()
    blocks: dict = {}
    order: list = []
    # Segments not yet tied to a conversation (mid-transcription / legacy rows)
    # are bucketed by silence gaps instead of all landing in one giant block.
    orphan_key: tuple | None = None
    orphan_end: datetime | None = None
    for r in rows:
        if r["speaker_id"] in opted:
            continue
        seg_start = _parse_utc_ts(r["start_at"])
        dur_s = max((r["end_offset_s"] or 0.0) - (r["start_offset_s"] or 0.0), 0.0)
        seg_end = seg_start + timedelta(seconds=dur_s) if seg_start else None
        cid = r["conv"]
        key: object = cid
        if cid is None:
            gap = ((seg_start - orphan_end).total_seconds()
                   if seg_start and orphan_end else None)
            if orphan_key is None or (gap is not None and gap > 300):
                orphan_key = ("orphan", len(order))
            key = orphan_key
            if seg_end and (orphan_end is None or seg_end > orphan_end):
                orphan_end = seg_end
        b = blocks.get(key)
        if b is None:
            b = blocks[key] = {
                "conversation_id": cid,
                "started_at": r["start_at"],
                "ended_at": r["start_at"],
                "participants": set(),
                "segments": [],
                "extractions": {},
                "_end_dt": seg_end or seg_start,
            }
            order.append(key)
        label = "Me" if r["is_owner"] else (r["speaker"] or "Unknown")
        b["participants"].add(label)
        b["segments"].append({
            "id": r["id"], "start_at": r["start_at"], "speaker": label,
            "text": r["text"], "time": _local_hhmm(r["start_at"]),
        })
        if seg_end and (b["_end_dt"] is None or seg_end > b["_end_dt"]):
            b["_end_dt"] = seg_end
    # Batch-fetch the day's extracted knowledge for all conversations at once.
    conv_ids = [c for c in order if c is not None and not isinstance(c, tuple)]
    if conv_ids:
        ph = ",".join("?" * len(conv_ids))
        for e in conn.execute(
            f"SELECT conversation_id, kind, predicate, object_text, source_segment_ids "
            f"FROM kg_edges WHERE conversation_id IN ({ph}) AND valid=1 ORDER BY kind",
            conv_ids,
        ).fetchall():
            blocks[e["conversation_id"]]["extractions"].setdefault(e["kind"], []).append({
                "predicate": e["predicate"],
                "object_text": e["object_text"],
                "segment_ids": json.loads(e["source_segment_ids"] or "[]"),
            })
    for b in blocks.values():
        b["participants"] = sorted(b["participants"])
        end_dt = b.pop("_end_dt")
        start_dt = _parse_utc_ts(b["started_at"])
        if end_dt is not None:
            b["ended_at"] = _utc_iso(end_dt)
        seconds = ((end_dt - start_dt).total_seconds()
                   if start_dt and end_dt else 0.0)
        b["duration_seconds"] = round(max(seconds, 0.0), 1)
        b["duration_minutes"] = round(b["duration_seconds"] / 60)
        b["duration_label"] = duration_label(b["duration_seconds"])
        b["segment_count"] = len(b["segments"])
        b["start_time"] = _local_hhmm(b["started_at"])
        b["end_time"] = _local_hhmm(b["ended_at"])
    return [blocks[c] for c in order]


def timeline_strip(blocks: list[dict], day: str) -> dict:
    """Presentation geometry for the timeline's day strip.

    Each conversation becomes a span positioned across the local 00:00–24:00
    axis (percentages), with greedy lane assignment so overlapping recordings
    (e.g. two capture sources) sit side by side instead of hiding each other.

    When everything recorded sits inside a narrow slice of the day (under a
    quarter of it — think evening-only days whose bars would render as
    slivers), ``zoom`` additionally carries a magnified axis over the
    surrounding whole hours: its hour labels plus ``zoom_left``/``zoom_width``
    percentages stamped onto each span. ``zoom`` is None for spread-out days.
    Keys are additive — existing consumers of lanes/spans are unaffected.
    """
    try:
        origin = datetime.strptime(day, "%Y-%m-%d").astimezone()  # local midnight
    except (ValueError, OverflowError, OSError):
        return {"lanes": 1, "spans": [], "zoom": None}
    spans: list[dict] = []
    bounds: list[tuple[datetime, datetime]] = []  # local (start, end) per span
    lane_ends: list[datetime] = []
    for i, b in enumerate(blocks, start=1):
        sd = _parse_utc_ts(b.get("started_at"))
        if sd is None:
            continue
        ed = _parse_utc_ts(b.get("ended_at")) or sd
        sd_l, ed_l = sd.astimezone(), max(ed.astimezone(), sd.astimezone())
        left = min(max((sd_l - origin).total_seconds() / 864.0, 0.0), 100.0)
        width = min((ed_l - sd_l).total_seconds() / 864.0, 100.0 - left)
        lane = next((n for n, lane_end in enumerate(lane_ends) if sd_l >= lane_end), None)
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(ed_l)
        else:
            lane_ends[lane] = ed_l
        n = b.get("segment_count") or len(b.get("segments") or [])
        label = (f"{b.get('start_time') or '?'}–{b.get('end_time') or '?'}"
                 f" · {b.get('duration_label') or ''}"
                 f" · {n} line{'s' if n != 1 else ''}"
                 f" · {', '.join(b.get('participants') or [])}")
        spans.append({"index": i, "left": round(left, 3), "width": round(width, 3),
                      "lane": lane, "label": label})
        bounds.append((sd_l, ed_l))
    return {"lanes": max(1, len(lane_ends)), "spans": spans,
            "zoom": _strip_zoom(spans, bounds, origin)}


def _strip_zoom(spans: list[dict], bounds: list[tuple[datetime, datetime]],
                origin: datetime) -> dict | None:
    """Magnified hour-window geometry when the recorded window is narrow.

    Mutates ``spans`` in place (adds ``zoom_left``/``zoom_width``) and returns
    the window metadata, or None when the day reads fine on the 24 h axis.
    """
    if not spans or not bounds:
        return None
    day_end = origin + timedelta(hours=24)
    win_start = min(max(sd, origin) for sd, _ in bounds)
    win_end = max(min(ed, day_end) for _, ed in bounds)
    if win_end < win_start:
        return None  # defensive: clamps crossed (shouldn't happen)
    if (win_end - win_start).total_seconds() >= 0.25 * 86400.0:
        return None  # active window already spans a readable share of the day
    start_h = min(int((win_start - origin).total_seconds()) // 3600, 23)
    end_h = min(-(-int((win_end - origin).total_seconds()) // 3600), 24)  # ceil
    end_h = max(end_h, start_h + 1)
    # An odd window wider than 6 h would get uneven 2-hour labels — widen it
    # by one hour (on whichever side stays inside the day) so labels divide.
    if (end_h - start_h) > 6 and (end_h - start_h) % 2:
        if start_h > 0:
            start_h -= 1
        else:
            end_h += 1
    hours = end_h - start_h
    step = 1 if hours <= 6 else 2
    zoom_start = origin + timedelta(hours=start_h)
    total = hours * 3600.0
    for sp, (sd_l, ed_l) in zip(spans, bounds, strict=True):
        left = min(max((sd_l - zoom_start).total_seconds() / total * 100.0, 0.0), 100.0)
        sp["zoom_left"] = round(left, 3)
        sp["zoom_width"] = round(min((ed_l - sd_l).total_seconds() / total * 100.0,
                                     100.0 - left), 3)
    start_pct = round(start_h / 24 * 100, 3)
    end_pct = round(end_h / 24 * 100, 3)
    return {
        "start_label": f"{start_h:02d}:00",
        "end_label": f"{end_h:02d}:00",
        "start_pct": start_pct,
        "end_pct": end_pct,
        "width_pct": round(end_pct - start_pct, 3),
        "labels": [{"pct": round((h - start_h) / hours * 100, 3), "text": f"{h:02d}:00"}
                   for h in range(start_h, end_h + 1, step)],
    }


def _friendly_day(ts: str | None, now: datetime | None = None) -> str | None:
    """Local-calendar friendly day for a stored UTC timestamp.

    'today' / 'yesterday', then 'Jun 3' (plus the year when it isn't this
    year). Uses the machine's local timezone — the same day bucketing as the
    /day view — so an evening conversation never reads as tomorrow's date.
    """
    dt = _parse_utc_ts(ts)
    if dt is None:
        return None
    local = dt.astimezone()
    today = (now or datetime.now(UTC)).astimezone().date()
    day = local.date()
    if day == today:
        return "today"
    if day == today - timedelta(days=1):
        return "yesterday"
    label = f"{local.strftime('%b')} {day.day}"
    return label if day.year == today.year else f"{label}, {day.year}"


# Ranking half-life: a conversation from ~3 months ago counts half as much as
# one from today, so current relationships outrank long-gone frequent ones.
_RANK_HALF_LIFE_DAYS = 90.0


def relationships(conn: sqlite3.Connection, settings: Settings | None = None) -> list[dict]:
    """People you interact with, ranked — plus who each of them talks with.

    Per person: all-time and last-30-days conversation counts, talk minutes,
    a friendly local last-seen label, and a top-3 ``often_with`` list built
    from co-presence (distinct conversations shared with other non-owner
    people). Owner, merged and opted-out speakers are excluded everywhere.
    Default order is conversation count decayed by recency (half-life ~90
    days); all raw fields ride along so clients can re-sort. Fields are
    additive over the original shape (CLI/menubar keep working).
    """
    settings = settings or get_settings()
    opted = registry.opted_out_speaker_ids(conn, settings)
    now = datetime.now(UTC)
    rows = conn.execute(
        """
        SELECT sp.id, sp.name, sp.display_label, sp.kind,
               COUNT(*) AS segments,
               COUNT(DISTINCT af.conversation_id) AS conversations,
               COALESCE(SUM(ts.end_offset_s - ts.start_offset_s), 0) AS talk_seconds,
               MAX(ts.start_at) AS last_seen
        FROM speakers sp
        JOIN transcript_segments ts ON ts.speaker_id = sp.id
        JOIN audio_files af ON af.id = ts.audio_file_id
        WHERE sp.is_owner=0 AND sp.merged_into IS NULL
        GROUP BY sp.id
        """
    ).fetchall()
    # Conversations in the last 30 days — the recent counterweight to the
    # all-time totals (ISO-8601 UTC strings compare lexicographically).
    cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
    recent = {
        r["sid"]: r["convos"]
        for r in conn.execute(
            """
            SELECT ts.speaker_id AS sid,
                   COUNT(DISTINCT af.conversation_id) AS convos
            FROM transcript_segments ts
            JOIN audio_files af ON af.id = ts.audio_file_id
            WHERE ts.start_at >= ? AND af.conversation_id IS NOT NULL
            GROUP BY ts.speaker_id
            """,
            (cutoff,),
        )
    }
    # Who talks with whom: distinct non-owner speaker pairs heard in the same
    # conversation, with how many conversations they share and how recently.
    pair_rows = conn.execute(
        """
        WITH sp_conv AS (
            SELECT DISTINCT ts.speaker_id AS sid, af.conversation_id AS cid
            FROM transcript_segments ts
            JOIN audio_files af ON af.id = ts.audio_file_id
            JOIN speakers sp ON sp.id = ts.speaker_id
            WHERE af.conversation_id IS NOT NULL
              AND sp.is_owner = 0 AND sp.merged_into IS NULL
        )
        SELECT a.sid AS s1, b.sid AS s2, COUNT(*) AS shared,
               MAX(c.started_at) AS last_together
        FROM sp_conv a
        JOIN sp_conv b ON b.cid = a.cid AND b.sid > a.sid
        LEFT JOIN conversations c ON c.id = a.cid
        GROUP BY a.sid, b.sid
        """
    ).fetchall()
    out = []
    for r in rows:
        if r["id"] in opted:
            continue
        days_since = None
        if r["last_seen"]:
            try:
                days_since = (now - parse_iso(r["last_seen"])).days
            except ValueError:
                days_since = None
        out.append({
            "speaker_id": r["id"],
            "label": r["name"] or r["display_label"] or f"Speaker {r['id']}",
            "kind": r["kind"],
            "conversations": r["conversations"],
            "segments": r["segments"],
            "talk_minutes": round((r["talk_seconds"] or 0) / 60.0, 1),
            "last_seen": r["last_seen"],
            "days_since_seen": days_since,
            "last_seen_label": _friendly_day(r["last_seen"], now),
            "conversations_30d": recent.get(r["id"], 0),
            "often_with": [],
        })
    by_id = {p["speaker_id"]: p for p in out}
    # Deterministic pair order: most shared first, then most recent together.
    pairs = [p for p in pair_rows if p["s1"] in by_id and p["s2"] in by_id]
    pairs.sort(key=lambda p: (p["s1"], p["s2"]))
    pairs.sort(key=lambda p: (p["shared"], p["last_together"] or ""), reverse=True)
    for p in pairs:
        for me, them in ((p["s1"], p["s2"]), (p["s2"], p["s1"])):
            mine = by_id[me]["often_with"]
            if len(mine) < 3:
                mine.append({
                    "speaker_id": them,
                    "label": by_id[them]["label"],
                    "shared": p["shared"],
                })

    def _rank(x: dict) -> tuple:
        days = x["days_since_seen"] if x["days_since_seen"] is not None else 365
        decay = 0.5 ** (max(days, 0) / _RANK_HALF_LIFE_DAYS)
        return (x["conversations"] * decay, x["talk_minutes"], x["last_seen"] or "")

    out.sort(key=_rank, reverse=True)
    return out


def _edge_provenance(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    """A kg_edge row plus provenance for deep-linking (/day?date=…#seg-…).

    Parses ``source_segment_ids`` (stored JSON) into a list of ints and adds
    ``source_seg`` (earliest cited segment id, for the anchor) and
    ``source_day`` (the *local* calendar day to open — same bucketing as the
    /day view). Falls back to the conversation's start when no segment cites
    survive. Additive: every selected column rides along untouched.
    """
    out = dict(row)
    seg_ids: list[int] = []
    raw = out.get("source_segment_ids")
    if raw:
        try:
            seg_ids = [int(x) for x in json.loads(raw)]
        except (ValueError, TypeError):
            seg_ids = []
    out["source_segment_ids"] = seg_ids
    anchor = day = None
    if seg_ids:
        ph = ",".join("?" * len(seg_ids))
        seg = conn.execute(
            f"SELECT id, start_at FROM transcript_segments WHERE id IN ({ph}) "
            "ORDER BY start_at, id LIMIT 1",
            seg_ids,
        ).fetchone()
        if seg:
            anchor, day = seg["id"], _local_day_of(seg["start_at"])
    if day is None and out.get("conversation_id") is not None:
        conv = conn.execute(
            "SELECT started_at FROM conversations WHERE id=?", (out["conversation_id"],)
        ).fetchone()
        if conv:
            day = _local_day_of(conv["started_at"])
    out["source_seg"] = anchor
    out["source_day"] = day
    return out


def _node_label(conn: sqlite3.Connection, node_id: int | None) -> str | None:
    """Display label of a kg node, following merges (None when it's gone)."""
    if node_id is None:
        return None
    nid = _resolve_node_id(conn, node_id)
    if nid is None:
        return None
    row = conn.execute("SELECT name, display_label FROM kg_nodes WHERE id=?", (nid,)).fetchone()
    return (row["display_label"] or row["name"]) if row else None


def _person_mentions(
    conn: sqlite3.Connection, node_id: int, settings: Settings, limit: int = 12
) -> list[dict]:
    """Mention/decision/idea edges that involve this person on either side.

    Facts and action items get their own dossier sections; everything else the
    knowledge graph heard about the person surfaces here. Each item carries
    provenance (``source_day``/``source_seg``) plus up to two cited transcript
    quotes (opt-out aware) so the UI can show the line that was heard and
    deep-link it into /day.
    """
    rows = conn.execute(
        """
        SELECT id, kind, predicate, object_text, due_date, confidence, conversation_id,
               source_segment_ids,
               CASE WHEN src_node_id=? THEN dst_node_id ELSE src_node_id END AS other_node_id
        FROM kg_edges
        WHERE valid=1 AND kind IN ('mention', 'decision', 'idea')
          AND (src_node_id=? OR dst_node_id=?)
        ORDER BY COALESCE(last_seen, created_at) DESC, id DESC
        LIMIT ?
        """,
        (node_id, node_id, node_id, limit),
    ).fetchall()
    out = []
    for r in rows:
        item = _edge_provenance(conn, r)
        item["other_label"] = _node_label(conn, item.pop("other_node_id"))
        item["quotes"] = _segment_quotes(conn, set(item["source_segment_ids"]), settings, limit=2)
        out.append(item)
    return out


def _person_conversations(conn: sqlite3.Connection, sid: int, limit: int = 10) -> list[dict]:
    """Conversations this speaker took part in, newest first, with anchors.

    ``day`` is the local calendar day (links straight into /day) and
    ``anchor_segment_id`` is the person's earliest line in that conversation
    (day.html renders id="seg-<id>" anchors, so the link lands on their line).
    """
    rows = conn.execute(
        """
        SELECT af.conversation_id AS conversation_id,
               COUNT(*) AS segments,
               MIN(ts.start_at) AS first_at,
               MAX(ts.start_at) AS last_at,
               COALESCE(SUM(ts.end_offset_s - ts.start_offset_s), 0) AS talk_seconds
        FROM transcript_segments ts JOIN audio_files af ON af.id = ts.audio_file_id
        WHERE ts.speaker_id = ? AND af.conversation_id IS NOT NULL
        GROUP BY af.conversation_id
        ORDER BY MIN(ts.start_at) DESC
        LIMIT ?
        """,
        (sid, limit),
    ).fetchall()
    out = []
    for r in rows:
        first_seg = conn.execute(
            "SELECT ts.id FROM transcript_segments ts "
            "JOIN audio_files af ON af.id = ts.audio_file_id "
            "WHERE af.conversation_id = ? AND ts.speaker_id = ? "
            "ORDER BY ts.start_at, ts.id LIMIT 1",
            (r["conversation_id"], sid),
        ).fetchone()
        out.append({
            "conversation_id": r["conversation_id"],
            "segments": r["segments"],
            "first_at": r["first_at"],
            "last_at": r["last_at"],
            "talk_minutes": round((r["talk_seconds"] or 0) / 60.0, 1),
            "day": _local_day_of(r["first_at"]),
            "anchor_segment_id": first_seg["id"] if first_seg else None,
        })
    return out


# Most-confident facts shown on a person page before the dossier defers to the
# knowledge graph (the "facts_total" field tells the UI when it was capped).
_DOSSIER_FACTS_LIMIT = 25


def person_dossier(
    conn: sqlite3.Connection, speaker_id: int, settings: Settings | None = None, *, quotes: int = 8
) -> dict | None:
    """Everything known about a person: identity, interactions, facts, commitments,
    recent quotes, and connections. Opt-out aware (owner exempt)."""
    settings = settings or get_settings()
    sid = registry.resolve_speaker_id(conn, speaker_id)
    spk = conn.execute(
        "SELECT id, name, display_label, kind, is_owner, opted_out, exemplar_count, "
        "segment_count, last_seen_at, centroid FROM speakers WHERE id=?",
        (sid,),
    ).fetchone()
    if spk is None:
        return None
    opted = sid in registry.opted_out_speaker_ids(conn, settings)
    # The owner's stored name wins ("Me" is only the enrollment default) so a
    # rename made here or on the People page shows up everywhere consistently.
    if spk["is_owner"]:
        label = spk["name"] or "Me"
    else:
        label = spk["name"] or spk["display_label"] or f"Speaker {sid}"
    inter = conn.execute(
        """
        SELECT COUNT(*) AS segments, MIN(ts.start_at) AS first_seen, MAX(ts.start_at) AS last_seen,
               COALESCE(SUM(ts.end_offset_s - ts.start_offset_s), 0) AS talk_seconds,
               COUNT(DISTINCT af.conversation_id) AS conversations
        FROM transcript_segments ts JOIN audio_files af ON af.id = ts.audio_file_id
        WHERE ts.speaker_id = ?
        """,
        (sid,),
    ).fetchone()
    low_conf = conn.execute(
        "SELECT COUNT(*) AS n FROM transcript_segments WHERE speaker_id=? "
        "AND speaker_confidence IS NOT NULL AND speaker_confidence < ?",
        (sid, settings.diarization.low_confidence_threshold),
    ).fetchone()["n"]
    node = conn.execute(
        "SELECT id FROM kg_nodes WHERE speaker_id=? AND type='person' AND merged_into IS NULL "
        "ORDER BY id LIMIT 1",
        (sid,),
    ).fetchone()
    node_id = int(node["id"]) if node else None

    dossier = {
        "speaker_id": sid,
        "node_id": node_id,
        "label": label,
        # Raw stored name (may be None). The rename form prefills from this —
        # never from the display label — so saving can't clobber a custom name.
        "name": spk["name"],
        "kind": spk["kind"],
        "is_owner": bool(spk["is_owner"]),
        "opted_out": opted,
        "aliases": [
            r["alias"]
            for r in conn.execute(
                "SELECT alias FROM kg_aliases WHERE node_id=?", (node_id,)
            ).fetchall()
        ] if node_id is not None else [],
        "profile": {
            "exemplar_count": spk["exemplar_count"],
            "segment_count": spk["segment_count"],
            "last_seen_at": spk["last_seen_at"],
            "voice_profile": spk["centroid"] is not None,
            "low_confidence_segments": low_conf,
        },
        "interactions": {
            "segments": inter["segments"],
            "conversations": inter["conversations"],
            "first_seen": inter["first_seen"],
            "last_seen": inter["last_seen"],
            "talk_minutes": round((inter["talk_seconds"] or 0) / 60.0, 1),
            # Human label ('under 1 min', '2 h 05 min') so the UI never has to
            # render a bare '0.0 min'. None when nothing was heard at all.
            "talk_label": duration_label(inter["talk_seconds"]) if inter["talk_seconds"] else None,
        },
        "connections": _person_connections(conn, sid, settings),
        "facts": [],
        # Totals ride along with the capped lists below so the UI can say
        # "showing X of Y" instead of truncating silently.
        "facts_total": 0,
        "commitments": {"owed_by": [], "owed_to": []},
        "mentions": [],
        "mentions_total": 0,
        "recent_conversations": [],
        "recent_quotes": [],
        # Other (unmerged) speakers carrying the same name — a naming slip
        # usually means the two voices should be merged, so the UI can hint it.
        "name_duplicates": [
            {"speaker_id": r["id"], "label": r["name"]}
            for r in conn.execute(
                "SELECT id, name FROM speakers WHERE merged_into IS NULL AND id<>? "
                "AND name IS NOT NULL AND name=? COLLATE NOCASE ORDER BY id",
                (sid, spk["name"]),
            ).fetchall()
        ]
        if spk["name"]
        else [],
    }

    # Privacy: an opted-out person gets identity/interaction shape but no content.
    if opted and not spk["is_owner"]:
        return dossier

    if node_id is not None:
        # Facts where they are the subject ("about") plus facts on other nodes
        # that name them as the object ("referenced", e.g. a project's led_by
        # edge pointing at this person) — the latter were previously invisible.
        facts = [
            {**_edge_provenance(conn, r), "direction": "about", "other_label": None}
            for r in conn.execute(
                "SELECT id, predicate, object_text, confidence, due_date, conversation_id, "
                "source_segment_ids FROM kg_edges "
                "WHERE src_node_id=? AND kind='fact' AND valid=1",
                (node_id,),
            ).fetchall()
        ]
        for r in conn.execute(
            "SELECT id, src_node_id, predicate, object_text, confidence, due_date, "
            "conversation_id, source_segment_ids FROM kg_edges "
            "WHERE dst_node_id=? AND src_node_id<>? AND kind='fact' AND valid=1",
            (node_id, node_id),
        ).fetchall():
            item = _edge_provenance(conn, r)
            item["direction"] = "referenced"
            item["other_label"] = _node_label(conn, item.pop("src_node_id"))
            facts.append(item)
        facts.sort(key=lambda f: f["confidence"] or 0, reverse=True)
        # Cap so a data-rich person stays scannable; the total lets the UI
        # point at the knowledge graph for the rest.
        dossier["facts_total"] = len(facts)
        dossier["facts"] = facts[:_DOSSIER_FACTS_LIMIT]
        # "task_id" says a commitment is already tracked in the backlog (via
        # /api/actions/{id}/promote, which is idempotent per edge). Dated items
        # come first (soonest due on top); undated ones follow — SQLite would
        # otherwise sort NULL due dates above urgent dated ones.
        dossier["commitments"] = {
            "owed_by": [
                _edge_provenance(conn, r)
                for r in conn.execute(
                    "SELECT id, object_text, due_date, confidence, conversation_id, "
                    "source_segment_ids, "
                    "(SELECT MIN(t.id) FROM tasks t WHERE t.source_edge_id=kg_edges.id) "
                    "AS task_id FROM kg_edges "
                    "WHERE src_node_id=? AND kind='action_item' AND valid=1 "
                    "ORDER BY due_date IS NULL, due_date, id",
                    (node_id,),
                ).fetchall()
            ],
            "owed_to": [
                _edge_provenance(conn, r)
                for r in conn.execute(
                    "SELECT id, object_text, due_date, confidence, conversation_id, "
                    "source_segment_ids, "
                    "(SELECT MIN(t.id) FROM tasks t WHERE t.source_edge_id=kg_edges.id) "
                    "AS task_id FROM kg_edges "
                    "WHERE dst_node_id=? AND kind='action_item' AND valid=1 "
                    "ORDER BY due_date IS NULL, due_date, id",
                    (node_id,),
                ).fetchall()
            ],
        }
        dossier["mentions"] = _person_mentions(conn, node_id, settings)
        dossier["mentions_total"] = conn.execute(
            "SELECT COUNT(*) AS n FROM kg_edges "
            "WHERE valid=1 AND kind IN ('mention', 'decision', 'idea') "
            "AND (src_node_id=? OR dst_node_id=?)",
            (node_id, node_id),
        ).fetchone()["n"]

    dossier["recent_conversations"] = _person_conversations(conn, sid)
    dossier["recent_quotes"] = [
        # "day" = local calendar day (matches /day bucketing) so the template
        # can deep-link each quote to /day?date=<day>#seg-<id>.
        {**dict(r), "day": _local_day_of(r["start_at"])}
        for r in conn.execute(
            "SELECT ts.id, ts.start_at, ts.text, af.conversation_id AS conversation_id "
            "FROM transcript_segments ts JOIN audio_files af ON af.id = ts.audio_file_id "
            "WHERE ts.speaker_id=? AND ts.text != ? "
            "ORDER BY ts.start_at DESC, ts.id DESC LIMIT ?",
            (sid, registry.REDACTED_TEXT, quotes),
        ).fetchall()
    ]
    return dossier


# --- project intelligence (Phase 9) ------------------------------------------


def _resolve_node_id(conn: sqlite3.Connection, node_id: int) -> int | None:
    """Follow the merged_into chain to the surviving node (or None if missing)."""
    nid = node_id
    for _ in range(16):
        row = conn.execute("SELECT id, merged_into FROM kg_nodes WHERE id=?", (nid,)).fetchone()
        if row is None:
            return None
        if row["merged_into"] is None:
            return int(row["id"])
        nid = row["merged_into"]
    return nid


def _project_people(conn, nid: int, settings: Settings, limit: int = 12) -> list[dict]:
    """People (person nodes) on the other end of edges touching this project."""
    opted = registry.opted_out_speaker_ids(conn, settings)
    rows = conn.execute(
        """
        SELECT n.id AS node_id, n.name, n.display_label, n.speaker_id,
               COUNT(*) AS edges
        FROM kg_edges e
        JOIN kg_nodes n
          ON n.id = CASE WHEN e.src_node_id=? THEN e.dst_node_id ELSE e.src_node_id END
        WHERE e.valid=1 AND (e.src_node_id=? OR e.dst_node_id=?)
          AND n.type='person' AND n.merged_into IS NULL
        GROUP BY n.id ORDER BY edges DESC
        """,
        (nid, nid, nid),
    ).fetchall()
    out = []
    for r in rows:
        sid = r["speaker_id"]
        if sid is not None and registry.resolve_speaker_id(conn, sid) in opted:
            continue
        out.append({
            "node_id": r["node_id"],
            "speaker_id": r["speaker_id"],
            "label": r["display_label"] or r["name"],
            "edges": r["edges"],
        })
        if len(out) >= limit:
            break
    return out


def _segment_quotes(conn, seg_ids: set[int], settings: Settings, limit: int = 8) -> list[dict]:
    """Recent non-redacted quotes from the given cited segments (opt-out aware)."""
    if not seg_ids:
        return []
    opted = registry.opted_out_speaker_ids(conn, settings)
    ph = ",".join("?" * len(seg_ids))
    rows = conn.execute(
        f"SELECT id, start_at, text, speaker_id FROM transcript_segments "
        f"WHERE id IN ({ph}) AND text != ? ORDER BY start_at DESC, id DESC",
        (*seg_ids, registry.REDACTED_TEXT),
    ).fetchall()
    speakers = _speaker_labels(conn, [r["id"] for r in rows])
    out = []
    for r in rows:
        sid = r["speaker_id"]
        if sid is not None and registry.resolve_speaker_id(conn, sid) in opted:
            continue
        out.append({
            "segment_id": r["id"],
            "start_at": r["start_at"],
            "text": r["text"],
            # Who said it (display label; None for unattributed lines).
            "speaker": speakers.get(r["id"], {}).get("speaker"),
            # Local calendar day (same bucketing as /day) so callers can
            # deep-link the quote to /day?date=<day>#seg-<segment_id>.
            "day": _local_day_of(r["start_at"]),
        })
        if len(out) >= limit:
            break
    return out


def list_projects(conn: sqlite3.Connection, settings: Settings | None = None) -> list[dict]:
    """Projects (kg nodes) ranked by activity — conversations then edge volume."""
    from secondbrain.tasks.store import DONE_STATUSES

    settings = settings or get_settings()
    rows = conn.execute(
        """
        SELECT n.id, n.name, n.display_label, n.first_seen, n.last_seen,
               COUNT(DISTINCT e.id) AS edges,
               COUNT(DISTINCT e.conversation_id) AS conversations
        FROM kg_nodes n
        LEFT JOIN kg_edges e
          ON (e.src_node_id=n.id OR e.dst_node_id=n.id) AND e.valid=1
        WHERE n.type='project' AND n.merged_into IS NULL
        GROUP BY n.id
        """
    ).fetchall()
    # Batch the per-project counts (avoid N+1): goals linked to any node
    # (DISTINCT — auto-linking plus a manual link to the same goal is still one
    # goal), and *open* action_item edges grouped by the node on either
    # endpoint. An action item is closed once the task it was promoted into
    # (tasks.source_edge_id) is done/dropped — mirrors project_dossier.
    goal_counts: dict[int, int] = {
        r["ref_id"]: r["n"]
        for r in conn.execute(
            "SELECT ref_id, COUNT(DISTINCT goal_id) AS n FROM goal_links "
            "WHERE kind='node' GROUP BY ref_id"
        ).fetchall()
    }
    done_ph = ",".join("?" * len(DONE_STATUSES))
    item_counts: dict[int, int] = {}
    for r in conn.execute(
        "SELECT node, COUNT(*) AS n FROM ("
        "  SELECT src_node_id AS node, id FROM kg_edges WHERE valid=1 AND kind='action_item'"
        "  UNION ALL"
        "  SELECT dst_node_id AS node, id FROM kg_edges "
        "  WHERE valid=1 AND kind='action_item' AND dst_node_id IS NOT NULL"
        ") x WHERE NOT ("
        f"  EXISTS (SELECT 1 FROM tasks t WHERE t.source_edge_id=x.id "
        f"          AND t.status IN ({done_ph}))"
        f"  AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.source_edge_id=x.id "
        f"                  AND t.status NOT IN ({done_ph}))"
        ") GROUP BY node",
        (*DONE_STATUSES, *DONE_STATUSES),
    ).fetchall():
        item_counts[r["node"]] = r["n"]
    out = []
    for r in rows:
        out.append({
            "node_id": r["id"],
            "label": r["display_label"] or r["name"],
            "edges": r["edges"],
            "conversations": r["conversations"],
            "linked_goals": goal_counts.get(r["id"], 0),
            "open_action_items": item_counts.get(r["id"], 0),
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
        })
    out.sort(key=lambda x: (x["conversations"], x["edges"]), reverse=True)
    return out


def project_dossier(
    conn: sqlite3.Connection, node_id: int, settings: Settings | None = None, *, quotes: int = 8
) -> dict | None:
    """Everything known about a project: identity, activity, linked goals, people,
    decisions, facts, open commitments, and recent quotes."""
    settings = settings or get_settings()
    nid = _resolve_node_id(conn, node_id)
    if nid is None:
        return None
    node = conn.execute(
        "SELECT id, type, name, display_label, first_seen, last_seen FROM kg_nodes WHERE id=?",
        (nid,),
    ).fetchone()
    if node is None or node["type"] != "project":
        return None

    edges = conn.execute(
        "SELECT e.id, e.src_node_id, e.dst_node_id, e.predicate, e.kind, e.object_text, "
        "e.due_date, e.confidence, e.conversation_id, e.source_segment_ids, e.last_seen, "
        "s.name AS src_name, s.display_label AS src_display, s.speaker_id AS src_speaker_id, "
        "d.name AS dst_name, d.display_label AS dst_display "
        "FROM kg_edges e "
        "LEFT JOIN kg_nodes s ON s.id = e.src_node_id "
        "LEFT JOIN kg_nodes d ON d.id = e.dst_node_id "
        "WHERE e.valid=1 AND (e.src_node_id=? OR e.dst_node_id=?)",
        (nid, nid),
    ).fetchall()

    from secondbrain.knowledge.graph import normalize_name

    aliases = [
        r["alias"]
        for r in conn.execute("SELECT alias FROM kg_aliases WHERE node_id=?", (nid,)).fetchall()
    ]
    # Every name this project answers to, normalized — used to spot facts whose
    # object_text merely repeats the project's own name ("Dana works_on 'Atlas'"
    # on the Atlas page) so the template can say "this project" instead.
    self_names = {
        normalize_name(x)
        for x in [node["name"], node["display_label"], *aliases]
        if x
    }
    # Consistent privacy: People and quotes already hide opted-out speakers, so
    # the subject labels we now expose on facts/items must not re-leak them.
    opted = registry.opted_out_speaker_ids(conn, settings)

    facts: list[dict] = []
    decisions: list[dict] = []
    action_items: list[dict] = []
    convos: set[int] = set()
    seg_ids: set[int] = set()
    for e in edges:
        if e["conversation_id"] is not None:
            convos.add(e["conversation_id"])
        # Provenance for deep links: earliest cited segment + its local /day
        # date (tolerates malformed stored JSON, falls back to the conversation).
        prov = _edge_provenance(conn, e)
        seg_ids.update(prov["source_segment_ids"])
        src_sid = e["src_speaker_id"]
        src_hidden = bool(
            src_sid is not None and registry.resolve_speaker_id(conn, src_sid) in opted
        )
        item = {
            "edge_id": e["id"],
            "predicate": e["predicate"],
            "object_text": e["object_text"],
            "due_date": e["due_date"],
            "confidence": e["confidence"],
            "segment_ids": prov["source_segment_ids"],
            "source_seg": prov["source_seg"],
            "source_day": prov["source_day"],
            "last_seen": e["last_seen"],
            # Both endpoints of the edge, labelled, so the UI can name the
            # subject ("Dana — works on …") instead of dropping it. Additive.
            "src_node_id": e["src_node_id"],
            "dst_node_id": e["dst_node_id"],
            "src_label": None if src_hidden else (e["src_display"] or e["src_name"]),
            "src_speaker_id": None if src_hidden else src_sid,
            "src_hidden": src_hidden,
            "dst_label": (e["dst_display"] or e["dst_name"])
            if e["dst_node_id"] is not None
            else None,
            # True when the object side is just this project again (dst is the
            # project and object_text repeats one of its names) — the card can
            # say "this project" rather than echoing the page title.
            "object_redundant": bool(
                e["dst_node_id"] == nid
                and normalize_name(e["object_text"] or "") in (self_names | {""})
            ),
        }
        if e["kind"] == "fact":
            facts.append(item)
        elif e["kind"] == "decision":
            decisions.append(item)
        elif e["kind"] == "action_item":
            action_items.append(item)
    facts.sort(key=lambda x: x["confidence"] or 0, reverse=True)
    # The same statement heard in several conversations lands as several edges;
    # show it once (the highest-confidence copy — the list is already sorted).
    # activity.edges still counts every mention.
    seen_facts: set[tuple] = set()
    unique_facts: list[dict] = []
    for f in facts:
        key = (
            f["src_node_id"],
            f["dst_node_id"],
            (f["predicate"] or "").strip().lower(),
            (f["object_text"] or "").strip().lower(),
        )
        if key in seen_facts:
            continue
        seen_facts.add(key)
        unique_facts.append(f)
    facts = unique_facts
    decisions.sort(key=lambda x: x["last_seen"] or "", reverse=True)

    # Action-item lifecycle: join the task each item was promoted into (via
    # /api/actions/{edge_id}/promote → tasks.source_edge_id). "done" means the
    # only tasks derived from it are done/dropped; a still-active task keeps
    # the commitment open ("tracked"), no task at all means promotable.
    from secondbrain.tasks.store import DONE_STATUSES

    tasks_by_edge: dict[int, list[sqlite3.Row]] = {}
    if action_items:
        ph = ",".join("?" * len(action_items))
        for t in conn.execute(
            f"SELECT id, source_edge_id, status FROM tasks "
            f"WHERE source_edge_id IN ({ph}) ORDER BY id",
            [it["edge_id"] for it in action_items],
        ).fetchall():
            tasks_by_edge.setdefault(t["source_edge_id"], []).append(t)
    today = local_today()
    for it in action_items:
        linked = tasks_by_edge.get(it["edge_id"], [])
        active = next((t for t in linked if t["status"] not in DONE_STATUSES), None)
        closed = next((t for t in linked if t["status"] in DONE_STATUSES), None)
        chosen = active or closed
        it["task_id"] = chosen["id"] if chosen else None
        it["task_status"] = chosen["status"] if chosen else None
        it["done"] = active is None and closed is not None
        it["overdue"] = bool(
            not it["done"] and it["due_date"] and str(it["due_date"])[:10] < today
        )
    # Open items first (soonest due date up top, undated after), done ones last.
    action_items.sort(
        key=lambda x: (x["done"], x["due_date"] is None, x["due_date"] or "", -x["edge_id"])
    )

    # One entry per goal even when several links point at it (auto-linking plus
    # a manual link); keep the most meaningful relation for display.
    relation_rank = {"advances": 0, "blocks": 1, "related": 2}
    goals_by_id: dict[int, dict] = {}
    for r in conn.execute(
        "SELECT g.id, g.title, g.status, g.priority, gl.relation FROM goal_links gl "
        "JOIN goals g ON g.id=gl.goal_id WHERE gl.kind='node' AND gl.ref_id=? "
        "ORDER BY g.priority, g.id",
        (nid,),
    ).fetchall():
        row = dict(r)
        seen = goals_by_id.get(row["id"])
        if seen is None or (
            relation_rank.get(row["relation"], 9) < relation_rank.get(seen["relation"], 9)
        ):
            goals_by_id[row["id"]] = row
    goals = list(goals_by_id.values())

    # The SQL behind _segment_quotes always scans every cited segment (the cap
    # is applied while collecting), so fetching all survivors to learn the true
    # total costs nothing extra — the page can then say "8 of 23" honestly.
    all_quotes = _segment_quotes(conn, seg_ids, settings, limit=max(len(seg_ids), 1))
    return {
        "node_id": nid,
        "label": node["display_label"] or node["name"],
        "type": node["type"],
        "aliases": aliases,
        "activity": {
            "edges": len(edges),
            "conversations": len(convos),
            "first_seen": node["first_seen"],
            "last_seen": node["last_seen"],
        },
        "linked_goals": goals,
        "people": _project_people(conn, nid, settings),
        "decisions": decisions,
        "facts": facts,
        "open_commitments": action_items,
        "recent_quotes": all_quotes[:quotes],
        # Additive: how many quotable lines exist in total (recent_quotes is
        # capped at `quotes`), so clients can show "the N most recent of M".
        "quotes_total": len(all_quotes),
    }


def name_speaker(conn: sqlite3.Connection, speaker_id: int, name: str,
                 settings: Settings | None = None) -> int:
    return registry.name_speaker(conn, speaker_id, name, settings)


def merge_speakers(conn: sqlite3.Connection, src: int, dst: int,
                   settings: Settings | None = None) -> int:
    return registry.merge_speakers(conn, src, dst, settings)


# Only the most recent web-app merge is undoable, for a short window. The
# snapshot (which segment/observation rows moved) lives in app_state so it
# survives a server restart without any schema change.
MERGE_UNDO_KEY = "speakers_merge_undo"
MERGE_UNDO_WINDOW_S = 15 * 60


def merge_speakers_undoable(conn: sqlite3.Connection, src: int, dst: int,
                            settings: Settings | None = None) -> tuple[int, bool]:
    """Merge ``src`` into ``dst`` and stash a one-shot undo snapshot.

    Returns ``(relabeled_segments, undo_available)``. No snapshot is kept when
    ``dst`` is opted out: the merge redacts the moved lines' text, which an
    undo could not bring back. Any previous snapshot is replaced either way —
    only the very last merge is ever undoable.
    """
    settings = settings or get_settings()
    src_r = registry.resolve_speaker_id(conn, src)
    dst_r = registry.resolve_speaker_id(conn, dst)
    snap = None
    if not registry.is_opted_out(conn, dst_r, settings):
        snap = {
            "src": src_r,
            "dst": dst_r,
            "src_label": speaker_label_for(conn, src_r),
            "dst_label": speaker_label_for(conn, dst_r),
            "segment_ids": [
                int(r["id"]) for r in conn.execute(
                    "SELECT id FROM transcript_segments WHERE speaker_id=?", (src_r,)
                ).fetchall()
            ],
            "observation_ids": [
                int(r["id"]) for r in conn.execute(
                    "SELECT id FROM speaker_observations WHERE speaker_id=?", (src_r,)
                ).fetchall()
            ],
            "at": utcnow_iso(),
        }
        # merge_speakers hands src's name to an unnamed dst (so "Alice" survives
        # a merge into "Unknown #2"). Remember dst's pre-merge identity so undo
        # can put it back instead of leaving a phantom second "Alice".
        src_row = conn.execute("SELECT name FROM speakers WHERE id=?", (src_r,)).fetchone()
        dst_row = conn.execute(
            "SELECT name, display_label, kind FROM speakers WHERE id=?", (dst_r,)
        ).fetchone()
        if (
            src_row is not None and dst_row is not None
            and (src_row["name"] or "").strip() and not (dst_row["name"] or "").strip()
        ):
            snap["adopted_name"] = src_row["name"]
            snap["dst_prior"] = {
                "name": dst_row["name"],
                "display_label": dst_row["display_label"],
                "kind": dst_row["kind"],
            }
    n = registry.merge_speakers(conn, src, dst, settings)
    if snap is not None:
        snap["relabeled"] = n
        state.set_state(conn, MERGE_UNDO_KEY, json.dumps(snap))
    else:
        state.set_state(conn, MERGE_UNDO_KEY, "")
    return n, snap is not None


def _load_merge_undo(conn: sqlite3.Connection) -> dict | None:
    raw = state.get_state(conn, MERGE_UNDO_KEY)
    if not raw:
        return None
    try:
        snap = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(snap, dict) or "src" not in snap or "dst" not in snap:
        return None
    return snap


def _merge_undo_intact(conn: sqlite3.Connection, snap: dict) -> bool:
    row = conn.execute(
        "SELECT merged_into FROM speakers WHERE id=?", (snap["src"],)
    ).fetchone()
    return row is not None and row["merged_into"] == snap["dst"]


def pending_merge_undo(conn: sqlite3.Connection) -> dict | None:
    """Summary of the last merge if it is still undoable (fresh and intact)."""
    snap = _load_merge_undo(conn)
    if snap is None:
        return None
    at = _parse_utc_ts(snap.get("at"))
    if at is None or (datetime.now(UTC) - at).total_seconds() > MERGE_UNDO_WINDOW_S:
        return None
    if not _merge_undo_intact(conn, snap):
        return None
    return {
        "src": snap["src"],
        "dst": snap["dst"],
        "src_label": snap.get("src_label") or f"Speaker #{snap['src']}",
        "dst_label": snap.get("dst_label") or f"Speaker #{snap['dst']}",
        "relabeled": snap.get("relabeled", len(snap.get("segment_ids", []))),
        "at": snap.get("at"),
    }


def undo_merge(conn: sqlite3.Connection) -> dict:
    """Reverse the last snapshot-recorded merge.

    Returns ``{"status": ...}`` with one of ``ok`` (plus restore details),
    ``none`` (nothing recorded), ``expired`` (window passed), or ``stale``
    (the voices changed since — e.g. a newer merge chained onto them).
    """
    snap = _load_merge_undo(conn)
    if snap is None:
        return {"status": "none"}
    at = _parse_utc_ts(snap.get("at"))
    if at is None or (datetime.now(UTC) - at).total_seconds() > MERGE_UNDO_WINDOW_S:
        state.set_state(conn, MERGE_UNDO_KEY, "")
        return {"status": "expired"}
    if not _merge_undo_intact(conn, snap):
        state.set_state(conn, MERGE_UNDO_KEY, "")
        return {"status": "stale"}
    restored = registry.unmerge_speakers(
        conn,
        int(snap["src"]),
        int(snap["dst"]),
        [int(i) for i in snap.get("segment_ids", [])],
        [int(i) for i in snap.get("observation_ids", [])],
    )
    prior = snap.get("dst_prior")
    if isinstance(prior, dict) and snap.get("adopted_name"):
        # dst only carries this name because the merge adopted it from src —
        # give dst back its anonymous pre-merge identity, unless the user
        # renamed it since (then their newer name wins and stays).
        row = conn.execute(
            "SELECT name FROM speakers WHERE id=?", (int(snap["dst"]),)
        ).fetchone()
        if row is not None and row["name"] == snap["adopted_name"]:
            conn.execute(
                "UPDATE speakers SET name=?, display_label=?, "
                "kind=CASE WHEN is_owner=1 THEN 'owner' ELSE ? END WHERE id=?",
                (prior.get("name"), prior.get("display_label"), prior.get("kind"),
                 int(snap["dst"])),
            )
    state.set_state(conn, MERGE_UNDO_KEY, "")
    return {
        "status": "ok",
        "restored_segments": restored,
        "src": {"id": snap["src"], "label": snap.get("src_label") or f"Speaker #{snap['src']}"},
        "dst": {"id": snap["dst"], "label": snap.get("dst_label") or f"Speaker #{snap['dst']}"},
    }


def set_owner(conn: sqlite3.Connection, speaker_id: int) -> None:
    """Mark an existing (history-discovered) speaker as the owner.

    Any previously-marked owner is demoted to a regular 'known' voice (not left
    with a stale kind='owner'), keeping exactly one owner row at all times.
    """
    sid = registry.resolve_speaker_id(conn, speaker_id)
    conn.execute(
        "UPDATE speakers SET is_owner=0, kind='known' WHERE is_owner=1 AND id<>?", (sid,)
    )
    conn.execute("UPDATE speakers SET is_owner=1, kind='owner' WHERE id=?", (sid,))


# --- speaker quality / self-correction (Phase 7) -----------------------------


def reassign_segment(
    conn, segment_id: int, speaker_id: int, settings: Settings | None = None
) -> bool:
    from secondbrain.speaker import correct

    return correct.reassign_segment(conn, segment_id, speaker_id, settings or get_settings())


def unassign_segment(conn, segment_id: int, settings: Settings | None = None) -> bool:
    from secondbrain.speaker import correct

    return correct.unassign_segment(conn, segment_id, settings or get_settings())


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
        # Dismissed ("ignored") voices aren't people being tracked — they'd
        # inflate the headline count with every TV and passer-by.
        "speakers": conn.execute(
            "SELECT COUNT(*) AS n FROM speakers WHERE merged_into IS NULL AND kind<>'ignored'"
        ).fetchone()["n"],
        "ignored_speakers": conn.execute(
            "SELECT COUNT(*) AS n FROM speakers WHERE merged_into IS NULL AND kind='ignored'"
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
        # Additive context fields (web UI microcopy); existing keys are contract.
        "attributed_segments": conn.execute(
            "SELECT COUNT(*) AS n FROM transcript_segments WHERE speaker_id IS NOT NULL"
        ).fetchone()["n"],
        "unattributed_segments": conn.execute(
            "SELECT COUNT(*) AS n FROM transcript_segments WHERE speaker_id IS NULL"
        ).fetchone()["n"],
        "total_segments": conn.execute(
            "SELECT COUNT(*) AS n FROM transcript_segments"
        ).fetchone()["n"],
        "low_confidence_threshold": low,
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


def dismiss_action_item(conn, edge_id: int) -> bool:
    """Mark a detected action item as not-a-todo (edge ``valid=0``) so it
    stops appearing on the Tasks page. The edge itself is kept (nothing is
    deleted); ``list_action_items`` already excludes invalidated edges.
    Idempotent. Returns False when no such action-item edge exists."""
    row = conn.execute(
        "SELECT id FROM kg_edges WHERE id=? AND kind='action_item'", (edge_id,)
    ).fetchone()
    if row is None:
        return False
    conn.execute("UPDATE kg_edges SET valid=0 WHERE id=? AND kind='action_item'", (edge_id,))
    return True


def release_stale_scheduled(conn, before_day: str | None = None) -> int:
    from secondbrain.tasks import store

    return store.release_stale_scheduled(conn, before_day or local_today())


def decompose_goal(conn, goal_id: int, settings: Settings | None = None) -> dict:
    from secondbrain.tasks import decompose

    return decompose.propose_plan(conn, goal_id, settings=settings or get_settings())


def accept_plan(conn, goal_id: int, plan: dict) -> list[int]:
    from secondbrain.tasks import decompose

    return decompose.accept_plan(conn, goal_id, plan)


def propose_day(conn, date=None, capacity_minutes=None, settings: Settings | None = None) -> dict:
    from secondbrain.tasks import planner

    # Default to the user's wall-clock day (planner alone would use UTC, which
    # files a morning plan under yesterday for anyone east of Greenwich).
    return planner.propose_day(conn, date or local_today(), capacity_minutes,
                               settings or get_settings())


def accept_day(conn, date=None) -> dict | None:
    from secondbrain.tasks import planner

    return planner.accept_day(conn, date or local_today())


def get_day(conn, date=None) -> dict | None:
    from secondbrain.tasks import planner

    return planner.get_day(conn, date or local_today())


def remove_from_day(conn, task_id: int, date=None) -> dict | None:
    from secondbrain.tasks import planner

    return planner.remove_from_day(conn, task_id, date or local_today())


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


def get_task(conn, task_id: int) -> dict | None:
    from secondbrain.tasks import store

    return store.get_task(conn, task_id)


def update_task(conn, task_id: int, **fields) -> None:
    from secondbrain.tasks import store

    store.update_task(conn, task_id, **fields)


def task_research_note_counts(conn) -> dict[int, int]:
    """task_id → stored research-note count (powers the "Notes (N)" toggles)."""
    rows = conn.execute(
        "SELECT task_id, COUNT(*) AS n FROM task_research GROUP BY task_id"
    ).fetchall()
    return {r["task_id"]: r["n"] for r in rows}


def segment_local_days(conn, segment_ids: list[int]) -> dict[int, str]:
    """segment_id → local calendar day it was heard on.

    Lets research-note ``seg:<id>`` citations link to ``/day?date=<day>#seg-<id>``
    even for notes stored before the day was recorded alongside the source.
    """
    ids = list({int(i) for i in segment_ids})
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, start_at FROM transcript_segments WHERE id IN ({marks})", ids
    ).fetchall()
    out: dict[int, str] = {}
    for r in rows:
        day = _local_day_of(r["start_at"])
        if day:
            out[r["id"]] = day
    return out


def annotate_task_priorities(
    conn, tasks: list[dict], settings: Settings | None = None
) -> list[dict]:
    """Attach the Eisenhower ``quadrant`` and planner ``priority_score`` to each
    task dict (in place, display-only fields). These are the exact signals
    ``propose_day`` ranks by, so a backlog sorted on ``priority_score`` matches
    what would be planned next."""
    from secondbrain.tasks import prioritize

    settings = settings or get_settings()
    today = datetime.strptime(local_today(), "%Y-%m-%d").date()
    for t in tasks:
        t["quadrant"] = prioritize.quadrant(conn, t, settings, today)
        t["priority_score"] = prioritize.score(conn, t, settings, today)
    return tasks


def list_action_items(conn) -> list[dict]:
    """Open action-item edges not yet promoted into a task.

    These are commitments the knowledge extractor heard in conversations
    ("I'll send the report Friday"); each can be turned into a real task with
    one tap. Edges already promoted (a task points at them via
    ``source_edge_id``) or invalidated/superseded are excluded."""
    rows = conn.execute(
        """
        SELECT e.id, e.object_text, e.due_date, e.created_at,
               COALESCE(e.first_seen, e.created_at) AS first_seen,
               s.name AS src_name, d.name AS dst_name
        FROM kg_edges e
        JOIN kg_nodes s ON s.id = e.src_node_id
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        WHERE e.kind = 'action_item' AND e.valid = 1
          AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.source_edge_id = e.id)
        ORDER BY (e.due_date IS NULL), e.due_date, e.id DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


# --- knowledge graph + Q&A (Phase 3) -----------------------------------------


def ask(
    conn: sqlite3.Connection,
    question: str,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Grounded Q&A. ``history`` is optional prior turns [{question, answer}]
    so follow-up questions can resolve pronouns; older callers omit it."""
    from secondbrain.knowledge import chat

    return chat.answer(conn, question, settings=settings or get_settings(), history=history)


def _graph_search_where(query: str, node_type: str | None = None) -> tuple[str, list[str]]:
    """WHERE fragment + params for non-merged nodes matching ``query``.

    Every whitespace-separated term must match ('atlas project' finds
    'Project Atlas'), either in the node's normalized name or in one of its
    aliases — so entities merged under a new name stay findable by the old one.
    An empty query matches the whole (non-merged) graph; ``node_type``
    optionally narrows it to a single entity type. LIKE wildcards in the
    terms are escaped, so '100%' matches a literal percent sign instead of
    everything and '_' stops matching any single character.
    """
    from secondbrain.knowledge.graph import normalize_name

    clauses = ["merged_into IS NULL"]
    params: list[str] = []
    if node_type:
        clauses.append("type = ?")
        params.append(node_type)
    for term in normalize_name(query).split():
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{escaped}%"
        clauses.append(
            "(normalized_name LIKE ? ESCAPE '\\' OR id IN "
            "(SELECT node_id FROM kg_aliases WHERE normalized_alias LIKE ? ESCAPE '\\'))"
        )
        params.extend([like, like])
    return " AND ".join(clauses), params


def _matched_alias(conn: sqlite3.Connection, row: sqlite3.Row, terms: list[str]) -> str | None:
    """The alias that got ``row`` into the results when its own name didn't.

    Escaped-LIKE matching is exactly substring matching on normalized text, so
    a plain ``in`` mirrors the SQL. Returns None when the name alone satisfies
    every term (nothing to explain) or, defensively, when no alias covers the
    leftover terms.
    """
    missing = [t for t in terms if t not in row["normalized_name"]]
    if not missing:
        return None
    aliases = conn.execute(
        "SELECT alias, normalized_alias FROM kg_aliases WHERE node_id=? ORDER BY alias",
        (row["id"],),
    ).fetchall()
    for a in aliases:
        if any(t in a["normalized_alias"] for t in missing):
            return a["alias"]
    return None


def graph_search(
    conn: sqlite3.Connection, query: str, limit: int = 20, offset: int = 0,
    node_type: str | None = None,
) -> list[dict]:
    """Nodes matching ``query`` by name or alias, most-connected first.

    ``query`` may be empty: that returns the most connected nodes overall,
    which the graph page uses as its default browse list; ``node_type``
    narrows either mode to one entity type. ``offset`` skips already-fetched
    rows so the page's "Show more" button can walk the whole graph in stable
    ``limit``-sized steps. Each node carries ``matched_alias`` — the alias
    that matched when its visible name didn't (else None) — so the UI can say
    *why* 'greg' surfaced 'Gregory Hall'.
    """
    from secondbrain.knowledge.graph import normalize_name

    where, params = _graph_search_where(query, node_type)
    rows = conn.execute(
        f"""
        SELECT id, type, name, normalized_name, display_label,
               (SELECT COUNT(*) FROM kg_edges e
                WHERE e.valid=1 AND (e.src_node_id=kg_nodes.id OR e.dst_node_id=kg_nodes.id))
               AS edge_count
        FROM kg_nodes
        WHERE {where}
        ORDER BY edge_count DESC, normalized_name ASC LIMIT ? OFFSET ?
        """,
        (*params, limit, offset),
    ).fetchall()
    terms = normalize_name(query).split()
    out = []
    for r in rows:
        d = dict(r)
        del d["normalized_name"]  # internal (drives matched_alias), not API payload
        d["label"] = r["display_label"] or r["name"]
        d["matched_alias"] = _matched_alias(conn, r, terms) if terms else None
        out.append(d)
    return out


def graph_search_total(conn: sqlite3.Connection, query: str = "",
                       node_type: str | None = None) -> int:
    """How many non-merged nodes match ``query`` ('' = the whole graph),
    optionally narrowed to one entity ``node_type``."""
    where, params = _graph_search_where(query, node_type)
    row = conn.execute(f"SELECT COUNT(*) AS n FROM kg_nodes WHERE {where}", params).fetchone()
    return int(row["n"])


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


def goal_status_counts(conn) -> dict:
    from secondbrain.goals import store

    return store.status_counts(conn)


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
    from secondbrain.knowledge.chat import _CITE
    from secondbrain.proactive import store

    d = store.get_digest(conn, date or _today(), kind)
    if d is not None:
        # Additive: resolve [seg_id] markers in the summary so the UI can link
        # each citation to its moment in the day view (same shape as /api/ask).
        d["citations"] = citation_meta(conn, [int(m) for m in _CITE.findall(d["summary_md"])])
    return d


def list_digest_dates(conn) -> dict[str, list[str]]:
    from secondbrain.proactive import store

    return store.list_digest_dates(conn)


def citation_meta(conn, seg_ids: list[int]) -> list[dict]:
    """Resolve segment ids to citation payloads ({segment_id, start_at, speaker,
    text}) matching the /api/ask citation shape. Unknown ids are skipped."""
    ids = sorted({int(i) for i in seg_ids})[:200]
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT ts.id AS segment_id, ts.start_at, ts.text,
               sp.id, sp.name, sp.display_label
        FROM transcript_segments ts
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE ts.id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return [
        {
            "segment_id": r["segment_id"],
            "start_at": r["start_at"],
            "speaker": _speaker_label(r) if r["id"] is not None else None,
            "text": (r["text"] or "")[:200],
        }
        for r in rows
    ]


def list_suggestions(conn, date: str | None = None, status: str = "open") -> list[dict]:
    from secondbrain.proactive import store

    return store.list_suggestions(conn, date, status)


def suggestion_action(conn, suggestion_id: int, action: str) -> bool:
    """Apply an action to a suggestion; False when the id doesn't exist."""
    from secondbrain.proactive import store

    return store.suggestion_action(conn, suggestion_id, action)


def digest_generation_status(conn, kind: str = "daily") -> dict:
    """In-flight generation marker plus today's digest stamp for ``kind``.

    Powers the brief page's resumable progress line: ``generating`` says a run
    is under way (``started_at`` = its UTC start), and ``created_at`` is the
    current stamp of today's digest row — once it moves past ``started_at``,
    the run has landed.
    """
    from secondbrain.proactive import store

    today = _today()
    started = store.generating_since(conn, kind)
    d = store.get_digest(conn, today, kind)
    return {
        "kind": kind,
        "date": today,
        "generating": started is not None,
        "started_at": started,
        "created_at": (d or {}).get("created_at"),
    }


def digest_generating(conn) -> dict[str, str | None]:
    """Per-kind started-at marker of any in-flight digest generation."""
    from secondbrain.proactive import store

    return {k: store.generating_since(conn, k) for k in ("daily", "weekly")}


def _today() -> str:
    # Local calendar day: matches the brief page's "Today" label, the daemon's
    # local-hour digest schedule, and how the owner experiences the day.
    return local_today()


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


def export_data(conn, out_dir, fmt: str = "both", settings: Settings | None = None,
                since: str | None = None, until: str | None = None) -> list:
    """Export transcripts/graph/goals/tasks as JSON and/or Markdown. Returns paths."""
    from secondbrain.storage import backup

    settings = settings or get_settings()
    paths = []
    if fmt in ("json", "both"):
        paths.append(backup.export_json(conn, out_dir, settings, since, until))
    if fmt in ("md", "markdown", "both"):
        paths.append(backup.export_markdown(conn, out_dir, settings, since, until))
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


def _quotes_for_segments(conn, seg_ids: set[int], settings: Settings) -> dict[int, dict]:
    """segment_id → {segment_id, start_at, text, speaker} for citable segments.

    Mirrors ``_segment_quotes``: redacted text and opted-out speakers are
    excluded, so provenance quotes respect the same privacy rules as every
    other read surface. ``speaker`` labels who said the line (resolved the
    same way as ``citation_meta``), or None while the voice is unidentified —
    provenance should answer *who* said it, not just which conversation.
    """
    if not seg_ids:
        return {}
    opted = registry.opted_out_speaker_ids(conn, settings)
    ph = ",".join("?" * len(seg_ids))
    rows = conn.execute(
        f"SELECT ts.id, ts.start_at, ts.text, ts.speaker_id, "
        f"       sp.id AS sp_id, sp.name AS sp_name, sp.display_label AS sp_label "
        f"FROM transcript_segments ts LEFT JOIN speakers sp ON sp.id = ts.speaker_id "
        f"WHERE ts.id IN ({ph}) AND ts.text != ?",
        (*seg_ids, registry.REDACTED_TEXT),
    ).fetchall()
    out: dict[int, dict] = {}
    for r in rows:
        sid = r["speaker_id"]
        if sid is not None and registry.resolve_speaker_id(conn, sid) in opted:
            continue
        speaker = None
        if r["sp_id"] is not None:  # same fallback chain as _speaker_label
            speaker = r["sp_name"] or r["sp_label"] or f"Speaker {r['sp_id']}"
        out[r["id"]] = {"segment_id": r["id"], "start_at": r["start_at"],
                        "text": r["text"], "speaker": speaker}
    return out


def graph_node(conn: sqlite3.Connection, node_id: int,
               settings: Settings | None = None) -> dict | None:
    """A node plus its valid edges, each with provenance: which conversation
    said it (local day/time for linking) and its citable source quotes —
    ``quotes`` lists every citable transcript line in citation order, while
    ``quote`` keeps the first one for older clients.

    Merged ids resolve to the canonical node. The raw ``embedding`` BLOB is
    deliberately not returned (binary is meaningless in JSON and would break
    serialization).
    """
    from secondbrain.knowledge.graph import normalize_name, resolve_node_id

    settings = settings or get_settings()
    nid = resolve_node_id(conn, node_id)
    node = conn.execute(
        """
        SELECT id, type, name, normalized_name, display_label, speaker_id,
               confidence, source_extraction_id, merged_into, first_seen,
               last_seen, created_at
        FROM kg_nodes WHERE id=?
        """,
        (nid,),
    ).fetchone()
    if node is None:
        return None
    edges = conn.execute(
        """
        SELECT e.id, e.predicate, e.kind, e.object_text, e.due_date, e.confidence,
               e.conversation_id, e.source_segment_ids, e.first_seen, e.last_seen,
               e.src_node_id AS src_id, s.name AS src_name,
               COALESCE(s.display_label, s.name) AS src_label,
               d.name AS dst_name, d.id AS dst_id,
               COALESCE(d.display_label, d.name) AS dst_label,
               c.started_at AS conversation_started_at
        FROM kg_edges e
        JOIN kg_nodes s ON s.id = e.src_node_id
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        LEFT JOIN conversations c ON c.id = e.conversation_id
        WHERE e.valid=1 AND (e.src_node_id=? OR e.dst_node_id=?)
        ORDER BY e.confidence DESC
        """,
        (nid, nid),
    ).fetchall()

    out_edges = [dict(e) for e in edges]
    all_seg_ids: set[int] = set()
    for e in out_edges:
        e["segment_ids"] = json.loads(e["source_segment_ids"] or "[]")
        all_seg_ids.update(e["segment_ids"])
        e["conversation_day"] = _local_day_of(e["conversation_started_at"])
        e["conversation_time"] = _local_hhmm(e["conversation_started_at"])
    quotes = _quotes_for_segments(conn, all_seg_ids, settings)
    for e in out_edges:
        e["quotes"] = [quotes[s] for s in e["segment_ids"] if s in quotes]
        e["quote"] = e["quotes"][0] if e["quotes"] else None  # legacy single-quote field

    d = dict(node)
    d["label"] = node["display_label"] or node["name"]
    aliases = [
        r["alias"]
        for r in conn.execute(
            "SELECT alias FROM kg_aliases WHERE node_id=? ORDER BY alias", (nid,)
        ).fetchall()
        if normalize_name(r["alias"]) != node["normalized_name"]
    ]
    return {"node": d, "edges": out_edges, "aliases": aliases, "edge_count": len(out_edges)}
