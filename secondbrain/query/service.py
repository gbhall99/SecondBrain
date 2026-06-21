"""Backend-agnostic query helpers shared by the API, CLI, and (later) chat."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.search import combined
from secondbrain.speaker import registry
from secondbrain.storage import retention, state
from secondbrain.storage.models import parse_iso, segments_for_day


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
           settings: Settings | None = None, since: str | None = None,
           until: str | None = None) -> list[dict]:
    settings = settings or get_settings()
    # Over-fetch when date-filtering so the post-filter can still return ~limit.
    fetch = limit * 4 if (since or until) else limit
    hits = combined.search(conn, query, fetch, settings=settings, mode=mode)
    results = [asdict(h) for h in hits]
    if since or until:
        results = [r for r in results if _in_day_range(r.get("start_at"), since, until)]
    results = results[:limit]
    labels = _speaker_labels(conn, [h["segment_id"] for h in results])
    for h in results:
        h.update(labels.get(h["segment_id"], {}))
    return results


def _in_day_range(start_at: str | None, since: str | None, until: str | None) -> bool:
    day = (start_at or "")[:10]
    if not day:
        return False
    if since and day < since:
        return False
    return not (until and day > until)


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

    span = conn.execute(
        "SELECT MIN(substr(start_at,1,10)) AS first, MAX(substr(start_at,1,10)) AS last "
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
        "first_day": span["first"],
        "last_day": span["last"],
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
    out = []
    for osid, shared in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
        s = conn.execute(
            "SELECT name, display_label, is_owner FROM speakers WHERE id=?", (osid,)
        ).fetchone()
        if s is None:
            continue
        lbl = "Me" if s["is_owner"] else (s["name"] or s["display_label"] or f"Speaker {osid}")
        out.append({"speaker_id": osid, "label": lbl, "shared_conversations": shared})
    return out


def timeline(conn: sqlite3.Connection, day: str | None = None,
             settings: Settings | None = None) -> list[dict]:
    """A day as a chronological list of conversations, each with attributed
    segments (opt-out filtered) and the knowledge extracted from it."""
    settings = settings or get_settings()
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    opted = registry.opted_out_speaker_ids(conn, settings)
    rows = conn.execute(
        """
        SELECT ts.id, ts.start_at, ts.text, ts.speaker_id, af.conversation_id AS conv,
               COALESCE(sp.name, sp.display_label) AS speaker,
               CASE WHEN sp.is_owner THEN 1 ELSE 0 END AS is_owner
        FROM transcript_segments ts
        JOIN audio_files af ON af.id = ts.audio_file_id
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE substr(ts.start_at, 1, 10) = ?
        ORDER BY ts.start_at, ts.id
        """,
        (day,),
    ).fetchall()
    blocks: dict = {}
    order: list = []
    for r in rows:
        if r["speaker_id"] in opted:
            continue
        cid = r["conv"]
        if cid not in blocks:
            blocks[cid] = {
                "conversation_id": cid,
                "started_at": r["start_at"],
                "participants": set(),
                "segments": [],
                "extractions": {},
            }
            order.append(cid)
        b = blocks[cid]
        label = "Me" if r["is_owner"] else (r["speaker"] or "Unknown")
        b["participants"].add(label)
        b["segments"].append(
            {"id": r["id"], "start_at": r["start_at"], "speaker": label, "text": r["text"]}
        )
    for cid, b in blocks.items():
        if cid is not None:
            grouped: dict = {}
            for e in conn.execute(
                "SELECT kind, predicate, object_text, source_segment_ids FROM kg_edges "
                "WHERE conversation_id=? AND valid=1 ORDER BY kind",
                (cid,),
            ).fetchall():
                grouped.setdefault(e["kind"], []).append({
                    "predicate": e["predicate"],
                    "object_text": e["object_text"],
                    "segment_ids": json.loads(e["source_segment_ids"] or "[]"),
                })
            b["extractions"] = grouped
        b["participants"] = sorted(b["participants"])
    return [blocks[c] for c in order]


def relationships(conn: sqlite3.Connection, settings: Settings | None = None) -> list[dict]:
    """People you interact with, ranked by interaction; opt-out/owner excluded."""
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
        })
    out.sort(key=lambda x: (x["conversations"], x["talk_minutes"]), reverse=True)
    return out


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
    label = "Me" if spk["is_owner"] else (spk["name"] or spk["display_label"] or f"Speaker {sid}")
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
        },
        "connections": _person_connections(conn, sid, settings),
        "facts": [],
        "commitments": {"owed_by": [], "owed_to": []},
        "recent_quotes": [],
    }

    # Privacy: an opted-out person gets identity/interaction shape but no content.
    if opted and not spk["is_owner"]:
        return dossier

    if node_id is not None:
        dossier["facts"] = [
            dict(r)
            for r in conn.execute(
                "SELECT predicate, object_text, confidence, due_date FROM kg_edges "
                "WHERE src_node_id=? AND kind='fact' AND valid=1 ORDER BY confidence DESC",
                (node_id,),
            ).fetchall()
        ]
        dossier["commitments"] = {
            "owed_by": [
                dict(r)
                for r in conn.execute(
                    "SELECT object_text, due_date, confidence FROM kg_edges "
                    "WHERE src_node_id=? AND kind='action_item' AND valid=1 ORDER BY due_date",
                    (node_id,),
                ).fetchall()
            ],
            "owed_to": [
                dict(r)
                for r in conn.execute(
                    "SELECT object_text, due_date, confidence FROM kg_edges "
                    "WHERE dst_node_id=? AND kind='action_item' AND valid=1 ORDER BY due_date",
                    (node_id,),
                ).fetchall()
            ],
        }

    dossier["recent_quotes"] = [
        dict(r)
        for r in conn.execute(
            "SELECT id, start_at, text FROM transcript_segments WHERE speaker_id=? "
            "AND text != ? ORDER BY start_at DESC, id DESC LIMIT ?",
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
    out = []
    for r in rows:
        sid = r["speaker_id"]
        if sid is not None and registry.resolve_speaker_id(conn, sid) in opted:
            continue
        out.append({"segment_id": r["id"], "start_at": r["start_at"], "text": r["text"]})
        if len(out) >= limit:
            break
    return out


def list_projects(conn: sqlite3.Connection, settings: Settings | None = None) -> list[dict]:
    """Projects (kg nodes) ranked by activity — conversations then edge volume."""
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
    out = []
    for r in rows:
        linked_goals = conn.execute(
            "SELECT COUNT(*) AS n FROM goal_links WHERE kind='node' AND ref_id=?", (r["id"],)
        ).fetchone()["n"]
        open_items = conn.execute(
            "SELECT COUNT(*) AS n FROM kg_edges WHERE valid=1 AND kind='action_item' "
            "AND (src_node_id=? OR dst_node_id=?)",
            (r["id"], r["id"]),
        ).fetchone()["n"]
        out.append({
            "node_id": r["id"],
            "label": r["display_label"] or r["name"],
            "edges": r["edges"],
            "conversations": r["conversations"],
            "linked_goals": linked_goals,
            "open_action_items": open_items,
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
        "SELECT id, src_node_id, dst_node_id, predicate, kind, object_text, due_date, "
        "confidence, conversation_id, source_segment_ids, last_seen FROM kg_edges "
        "WHERE valid=1 AND (src_node_id=? OR dst_node_id=?)",
        (nid, nid),
    ).fetchall()

    facts: list[dict] = []
    decisions: list[dict] = []
    action_items: list[dict] = []
    convos: set[int] = set()
    seg_ids: set[int] = set()
    for e in edges:
        if e["conversation_id"] is not None:
            convos.add(e["conversation_id"])
        seg_ids.update(json.loads(e["source_segment_ids"] or "[]"))
        item = {
            "predicate": e["predicate"],
            "object_text": e["object_text"],
            "due_date": e["due_date"],
            "confidence": e["confidence"],
            "segment_ids": json.loads(e["source_segment_ids"] or "[]"),
        }
        if e["kind"] == "fact":
            facts.append(item)
        elif e["kind"] == "decision":
            decisions.append(item)
        elif e["kind"] == "action_item":
            action_items.append(item)
    facts.sort(key=lambda x: x["confidence"] or 0, reverse=True)
    action_items.sort(key=lambda x: x["due_date"] or "9999")

    goals = [
        dict(r)
        for r in conn.execute(
            "SELECT g.id, g.title, g.status, g.priority, gl.relation FROM goal_links gl "
            "JOIN goals g ON g.id=gl.goal_id WHERE gl.kind='node' AND gl.ref_id=? "
            "ORDER BY g.priority, g.id",
            (nid,),
        ).fetchall()
    ]

    return {
        "node_id": nid,
        "label": node["display_label"] or node["name"],
        "type": node["type"],
        "aliases": [
            r["alias"]
            for r in conn.execute(
                "SELECT alias FROM kg_aliases WHERE node_id=?", (nid,)
            ).fetchall()
        ],
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
        "recent_quotes": _segment_quotes(conn, seg_ids, settings, limit=quotes),
    }


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
