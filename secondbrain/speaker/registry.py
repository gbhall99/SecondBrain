"""Global speaker registry: embeddings, matching, centroids, opt-out.

Embeddings are L2-normalized float32 vectors stored as struct-packed BLOBs
(profile centroid on ``speakers.centroid``; per-observation on
``speaker_observations.embedding``) and compared with cosine similarity in pure
Python. At single-user scale (a handful of profiles, a few thousand
observations) this needs no ANN index and keeps the logic dependency-free and
fully testable on CI.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass

from secondbrain.config import Settings, get_settings

REDACTED_TEXT = "[redacted: opted-out speaker]"


# --- embedding helpers -------------------------------------------------------


def serialize_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# --- speaker rows ------------------------------------------------------------


def get_or_create_owner(conn: sqlite3.Connection, name: str = "Me") -> int:
    row = conn.execute("SELECT id FROM speakers WHERE is_owner=1 LIMIT 1").fetchone()
    if row is not None:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO speakers (name, is_owner, kind, display_label) VALUES (?, 1, 'owner', ?)",
        (name, name),
    )
    return int(cur.lastrowid)


def create_unknown_speaker(conn: sqlite3.Connection) -> int:
    n = conn.execute("SELECT COUNT(*) AS n FROM speakers WHERE kind='unknown'").fetchone()["n"]
    label = f"Unknown #{n + 1}"
    cur = conn.execute(
        "INSERT INTO speakers (kind, display_label) VALUES ('unknown', ?)", (label,)
    )
    return int(cur.lastrowid)


def resolve_speaker_id(conn: sqlite3.Connection, speaker_id: int) -> int:
    """Follow the merged_into chain to the canonical speaker id."""
    seen = set()
    cur = speaker_id
    while cur not in seen:
        seen.add(cur)
        row = conn.execute("SELECT merged_into FROM speakers WHERE id=?", (cur,)).fetchone()
        if row is None or row["merged_into"] is None:
            return cur
        cur = int(row["merged_into"])
    return cur


# --- matching ----------------------------------------------------------------


@dataclass
class MatchResult:
    speaker_id: int | None
    similarity: float
    is_owner: bool = False


def _candidate_profiles(conn: sqlite3.Connection) -> list[tuple[int, str, list[float]]]:
    rows = conn.execute(
        "SELECT id, kind, centroid FROM speakers "
        "WHERE merged_into IS NULL AND centroid IS NOT NULL"
    ).fetchall()
    out = []
    for r in rows:
        vec = deserialize_embedding(r["centroid"])
        if vec:
            out.append((int(r["id"]), r["kind"], vec))
    return out


def match_embedding(
    conn: sqlite3.Connection, emb: list[float], settings: Settings | None = None
) -> MatchResult:
    """Resolve a cluster embedding to a known speaker, or no match.

    Owner is checked first with a slightly looser threshold so the user's own
    voice is the most reliable distinction.
    """
    settings = settings or get_settings()
    d = settings.diarization
    candidates = _candidate_profiles(conn)
    if not candidates:
        return MatchResult(None, 0.0)

    best_owner = (-1.0, None)
    best_any = (-1.0, None)
    for sid, kind, vec in candidates:
        sim = cosine(emb, vec)
        if sim > best_any[0]:
            best_any = (sim, sid)
        if kind == "owner" and sim > best_owner[0]:
            best_owner = (sim, sid)

    if best_owner[1] is not None and best_owner[0] >= d.owner_match_threshold:
        return MatchResult(best_owner[1], best_owner[0], is_owner=True)
    if best_any[1] is not None and best_any[0] >= d.match_threshold:
        return MatchResult(best_any[1], best_any[0])
    return MatchResult(None, best_any[0] if best_any[0] >= 0 else 0.0)


# --- observations + centroid updates -----------------------------------------


def record_observation(
    conn: sqlite3.Connection,
    *,
    speaker_id: int,
    audio_file_id: int | None,
    conversation_id: int | None,
    start_offset_s: float,
    end_offset_s: float,
    start_at: str | None,
    confidence: float | None,
    embedding: list[float],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO speaker_observations
            (speaker_id, conversation_id, audio_file_id, start_offset_s,
             end_offset_s, start_at, confidence, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            speaker_id,
            conversation_id,
            audio_file_id,
            start_offset_s,
            end_offset_s,
            start_at,
            confidence,
            serialize_embedding(embedding),
        ),
    )
    return int(cur.lastrowid)


def update_centroid(conn: sqlite3.Connection, speaker_id: int, emb: list[float]) -> None:
    """Fold an embedding into the speaker's centroid as a running mean."""
    row = conn.execute(
        "SELECT centroid, exemplar_count FROM speakers WHERE id=?", (speaker_id,)
    ).fetchone()
    count = int(row["exemplar_count"]) if row else 0
    current = deserialize_embedding(row["centroid"]) if row else None
    if current is None or count == 0:
        new = normalize(emb)
    else:
        new = normalize([(c * count + e) / (count + 1) for c, e in zip(current, emb, strict=False)])
    conn.execute(
        "UPDATE speakers SET centroid=?, exemplar_count=? WHERE id=?",
        (serialize_embedding(new), count + 1, speaker_id),
    )


def touch_speaker_stats(
    conn: sqlite3.Connection, speaker_id: int, *, last_seen_at: str | None, segments_added: int
) -> None:
    conn.execute(
        "UPDATE speakers SET segment_count = segment_count + ?, "
        "last_seen_at = MAX(COALESCE(last_seen_at, ''), COALESCE(?, '')) WHERE id=?",
        (segments_added, last_seen_at, speaker_id),
    )


def assign_segment_speaker(
    conn: sqlite3.Connection, segment_id: int, speaker_id: int, confidence: float | None
) -> None:
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=? WHERE id=?",
        (speaker_id, confidence, segment_id),
    )


# --- opt-out enforcement -----------------------------------------------------


def is_opted_out(
    conn: sqlite3.Connection, speaker_id: int, settings: Settings | None = None
) -> bool:
    settings = settings or get_settings()
    row = conn.execute(
        "SELECT name, opted_out, is_owner FROM speakers WHERE id=?", (speaker_id,)
    ).fetchone()
    if row is None or row["is_owner"]:
        return False
    if row["opted_out"]:
        return True
    return bool(row["name"]) and row["name"] in set(settings.consent.speaker_opt_out)


def redact_segment(conn: sqlite3.Connection, segment_id: int) -> None:
    """Replace a segment's text with a sentinel and purge its search vectors.

    The FTS index updates automatically via the AFTER UPDATE trigger.
    """
    conn.execute(
        "UPDATE transcript_segments SET text=? WHERE id=?", (REDACTED_TEXT, segment_id)
    )
    # Best-effort purge of any semantic vector (table may not exist).
    try:
        conn.execute("DELETE FROM segment_vectors WHERE segment_id=?", (segment_id,))
    except sqlite3.OperationalError:
        pass


def redact_speaker_segments(conn: sqlite3.Connection, speaker_id: int) -> int:
    rows = conn.execute(
        "SELECT id FROM transcript_segments WHERE speaker_id=? AND text<>?",
        (speaker_id, REDACTED_TEXT),
    ).fetchall()
    for r in rows:
        redact_segment(conn, r["id"])
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=?", (speaker_id,))
    return len(rows)


# --- naming, recompute, merge ------------------------------------------------


def recompute_centroid(conn: sqlite3.Connection, speaker_id: int) -> None:
    """Recompute a speaker's centroid as the mean of all its observations."""
    rows = conn.execute(
        "SELECT embedding FROM speaker_observations WHERE speaker_id=? AND embedding IS NOT NULL",
        (speaker_id,),
    ).fetchall()
    vecs = [v for v in (deserialize_embedding(r["embedding"]) for r in rows) if v]
    if not vecs:
        return
    dim = len(vecs[0])
    mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    conn.execute(
        "UPDATE speakers SET centroid=?, exemplar_count=? WHERE id=?",
        (serialize_embedding(normalize(mean)), len(vecs), speaker_id),
    )


def _recount_segments(conn: sqlite3.Connection, speaker_id: int) -> None:
    row = conn.execute(
        "SELECT COUNT(*) AS n, MAX(start_at) AS last FROM transcript_segments WHERE speaker_id=?",
        (speaker_id,),
    ).fetchone()
    conn.execute(
        "UPDATE speakers SET segment_count=?, last_seen_at=? WHERE id=?",
        (row["n"], row["last"], speaker_id),
    )


def name_speaker(
    conn: sqlite3.Connection, speaker_id: int, name: str, settings: Settings | None = None
) -> int:
    """Name a speaker (→ kind 'known'). History updates implicitly (same id).

    If the assigned name is opted out, retroactively redact their segments.
    """
    settings = settings or get_settings()
    sid = resolve_speaker_id(conn, speaker_id)
    conn.execute(
        "UPDATE speakers SET name=?, display_label=?, kind=CASE WHEN is_owner=1 THEN 'owner' "
        "ELSE 'known' END WHERE id=?",
        (name, name, sid),
    )
    redacted = 0
    if name in set(settings.consent.speaker_opt_out):
        redacted = redact_speaker_segments(conn, sid)
    return redacted


def merge_speakers(
    conn: sqlite3.Connection, src_id: int, dst_id: int, settings: Settings | None = None
) -> int:
    """Merge ``src`` into ``dst``: relabel history + observations, recompute
    centroid, soft-mark ``src.merged_into=dst``. Returns segments relabeled."""
    settings = settings or get_settings()
    src = resolve_speaker_id(conn, src_id)
    dst = resolve_speaker_id(conn, dst_id)
    if src == dst:
        return 0
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM transcript_segments WHERE speaker_id=?", (src,)
    ).fetchone()["n"]
    conn.execute("UPDATE transcript_segments SET speaker_id=? WHERE speaker_id=?", (dst, src))
    conn.execute("UPDATE speaker_observations SET speaker_id=? WHERE speaker_id=?", (dst, src))
    conn.execute("UPDATE speakers SET merged_into=? WHERE id=?", (dst, src))
    recompute_centroid(conn, dst)
    _recount_segments(conn, dst)
    if is_opted_out(conn, dst, settings):
        redact_speaker_segments(conn, dst)
    return n
