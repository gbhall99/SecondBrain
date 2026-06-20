"""User correction of a segment's speaker — reassign, lock, and feed learning."""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.speaker import registry


def reassign_segment(
    conn: sqlite3.Connection, segment_id: int, speaker_id: int, settings: Settings | None = None
) -> bool:
    """Reassign a segment to the correct speaker, lock it, and add a confirmed
    exemplar (so future matching improves). Returns True on success."""
    settings = settings or get_settings()
    seg = conn.execute(
        "SELECT observation_id FROM transcript_segments WHERE id=?", (segment_id,)
    ).fetchone()
    if seg is None:
        return False
    target = registry.resolve_speaker_id(conn, speaker_id)
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=1.0, "
        "speaker_locked=1, speaker_source='user' WHERE id=?",
        (target, segment_id),
    )
    # Feed the correction back into the profile via the observation's embedding.
    if seg["observation_id"]:
        obs = conn.execute(
            "SELECT embedding, start_at FROM speaker_observations WHERE id=?",
            (seg["observation_id"],),
        ).fetchone()
        emb = registry.deserialize_embedding(obs["embedding"]) if obs else None
        if emb:
            registry.add_confirmed_exemplar(conn, target, emb, start_at=obs["start_at"])
    registry._recount_segments(conn, target)
    return True
