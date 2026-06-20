"""Re-attribute past UNKNOWN / LOW-confidence segments as profiles improve.

Operates at the observation level (observations carry the acoustic embedding) and
cascades to the transcript segments aligned to them. Only relabels when a segment
is not user-locked and a KNOWN/owner profile now clears the HIGH
``reattribute_threshold``. Never overwrites user-confirmed labels.
"""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.speaker import registry
from secondbrain.storage import state
from secondbrain.storage.models import utcnow_iso

LAST_RUN_KEY = "reattribute_last_run"


def _candidate_observation_ids(conn: sqlite3.Connection, low: float) -> list[int]:
    rows = conn.execute(
        """
        SELECT DISTINCT observation_id FROM transcript_segments
        WHERE observation_id IS NOT NULL AND speaker_locked=0
          AND (speaker_id IS NULL OR speaker_confidence IS NULL OR speaker_confidence < ?)
        """,
        (low,),
    ).fetchall()
    return [int(r["observation_id"]) for r in rows]


def run_reattribution(conn: sqlite3.Connection, settings: Settings | None = None) -> int:
    """Relabel eligible segments whose observation now matches a known voice.

    Returns the number of segments relabeled.
    """
    settings = settings or get_settings()
    d = settings.diarization
    relabeled = 0
    for obs_id in _candidate_observation_ids(conn, d.low_confidence_threshold):
        obs = conn.execute(
            "SELECT speaker_id, embedding, start_at FROM speaker_observations WHERE id=?",
            (obs_id,),
        ).fetchone()
        if obs is None:
            continue
        emb = registry.deserialize_embedding(obs["embedding"])
        if not emb:
            continue
        m = registry.match_embedding(conn, emb, settings)
        if m.speaker_id is None or m.similarity < d.reattribute_threshold:
            continue
        target = registry.resolve_speaker_id(conn, m.speaker_id)
        cur_obs_speaker = (
            registry.resolve_speaker_id(conn, obs["speaker_id"]) if obs["speaker_id"] else None
        )
        if target == cur_obs_speaker:
            continue
        kind = conn.execute("SELECT kind FROM speakers WHERE id=?", (target,)).fetchone()
        if kind is None or kind["kind"] not in ("owner", "known"):
            continue  # don't shuffle among unknowns — clustering handles that

        conn.execute("UPDATE speaker_observations SET speaker_id=? WHERE id=?", (target, obs_id))
        cur = conn.execute(
            "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=?, "
            "speaker_source='reattributed' WHERE observation_id=? AND speaker_locked=0",
            (target, round(m.similarity, 4), obs_id),
        )
        relabeled += cur.rowcount or 0
    state.set_state(conn, LAST_RUN_KEY, utcnow_iso())
    return relabeled
