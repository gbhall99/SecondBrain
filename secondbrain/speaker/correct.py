"""User correction of a segment's speaker — reassign, lock, and feed learning."""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.speaker import registry
from secondbrain.storage.db import transaction


def _withdraw_stale_exemplar(
    conn: sqlite3.Connection, segment_id: int, seg: sqlite3.Row, target: int | None
) -> None:
    """Remove the confirmed exemplar an earlier correction taught the *old* speaker.

    When a locked line is re-corrected, the previous correction's teaching sample
    is now known to be wrong — leaving it would keep steering the old speaker's
    profile toward someone else's voice. Only rows created by
    ``add_confirmed_exemplar`` are touched (source='correction', no audio file),
    matched by the exact embedding of this segment's observation, and kept if
    another still-locked segment sharing the observation vouches for them.
    """
    old_speaker = seg["speaker_id"]
    if not seg["speaker_locked"] or old_speaker is None or old_speaker == target:
        return
    obs = conn.execute(
        "SELECT embedding FROM speaker_observations WHERE id=?", (seg["observation_id"],)
    ).fetchone()
    if obs is None or obs["embedding"] is None:
        return
    still_vouched = conn.execute(
        "SELECT 1 FROM transcript_segments "
        "WHERE observation_id=? AND speaker_id=? AND speaker_locked=1 AND id<>? LIMIT 1",
        (seg["observation_id"], old_speaker, segment_id),
    ).fetchone()
    if still_vouched:
        return
    stale = conn.execute(
        "SELECT id FROM speaker_observations WHERE speaker_id=? AND source='correction' "
        "AND audio_file_id IS NULL AND embedding=?",
        (old_speaker, obs["embedding"]),
    ).fetchall()
    if not stale:
        return
    conn.executemany(
        "DELETE FROM speaker_observations WHERE id=?", [(r["id"],) for r in stale]
    )
    remaining = conn.execute(
        "SELECT COUNT(*) AS n FROM speaker_observations "
        "WHERE speaker_id=? AND pruned=0 AND embedding IS NOT NULL",
        (old_speaker,),
    ).fetchone()["n"]
    if remaining:
        registry.recompute_centroid(conn, old_speaker)
    else:
        # The withdrawn exemplar was their only voice sample: leave no profile
        # rather than a profile built solely from someone else's voice.
        conn.execute(
            "UPDATE speakers SET centroid=NULL, exemplar_count=0 WHERE id=?", (old_speaker,)
        )


def reassign_segment(
    conn: sqlite3.Connection, segment_id: int, speaker_id: int,
    settings: Settings | None = None, *, propagate: bool = True,
) -> bool:
    """Reassign a segment to the correct speaker, lock it, and add a confirmed
    exemplar (so future matching improves). Returns True on success.

    With ``propagate`` (the default) the correction also flows to every other
    non-locked segment sharing the same diarized observation — one turn of
    speech is one voice, so its sibling lines were wrong for the same reason —
    and the observation itself moves to the corrected speaker (mirroring
    re-attribution). Only the corrected line is locked.

    Also handles the follow-on effects of a correction:
    - re-correcting an already-locked line withdraws the stale exemplar the
      earlier correction taught the old speaker (see _withdraw_stale_exemplar);
    - every affected speaker's profile/segment stats are refreshed once.
    Confirming the current guess (same speaker_id) is a supported teaching
    action: it locks the line and adds the exemplar exactly once (idempotent).
    The whole write sequence is one atomic transaction.
    """
    settings = settings or get_settings()
    seg = conn.execute(
        "SELECT observation_id, speaker_id, speaker_locked FROM transcript_segments WHERE id=?",
        (segment_id,),
    ).fetchone()
    if seg is None:
        return False
    target = registry.resolve_speaker_id(conn, speaker_id)
    # Refuse ghost targets: resolve_speaker_id passes unknown ids through, and a
    # correction must never write a dangling speaker_id or teach a profile that
    # doesn't exist.
    if conn.execute("SELECT 1 FROM speakers WHERE id=?", (target,)).fetchone() is None:
        return False
    old_speaker = seg["speaker_id"]
    touched: set[int] = {target}
    if old_speaker is not None:
        touched.add(int(old_speaker))
    with transaction(conn):
        if seg["observation_id"]:
            _withdraw_stale_exemplar(conn, segment_id, seg, target)
        conn.execute(
            "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=1.0, "
            "speaker_locked=1, speaker_source='user' WHERE id=?",
            (target, segment_id),
        )
        if propagate and seg["observation_id"]:
            # The sibling lines of the same diarized turn carry the same voice:
            # move every non-locked one along (locked lines are the user's word).
            prev = conn.execute(
                "SELECT DISTINCT speaker_id FROM transcript_segments "
                "WHERE observation_id=? AND speaker_locked=0 AND speaker_id IS NOT NULL",
                (seg["observation_id"],),
            ).fetchall()
            touched.update(int(r["speaker_id"]) for r in prev)
            conn.execute(
                "UPDATE transcript_segments SET speaker_id=?, speaker_confidence=1.0, "
                "speaker_source='user' WHERE observation_id=? AND speaker_locked=0",
                (target, seg["observation_id"]),
            )
            obs_row = conn.execute(
                "SELECT speaker_id FROM speaker_observations WHERE id=?",
                (seg["observation_id"],),
            ).fetchone()
            if obs_row is not None and obs_row["speaker_id"] is not None:
                touched.add(int(obs_row["speaker_id"]))
            conn.execute(
                "UPDATE speaker_observations SET speaker_id=? WHERE id=?",
                (target, seg["observation_id"]),
            )
        # Feed the correction back into the profile via the observation's embedding.
        if seg["observation_id"]:
            obs = conn.execute(
                "SELECT embedding, start_at FROM speaker_observations WHERE id=?",
                (seg["observation_id"],),
            ).fetchone()
            emb = registry.deserialize_embedding(obs["embedding"]) if obs else None
            already_taught = obs is not None and conn.execute(
                "SELECT 1 FROM speaker_observations WHERE speaker_id=? AND source='correction' "
                "AND audio_file_id IS NULL AND embedding=? LIMIT 1",
                (target, obs["embedding"]),
            ).fetchone()
            if emb and not already_taught:
                registry.add_confirmed_exemplar(conn, target, emb, start_at=obs["start_at"])
        # Recount/recompute every affected speaker exactly once, after all moves.
        for sid in touched:
            if sid != target and propagate and seg["observation_id"]:
                registry.refresh_profile(conn, sid)  # observation may have moved away
            registry._recount_segments(conn, sid)
    return True


def unassign_segment(
    conn: sqlite3.Connection, segment_id: int, settings: Settings | None = None
) -> bool:
    """Dispute a wrong attribution when there is no correct person to move to yet.

    Clears the segment's speaker, but *locks* it (``speaker_locked=1``,
    ``speaker_source='user'``) so re-attribution never silently re-guesses the
    voice the user just rejected — the line reads as an honest "Unknown" until
    the real speaker is named on the People page and the line reassigned. Any
    confirmed exemplar an earlier correction taught the old speaker is withdrawn
    (same rule as re-correcting a locked line), so a mistaken profile stops
    being steered by this segment. Returns True on success, False if there is no
    such segment. Idempotent: an already-unattributed line stays unattributed.
    """
    settings = settings or get_settings()
    seg = conn.execute(
        "SELECT observation_id, speaker_id, speaker_locked FROM transcript_segments WHERE id=?",
        (segment_id,),
    ).fetchone()
    if seg is None:
        return False
    old_speaker = seg["speaker_id"]
    if seg["observation_id"]:
        # target=None: withdraw the stale exemplar the old locked correction
        # taught (the None never equals a real old_speaker, so an already-blank
        # line is a no-op).
        _withdraw_stale_exemplar(conn, segment_id, seg, None)
    conn.execute(
        "UPDATE transcript_segments SET speaker_id=NULL, speaker_confidence=NULL, "
        "speaker_locked=1, speaker_source='user' WHERE id=?",
        (segment_id,),
    )
    if old_speaker is not None:
        registry._recount_segments(conn, old_speaker)
    return True
