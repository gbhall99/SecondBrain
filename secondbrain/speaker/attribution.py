"""Diarize a whole conversation and attribute its transcript segments.

Concatenate the conversation's chunk audio (in order), diarize the concatenation,
match each local speaker cluster to the global registry, then map diarization
turns back to per-chunk transcript segments by maximum temporal overlap. The
stored ``speaker_id`` is always a GLOBAL id, giving cross-chunk/cross-day
identity; ``speaker_confidence`` combines alignment overlap with acoustic match.

The audio-concatenation step (soundfile, Mac/`audio` extra) is injected via
``audio_builder`` so the attribution logic is unit-testable on CI with a fake
builder + ``MockDiarizer``.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline.diarize import DiarizationResult, Diarizer, SpeakerTurn
from secondbrain.speaker import registry
from secondbrain.storage import retention
from secondbrain.storage.db import transaction

log = logging.getLogger("secondbrain.attribution")


@dataclass
class ChunkOffset:
    audio_file_id: int
    concat_start_s: float
    duration_s: float
    row: sqlite3.Row


def best_overlap(turns: list[SpeakerTurn], s0: float, s1: float) -> tuple[str | None, float]:
    """Return (local_label, overlap_fraction) for the turn-label covering the
    segment [s0, s1] the most. Fraction is covered-seconds / segment-duration."""
    seg_dur = max(1e-6, s1 - s0)
    per_label: dict[str, float] = {}
    for t in turns:
        ov = max(0.0, min(s1, t.end_s) - max(s0, t.start_s))
        if ov > 0:
            per_label[t.local_label] = per_label.get(t.local_label, 0.0) + ov
    if not per_label:
        return None, 0.0
    label = max(per_label, key=per_label.get)
    return label, min(1.0, per_label[label] / seg_dur)


def _overlap_count(turns: list[SpeakerTurn], s0: float, s1: float) -> int:
    """How many distinct speaker labels overlap [s0, s1] (>1 ⇒ overlapped speech)."""
    return len({t.local_label for t in turns if min(s1, t.end_s) - max(s0, t.start_s) > 0})


def concat_offsets_from_db(
    conn: sqlite3.Connection, chunks: list[sqlite3.Row]
) -> list[ChunkOffset]:
    """Concat offsets derived from stored durations (used by the fake test builder)."""
    offsets: list[ChunkOffset] = []
    cur = 0.0
    for ch in chunks:
        dur = float(ch["duration_s"] or 0.0)
        offsets.append(ChunkOffset(int(ch["id"]), cur, dur, ch))
        cur += dur
    return offsets


def default_audio_builder(
    conn: sqlite3.Connection, chunks: list[sqlite3.Row], settings: Settings
) -> tuple[Path, list[ChunkOffset]]:
    """Concatenate chunk FLACs into one temp WAV; return (path, offsets)."""
    import numpy as np  # lazy
    import soundfile as sf  # lazy: `audio` extra

    offsets: list[ChunkOffset] = []
    parts = []
    cur = 0.0
    for ch in chunks:
        p = Path(ch["path"])
        if not p.exists():  # already swept by retention — skip
            continue
        audio, sr = sf.read(str(p))
        dur = len(audio) / float(sr)
        offsets.append(ChunkOffset(int(ch["id"]), cur, dur, ch))
        parts.append(audio)
        cur += dur
    settings.ensure_dirs()
    out = settings.audio_processed_dir / "conv_concat.wav"
    if parts:
        sf.write(str(out), np.concatenate(parts), settings.capture.sample_rate)
    return out, offsets


def _concat_to_chunk(offsets: list[ChunkOffset], t: float) -> ChunkOffset | None:
    for off in offsets:
        if off.concat_start_s <= t < off.concat_start_s + off.duration_s:
            return off
    return offsets[-1] if offsets else None


def attribute_conversation(
    conn: sqlite3.Connection,
    conversation_id: int,
    *,
    diarizer: Diarizer,
    settings: Settings | None = None,
    audio_builder=default_audio_builder,
) -> int:
    """Run diarization + attribution for one conversation. Returns segments labeled."""
    settings = settings or get_settings()

    chunks = conn.execute(
        "SELECT * FROM audio_files WHERE conversation_id=? ORDER BY started_at, id",
        (conversation_id,),
    ).fetchall()
    if not chunks:
        conn.execute("UPDATE conversations SET status='diarized' WHERE id=?", (conversation_id,))
        return 0

    conn.execute("UPDATE conversations SET status='diarizing' WHERE id=?", (conversation_id,))
    concat_path, offsets = audio_builder(conn, chunks, settings)
    # If any chunk audio is missing, the concat timeline no longer matches the
    # segments — skip rather than silently mislabel (offsets[-1] fallback).
    if len(offsets) < len(chunks):
        log.warning(
            "conversation %s: %d/%d chunks available; skipping diarization to avoid mislabeling",
            conversation_id, len(offsets), len(chunks),
        )
        _finalize(conn, conversation_id, chunks, settings)
        return 0

    diar: DiarizationResult = diarizer.diarize(concat_path)
    # All post-diarization DB writes are one atomic unit; the slow diarize() ran
    # above (outside the transaction) so no write lock is held during compute.
    with transaction(conn):
        return _attribute_and_finalize(conn, conversation_id, chunks, offsets, diar, settings)


def _attribute_and_finalize(
    conn: sqlite3.Connection,
    conversation_id: int,
    chunks: list[sqlite3.Row],
    offsets: list,
    diar: DiarizationResult,
    settings: Settings,
) -> int:
    d = settings.diarization
    # 1. Resolve each local cluster to a global speaker.
    cluster_speaker: dict[str, int] = {}
    cluster_sim: dict[str, float] = {}
    cluster_margin: dict[str, float] = {}
    cluster_obs: dict[str, int] = {}
    for cluster in diar.clusters:
        m = registry.match_embedding(conn, cluster.embedding, settings)
        if m.speaker_id is None:
            sid = registry.create_unknown_speaker(conn)
            sim = 0.0
        else:
            sid = registry.resolve_speaker_id(conn, m.speaker_id)
            sim = m.similarity
        cluster_speaker[cluster.local_label] = sid
        cluster_sim[cluster.local_label] = sim
        cluster_margin[cluster.local_label] = m.margin

        # record one observation per cluster (mapped back to a chunk) + centroid.
        rep = cluster.turns[0] if cluster.turns else SpeakerTurn(0.0, 0.0, cluster.local_label)
        off = _concat_to_chunk(offsets, rep.start_s)
        cluster_obs[cluster.local_label] = registry.record_observation(
            conn,
            speaker_id=sid,
            audio_file_id=off.audio_file_id if off else None,
            conversation_id=conversation_id,
            start_offset_s=(rep.start_s - off.concat_start_s) if off else rep.start_s,
            end_offset_s=(rep.end_s - off.concat_start_s) if off else rep.end_s,
            start_at=off.row["started_at"] if off else None,
            confidence=sim,
            embedding=cluster.embedding,
            duration_s=cluster.total_speech_s,
            quality=sim,
        )
        if m.speaker_id is None or sim >= d.centroid_update_threshold:
            registry.update_centroid(conn, sid, cluster.embedding)

    # 2. Align turns onto each chunk's transcript segments (concat timeline).
    labeled = 0
    per_speaker_count: dict[int, int] = {}
    per_speaker_last: dict[int, str] = {}
    for off in offsets:
        segs = conn.execute(
            "SELECT * FROM transcript_segments WHERE audio_file_id=?", (off.audio_file_id,)
        ).fetchall()
        for seg in segs:
            s0 = off.concat_start_s + float(seg["start_offset_s"])
            s1 = off.concat_start_s + float(seg["end_offset_s"])
            label, frac = best_overlap(diar.turns, s0, s1)
            if label is None or label not in cluster_speaker:
                continue
            sid = cluster_speaker[label]
            conf = round(frac * cluster_sim[label], 4)
            # flag overlapped speech (multiple speakers cover this segment)
            if settings.diarization.overlap_flag and _overlap_count(diar.turns, s0, s1) > 1:
                conf = min(conf, max(0.0, settings.diarization.low_confidence_threshold - 0.01))
            registry.assign_segment_speaker(
                conn, seg["id"], sid, conf,
                observation_id=cluster_obs.get(label), source="auto",
            )
            labeled += 1
            per_speaker_count[sid] = per_speaker_count.get(sid, 0) + 1
            if seg["start_at"]:
                per_speaker_last[sid] = max(per_speaker_last.get(sid, ""), seg["start_at"])

    for sid, cnt in per_speaker_count.items():
        registry.touch_speaker_stats(
            conn, sid, last_seen_at=per_speaker_last.get(sid), segments_added=cnt
        )

    # 3. Enforce per-speaker opt-out (redact opted-out speakers' segments).
    for sid in per_speaker_count:
        if registry.is_opted_out(conn, sid, settings):
            registry.redact_speaker_segments(conn, sid)

    _finalize(conn, conversation_id, chunks, settings)
    return labeled


def _finalize(
    conn: sqlite3.Connection, conversation_id: int, chunks: list[sqlite3.Row], settings: Settings
) -> None:
    """Mark diarized and NOW set the raw-audio retention deadline on the chunks."""
    delete_after = retention.compute_delete_after(settings)
    for ch in chunks:
        conn.execute(
            "UPDATE audio_files SET retention_delete_after=? WHERE id=? AND status='transcribed'",
            (delete_after, ch["id"]),
        )
    conn.execute("UPDATE conversations SET status='diarized' WHERE id=?", (conversation_id,))

    # Hand off to knowledge extraction (Phase 3), if enabled.
    if settings.extraction.enabled:
        from secondbrain.knowledge.extract import enqueue_extraction

        enqueue_extraction(conn, conversation_id)
