"""Transcription worker: drains ``transcribe`` jobs from the queue.

For each recorded audio chunk: run VAD (skip silence), transcribe, persist the
transcript + segments (with absolute wall-clock timestamps and provenance back
to the audio file + offset), index for semantic search, and set the raw-audio
retention deadline.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import conversation
from secondbrain.pipeline import queue as q
from secondbrain.pipeline.diarize import Diarizer, get_diarizer
from secondbrain.pipeline.transcribe import Transcriber, get_transcriber
from secondbrain.pipeline.vad import Vad, get_vad
from secondbrain.search import semantic
from secondbrain.speaker import attribution
from secondbrain.storage import models, retention

JOB_TRANSCRIBE = "transcribe"
JOB_CLUSTER = "cluster_speakers"


def enqueue_transcription(conn: sqlite3.Connection, audio_file_id: int) -> int | None:
    return q.enqueue(
        conn,
        JOB_TRANSCRIBE,
        {"audio_file_id": audio_file_id},
        dedupe_key="audio_file_id",
    )


def _abs_start(base_started_at: str, offset_s: float) -> str:
    return models.iso_from_dt(models.parse_iso(base_started_at) + timedelta(seconds=offset_s))


def process_audio_file(
    conn: sqlite3.Connection,
    audio_file_id: int,
    *,
    transcriber: Transcriber,
    vad: Vad,
    settings: Settings,
) -> int:
    """Transcribe one audio file. Returns number of segments stored."""
    from pathlib import Path

    row = models.get_audio_file(conn, audio_file_id)
    if row is None:
        return 0
    audio_path = Path(row["path"])
    delete_after = retention.compute_delete_after(settings)

    # 1. VAD gate — never transcribe (or keep) silence.
    if settings.vad.enabled and audio_path.exists():
        vres = vad.detect(audio_path)
        if not vres.has_speech:
            conn.execute(
                "UPDATE audio_files SET has_speech=0, status='transcribed', "
                "retention_delete_after=? WHERE id=?",
                (delete_after, audio_file_id),
            )
            return 0

    # 2. Transcribe.
    models.set_audio_status(conn, audio_file_id, "transcribing")
    result = transcriber.transcribe(audio_path, language=settings.transcription.language or None)
    transcript_id = models.insert_transcript(
        conn, audio_file_id, result.backend, result.model, result.language
    )

    # 3. Persist segments with absolute timestamps + provenance.
    segs = [
        models.Segment(
            transcript_id=transcript_id,
            audio_file_id=audio_file_id,
            start_offset_s=s.start_offset_s,
            end_offset_s=s.end_offset_s,
            text=s.text,
            start_at=_abs_start(row["started_at"], s.start_offset_s),
            confidence=s.confidence,
        )
        for s in result.segments
        if s.text.strip()
    ]
    if segs:
        models.insert_segments(conn, segs)

    # 4. Mark transcribed. When diarization is enabled, DEFER the retention
    #    deadline (NULL) so the raw audio survives until the chunk's conversation
    #    is diarized; attribution sets the deadline afterward.
    deferred = None if settings.diarization.enabled else delete_after
    conn.execute(
        "UPDATE audio_files SET has_speech=1, status='transcribed', "
        "retention_delete_after=? WHERE id=?",
        (deferred, audio_file_id),
    )

    # 4b. Group the chunk into a conversation (diarized as a whole later).
    if settings.diarization.enabled:
        try:
            conversation.assign_chunk(conn, audio_file_id, settings)
        except Exception:  # noqa: BLE001 - never block transcription on this
            pass

    # 5. Best-effort semantic indexing (no-op if unavailable).
    if segs:
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM transcript_segments WHERE transcript_id=? ORDER BY id",
                (transcript_id,),
            ).fetchall()
        ]
        try:
            semantic.index_segments(conn, ids, [s.text for s in segs], settings)
        except Exception:
            pass  # semantic search is optional; full-text still works

    return len(segs)


def run_once(
    conn: sqlite3.Connection,
    *,
    transcriber: Transcriber | None = None,
    vad: Vad | None = None,
    diarizer: Diarizer | None = None,
    settings: Settings | None = None,
) -> bool:
    """Claim and process a single job of any type. Returns True if one ran."""
    settings = settings or get_settings()
    job = q.claim_next(conn)
    if job is None:
        return False
    try:
        if job.type == JOB_TRANSCRIBE:
            process_audio_file(
                conn,
                int(job.payload["audio_file_id"]),
                transcriber=transcriber or get_transcriber(settings),
                vad=vad or get_vad(settings),
                settings=settings,
            )
        elif job.type == conversation.JOB_DIARIZE:
            attribution.attribute_conversation(
                conn,
                int(job.payload["conversation_id"]),
                diarizer=diarizer or get_diarizer(settings),
                settings=settings,
            )
        elif job.type == JOB_CLUSTER:
            from secondbrain.speaker import cluster

            cluster.run_clustering(conn, settings=settings)
        else:
            raise ValueError(f"unknown job type {job.type!r}")
        q.complete(conn, job.id)
    except Exception as exc:  # noqa: BLE001 - record and let queue retry
        if job.type == JOB_TRANSCRIBE:
            models.set_audio_status(conn, int(job.payload.get("audio_file_id", 0)), "failed")
        elif job.type == conversation.JOB_DIARIZE:
            conn.execute(
                "UPDATE conversations SET status='failed' WHERE id=?",
                (int(job.payload.get("conversation_id", 0)),),
            )
        q.fail(conn, job, repr(exc))
    return True


def drain(
    conn: sqlite3.Connection,
    *,
    transcriber: Transcriber | None = None,
    vad: Vad | None = None,
    diarizer: Diarizer | None = None,
    settings: Settings | None = None,
    max_jobs: int | None = None,
) -> int:
    """Process queued jobs until empty (or ``max_jobs`` reached). Returns count."""
    settings = settings or get_settings()
    transcriber = transcriber or get_transcriber(settings)
    vad = vad or get_vad(settings)
    processed = 0
    while max_jobs is None or processed < max_jobs:
        if not run_once(
            conn, transcriber=transcriber, vad=vad, diarizer=diarizer, settings=settings
        ):
            break
        processed += 1
    return processed
