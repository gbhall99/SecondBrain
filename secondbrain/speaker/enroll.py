"""Owner voice enrollment.

Reuses the diarizer to embed clean enrollment clips: diarize a single-speaker
clip, take the dominant cluster's speaker embedding, fold it into the owner's
profile centroid. Works with MockDiarizer on CI (deterministic embeddings).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline.diarize import Diarizer, get_diarizer
from secondbrain.speaker import registry


def enroll_owner_from_files(
    conn: sqlite3.Connection,
    files: list[Path],
    *,
    diarizer: Diarizer | None = None,
    settings: Settings | None = None,
    name: str = "Me",
    replace: bool = False,
) -> int:
    """Enroll the owner from one or more clean voice clips. Returns owner id.

    Quality gate: the clips must contain at least
    ``[diarization] min_enroll_speech_s`` seconds of speech in total — a profile
    seeded from a couple of seconds mislabels everyone. Raises ``ValueError``
    (before any DB write) when they don't.

    ``replace=True`` re-enrolls from scratch: prior enrollment exemplars are
    dropped (user corrections are kept) so a bad first enrollment can't keep
    steering the owner profile.
    """
    settings = settings or get_settings()
    diarizer = diarizer or get_diarizer(settings)
    # Embed every clip FIRST so a failed quality gate writes nothing.
    clusters = []
    for f in files:
        result = diarizer.diarize(Path(f))
        if not result.clusters:
            continue
        clusters.append(max(result.clusters, key=lambda c: c.total_speech_s))
    total_speech_s = sum(c.total_speech_s for c in clusters)
    min_speech = settings.diarization.min_enroll_speech_s
    if total_speech_s < min_speech:
        raise ValueError(
            f"enrollment clips contain only {total_speech_s:.1f}s of speech; "
            f"at least {min_speech:.0f}s is required — record longer clips"
        )
    owner_id = registry.get_or_create_owner(conn, name)
    if replace:
        # Enrollment-shaped rows only (no audio/conversation provenance and not
        # a user correction) — includes legacy rows recorded as source='auto'.
        conn.execute(
            "DELETE FROM speaker_observations WHERE speaker_id=? "
            "AND source IN ('enroll', 'auto') "
            "AND audio_file_id IS NULL AND conversation_id IS NULL",
            (owner_id,),
        )
    for cluster in clusters:
        registry.record_observation(
            conn,
            speaker_id=owner_id,
            audio_file_id=None,
            conversation_id=None,
            start_offset_s=0.0,
            end_offset_s=cluster.total_speech_s,
            start_at=None,
            confidence=1.0,
            embedding=cluster.embedding,
            source="enroll",
        )
    # Ensure the centroid reflects all enrollment exemplars.
    registry.recompute_centroid(conn, owner_id)
    return owner_id


def record_clip(path: Path, seconds: float, settings: Settings) -> Path:
    """Record a fixed-length mono clip from the configured mic (Mac/`audio`)."""
    import sounddevice as sd  # lazy
    import soundfile as sf  # lazy

    from secondbrain.capture.devices import resolve_device

    cfg = settings.capture
    device = resolve_device(cfg.input_device)
    frames = int(cfg.sample_rate * seconds)
    audio = sd.rec(frames, samplerate=cfg.sample_rate, channels=1, dtype="float32", device=device)
    sd.wait()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, cfg.sample_rate, format="FLAC")
    return path
