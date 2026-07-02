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
) -> int:
    """Enroll the owner from one or more clean voice clips. Returns owner id."""
    settings = settings or get_settings()
    diarizer = diarizer or get_diarizer(settings)
    owner_id = registry.get_or_create_owner(conn, name)
    for f in files:
        result = diarizer.diarize(Path(f))
        if not result.clusters:
            continue
        cluster = max(result.clusters, key=lambda c: c.total_speech_s)
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
        )
        registry.update_centroid(conn, owner_id, cluster.embedding)
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
