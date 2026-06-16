"""Transcription behind a backend interface.

The real backends (Parakeet / Whisper via MLX) only run on Apple Silicon and are
imported lazily, so this module imports cleanly on Linux/CI where the
``MockTranscriber`` stands in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from secondbrain.config import Settings, get_settings


@dataclass
class TranscribedSegment:
    start_offset_s: float
    end_offset_s: float
    text: str
    confidence: float | None = None


@dataclass
class TranscriptionResult:
    segments: list[TranscribedSegment]
    backend: str
    model: str | None = None
    language: str | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


class Transcriber(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        ...


class MockTranscriber(Transcriber):
    """Deterministic transcriber for CI/dev on non-Apple hardware.

    Returns ``scripted`` segments if provided; otherwise emits a single
    placeholder segment so the full pipeline can be exercised end-to-end.
    """

    backend_name = "mock"

    def __init__(self, scripted: list[TranscribedSegment] | None = None):
        self.scripted = scripted

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        segments = self.scripted if self.scripted is not None else [
            TranscribedSegment(0.0, 1.0, f"[mock transcript of {Path(audio_path).name}]", 1.0)
        ]
        return TranscriptionResult(
            segments=list(segments),
            backend=self.backend_name,
            model="mock",
            language=language or "en",
        )


class WhisperMLXTranscriber(Transcriber):
    """OpenAI Whisper (large-v3-turbo) via mlx-whisper. Apple Silicon only."""

    backend_name = "whisper"

    def __init__(self, model: str):
        self.model = model

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        import mlx_whisper  # lazy: Apple Silicon only

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model,
            language=language or None,
            word_timestamps=False,
        )
        segments = [
            TranscribedSegment(
                start_offset_s=float(s["start"]),
                end_offset_s=float(s["end"]),
                text=str(s["text"]).strip(),
                confidence=_avg_logprob_to_conf(s.get("avg_logprob")),
            )
            for s in result.get("segments", [])
        ]
        return TranscriptionResult(
            segments=segments,
            backend=self.backend_name,
            model=self.model,
            language=result.get("language", language),
        )


class ParakeetMLXTranscriber(Transcriber):
    """NVIDIA Parakeet via parakeet-mlx. Apple Silicon only; strong English WER."""

    backend_name = "parakeet"

    def __init__(self, model: str):
        self.model = model
        self._model_obj = None

    def _ensure_model(self):
        if self._model_obj is None:
            from parakeet_mlx import from_pretrained  # lazy: Apple Silicon only

            self._model_obj = from_pretrained(self.model)
        return self._model_obj

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        model = self._ensure_model()
        result = model.transcribe(str(audio_path))
        segments = [
            TranscribedSegment(
                start_offset_s=float(getattr(s, "start", 0.0)),
                end_offset_s=float(getattr(s, "end", 0.0)),
                text=str(getattr(s, "text", "")).strip(),
                confidence=None,
            )
            for s in getattr(result, "sentences", [])
        ]
        if not segments and getattr(result, "text", ""):
            segments = [TranscribedSegment(0.0, 0.0, result.text.strip(), None)]
        return TranscriptionResult(
            segments=segments,
            backend=self.backend_name,
            model=self.model,
            language=language or "en",
        )


def _avg_logprob_to_conf(avg_logprob) -> float | None:
    if avg_logprob is None:
        return None
    import math

    return round(math.exp(float(avg_logprob)), 4)


def get_transcriber(settings: Settings | None = None) -> Transcriber:
    """Factory selecting the configured backend."""
    settings = settings or get_settings()
    backend = settings.transcription.backend.lower()
    if backend == "mock":
        return MockTranscriber()
    if backend == "whisper":
        return WhisperMLXTranscriber(settings.transcription.whisper_model)
    if backend == "parakeet":
        return ParakeetMLXTranscriber(settings.transcription.parakeet_model)
    raise ValueError(f"Unknown transcription backend: {backend!r}")
