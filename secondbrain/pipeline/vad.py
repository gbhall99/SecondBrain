"""Voice-activity detection so we never transcribe (or keep) silence.

Real detection uses silero-vad (lazy import). On CI/non-Apple hardware the
``MockVad`` reports speech across the whole clip so the pipeline still runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from secondbrain.config import Settings, get_settings


@dataclass
class SpeechSpan:
    start_s: float
    end_s: float


@dataclass
class VadResult:
    has_speech: bool
    spans: list[SpeechSpan]

    @property
    def speech_seconds(self) -> float:
        return sum(s.end_s - s.start_s for s in self.spans)


class Vad(ABC):
    @abstractmethod
    def detect(self, audio_path: Path) -> VadResult:
        ...


class MockVad(Vad):
    """Always reports speech (whole-file span). For CI/dev."""

    def __init__(self, has_speech: bool = True, duration_s: float = 1.0):
        self._has_speech = has_speech
        self._duration_s = duration_s

    def detect(self, audio_path: Path) -> VadResult:
        if not self._has_speech:
            return VadResult(has_speech=False, spans=[])
        return VadResult(has_speech=True, spans=[SpeechSpan(0.0, self._duration_s)])


class SileroVad(Vad):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._utils = None

    def _ensure(self):
        if self._model is None:
            from silero_vad import (  # lazy
                get_speech_timestamps,
                load_silero_vad,
                read_audio,
            )

            self._model = load_silero_vad()
            self._utils = (get_speech_timestamps, read_audio)
        return self._model, self._utils

    def detect(self, audio_path: Path) -> VadResult:
        model, (get_speech_timestamps, read_audio) = self._ensure()
        sr = self.settings.capture.sample_rate
        wav = read_audio(str(audio_path), sampling_rate=sr)
        cfg = self.settings.vad
        ts = get_speech_timestamps(
            wav,
            model,
            sampling_rate=sr,
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_silence_ms,
            return_seconds=True,
        )
        spans = [SpeechSpan(float(t["start"]), float(t["end"])) for t in ts]
        return VadResult(has_speech=bool(spans), spans=spans)


def get_vad(settings: Settings | None = None) -> Vad:
    settings = settings or get_settings()
    if not settings.vad.enabled:
        return MockVad(has_speech=True)
    return SileroVad(settings)
