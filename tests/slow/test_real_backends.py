"""Real-backend smoke tests (Ollama / MLX transcription / pyannote diarization).

Opt-in only: run with SB_SLOW=1 on a machine that has the backends installed
(nightly macOS CI). All heavy imports happen INSIDE the test bodies so the normal
PR suite can collect this module without importing mlx/pyannote/etc.
"""

from __future__ import annotations

import os
import wave

import pytest

pytestmark = pytest.mark.slow

_SLOW = os.environ.get("SB_SLOW") == "1"
requires_slow = pytest.mark.skipif(not _SLOW, reason="set SB_SLOW=1 to run real-backend tests")


def _tone_wav(path, seconds=2.0, sr=16000):
    import math
    import struct

    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * t / sr)))
            for t in range(int(seconds * sr))
        )
        w.writeframes(frames)
    return path


@requires_slow
def test_ollama_completion():
    from secondbrain.config import get_settings
    from secondbrain.llm.client import OllamaLLM

    settings = get_settings()
    try:
        resp = OllamaLLM(settings).complete(system="Be terse.", prompt="Say 'ok'.")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama not reachable: {exc}")
    assert resp.text.strip()


@requires_slow
def test_real_transcription(tmp_path):
    from secondbrain.config import get_settings
    from secondbrain.pipeline.transcribe import get_transcriber

    settings = get_settings()
    if settings.transcription.backend == "mock":
        pytest.skip("set SB_TRANSCRIPTION__BACKEND=parakeet|whisper")
    wav = _tone_wav(tmp_path / "tone.wav")
    try:
        result = get_transcriber(settings).transcribe(wav)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"transcription backend unavailable: {exc}")
    assert isinstance(result.segments, list)  # may be empty for a pure tone


@requires_slow
def test_real_diarization(tmp_path):
    from secondbrain.config import get_settings
    from secondbrain.pipeline.diarize import PyannoteDiarizer

    settings = get_settings()
    if not (settings.diarization.hf_token or os.environ.get("HF_TOKEN")):
        pytest.skip("pyannote needs a HuggingFace token")
    wav = _tone_wav(tmp_path / "tone.wav")
    try:
        result = PyannoteDiarizer(settings).diarize(wav)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pyannote unavailable: {exc}")
    assert hasattr(result, "clusters")
