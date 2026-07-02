"""Speaker diarization behind a backend interface.

Mirrors ``pipeline/vad.py`` / ``pipeline/transcribe.py``: the real backend
(pyannote.audio 3.1) is Apple-Silicon/heavy and lazily imported, so this module
imports cleanly on Linux/CI where ``MockDiarizer`` stands in. A diarizer returns
both "who spoke when" (turns) and a per-local-cluster speaker embedding, so the
registry can resolve local clusters to global identities.
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from secondbrain.config import Settings, get_settings


def _load_pipeline(pipeline_cls, model: str, token: str | None):
    """Load a pyannote Pipeline across API versions.

    pyannote.audio renamed the auth argument ``use_auth_token`` → ``token`` in
    newer releases, so pin the right keyword from the actual signature (falling
    back to a retry) instead of hard-coding one and crashing on the other.
    """
    import inspect

    try:
        params = inspect.signature(pipeline_cls.from_pretrained).parameters
    except (TypeError, ValueError):
        params = {}
    if "use_auth_token" in params and "token" not in params:
        return pipeline_cls.from_pretrained(model, use_auth_token=token)
    try:
        return pipeline_cls.from_pretrained(model, token=token)
    except TypeError:
        return pipeline_cls.from_pretrained(model, use_auth_token=token)


def _unpack_diarize(out):
    """The 3.x pipeline returns ``(diarization, embeddings)`` when called with
    ``return_embeddings=True``. pyannote.audio 4.x changed this (ignores the kwarg
    and returns a single ``DiarizeOutput``), which we don't support yet — surface a
    clear, actionable error instead of a cryptic unpack failure."""
    if isinstance(out, tuple) and len(out) == 2:
        return out
    raise RuntimeError(
        "Unsupported pyannote.audio pipeline output (got "
        f"{type(out).__name__}). SecondBrain needs the 3.x diarization API — "
        "install it with:  pip install 'pyannote.audio>=3.1,<4'"
    )


def _shim_torchaudio_metadata() -> None:
    """Make pyannote.audio 3.x importable on torchaudio >= 2.9.

    pyannote's ``core/io.py`` references ``torchaudio.AudioMetaData`` at import
    time (an eagerly-evaluated return-type annotation). torchaudio removed that
    class in 2.9+, so ``from pyannote.audio import Pipeline`` raises
    ``AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'``.

    On our diarization/enrollment path the class is only used as an annotation
    (``torchaudio.info()`` returns its own metadata object), so a lightweight
    stand-in is enough to let the import succeed. No-op when torchaudio still
    ships the real class.
    """
    import torchaudio

    if hasattr(torchaudio, "AudioMetaData"):
        return

    from dataclasses import dataclass as _dataclass

    @_dataclass
    class AudioMetaData:  # minimal stand-in matching torchaudio.info() fields
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = "UNKNOWN"

    torchaudio.AudioMetaData = AudioMetaData


def _shim_hf_hub_use_auth_token() -> None:
    """Let pyannote.audio 3.x download gated models on huggingface_hub >= 1.0.

    pyannote 3.x calls ``huggingface_hub.hf_hub_download(use_auth_token=...)``
    throughout (models, pipeline configs), but huggingface_hub 1.0 removed that
    argument (renamed to ``token``), so downloads raise
    ``TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'``.

    We can't simply pin ``huggingface_hub<1.0`` because ``transformers>=5``
    (pulled in by the MLX/embedding stack) requires ``>=1.5``. Instead, wrap the
    single download entry point that every pyannote/speechbrain call funnels
    through and translate the deprecated kwarg. Idempotent; must run before
    ``from pyannote.audio import ...`` so pyannote binds the wrapped function.
    """
    import huggingface_hub as hf

    fn = getattr(hf, "hf_hub_download", None)
    if fn is None or getattr(fn, "_sb_auth_shim", False):
        return

    import functools

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        if "use_auth_token" in kwargs:
            token = kwargs.pop("use_auth_token")
            kwargs.setdefault("token", token)
        return fn(*args, **kwargs)

    _wrapped._sb_auth_shim = True
    hf.hf_hub_download = _wrapped


@dataclass
class SpeakerTurn:
    start_s: float
    end_s: float
    local_label: str


@dataclass
class LocalCluster:
    local_label: str
    embedding: list[float]              # L2-normalized speaker embedding
    turns: list[SpeakerTurn] = field(default_factory=list)
    total_speech_s: float = 0.0


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    clusters: list[LocalCluster]
    backend: str
    model: str | None = None


def deterministic_embedding(seed: str, dim: int = 256) -> list[float]:
    """A stable, L2-normalized pseudo-embedding derived from ``seed``.

    Used by MockDiarizer/tests so the full match→cluster→relabel path can run
    deterministically with no ML dependency.
    """
    raw = bytearray()
    counter = 0
    while len(raw) < dim * 4:
        raw += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    vals = [((raw[i] / 255.0) * 2.0 - 1.0) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


class Diarizer(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    def diarize(self, audio_path: Path) -> DiarizationResult:
        ...


class MockDiarizer(Diarizer):
    """Deterministic diarizer for CI/dev. Emits scripted turns + embeddings.

    Default: a single speaker ``SPEAKER_00`` over [0, duration_s] with a
    deterministic embedding seeded by the file name, so the same audio always
    maps to the same voice.
    """

    backend_name = "mock"

    def __init__(
        self,
        turns: list[tuple[float, float, str]] | None = None,
        embeddings: dict[str, list[float]] | None = None,
        duration_s: float = 2.0,
        dim: int = 256,
    ):
        self._turns = turns
        self._embeddings = embeddings
        self._duration_s = duration_s
        self._dim = dim

    def diarize(self, audio_path: Path) -> DiarizationResult:
        if self._turns is None:
            turns = [SpeakerTurn(0.0, self._duration_s, "SPEAKER_00")]
        else:
            turns = [SpeakerTurn(s, e, lbl) for s, e, lbl in self._turns]

        clusters: dict[str, LocalCluster] = {}
        for t in turns:
            c = clusters.get(t.local_label)
            if c is None:
                emb = (self._embeddings or {}).get(t.local_label) or deterministic_embedding(
                    f"{Path(audio_path).name}:{t.local_label}", self._dim
                )
                c = LocalCluster(local_label=t.local_label, embedding=emb)
                clusters[t.local_label] = c
            c.turns.append(t)
            c.total_speech_s += max(0.0, t.end_s - t.start_s)

        return DiarizationResult(
            turns=turns, clusters=list(clusters.values()), backend=self.backend_name, model="mock"
        )


class PyannoteDiarizer(Diarizer):
    """pyannote.audio 3.1 speaker diarization. Apple Silicon (MPS) / CPU."""

    backend_name = "pyannote"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.diarization.model
        self._pipeline = None

    def _token(self) -> str | None:
        return (
            self.settings.diarization.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or None
        )

    def _ensure(self):
        if self._pipeline is None:
            # Keep model cache inside the project data dir for a self-contained setup.
            os.environ.setdefault("HF_HOME", str(self.settings.models_dir))
            import torch  # lazy

            _shim_torchaudio_metadata()  # pyannote 3.x needs torchaudio.AudioMetaData
            _shim_hf_hub_use_auth_token()  # pyannote 3.x passes the removed use_auth_token kwarg
            from pyannote.audio import Pipeline  # lazy: heavy, gated models

            self._pipeline = _load_pipeline(Pipeline, self.model, self._token())
            if torch.backends.mps.is_available():
                self._pipeline.to(torch.device("mps"))
        return self._pipeline

    def diarize(self, audio_path: Path) -> DiarizationResult:
        import numpy as np  # lazy

        pipeline = self._ensure()
        diarization, embeddings = _unpack_diarize(
            pipeline(str(audio_path), return_embeddings=True)
        )

        # `embeddings` is an (n_local_speakers, dim) array aligned to the sorted
        # local labels of the annotation.
        labels = diarization.labels()
        turns: list[SpeakerTurn] = []
        per_label_turns: dict[str, list[SpeakerTurn]] = {lbl: [] for lbl in labels}
        for segment, _, label in diarization.itertracks(yield_label=True):
            t = SpeakerTurn(float(segment.start), float(segment.end), str(label))
            turns.append(t)
            per_label_turns.setdefault(label, []).append(t)
        turns.sort(key=lambda t: t.start_s)

        clusters: list[LocalCluster] = []
        for idx, label in enumerate(labels):
            vec = embeddings[idx]
            arr = np.asarray(vec, dtype="float32")
            norm = float(np.linalg.norm(arr)) or 1.0
            emb = (arr / norm).tolist()
            lt = per_label_turns.get(label, [])
            clusters.append(
                LocalCluster(
                    local_label=str(label),
                    embedding=emb,
                    turns=lt,
                    total_speech_s=sum(t.end_s - t.start_s for t in lt),
                )
            )

        return DiarizationResult(
            turns=turns, clusters=clusters, backend=self.backend_name, model=self.model
        )


def get_diarizer(settings: Settings | None = None) -> Diarizer:
    settings = settings or get_settings()
    if not settings.diarization.enabled:
        return MockDiarizer(dim=settings.diarization.embedding_dim)
    backend = settings.diarization.backend.lower()
    if backend == "mock":
        return MockDiarizer(dim=settings.diarization.embedding_dim)
    if backend == "pyannote":
        return PyannoteDiarizer(settings)
    raise ValueError(f"Unknown diarization backend: {backend!r}")
