from pathlib import Path

import pytest

from secondbrain.pipeline.diarize import (
    MockDiarizer,
    _apply_segmentation_threshold,
    _load_pipeline,
    _shim_hf_hub_use_auth_token,
    _shim_torchaudio_metadata,
    _speaker_count_kwargs,
    _trusted_torch_load,
    _unpack_diarize,
    deterministic_embedding,
)
from secondbrain.speaker.attribution import best_overlap


def test_unpack_diarize_tuple_ok():
    assert _unpack_diarize(("annotation", "embeddings")) == ("annotation", "embeddings")


def test_unpack_diarize_rejects_pyannote4_output():
    class DiarizeOutput:  # what pyannote.audio 4.x returns (not a 2-tuple)
        pass

    with pytest.raises(RuntimeError, match="pyannote.audio>=3.1,<4"):
        _unpack_diarize(DiarizeOutput())


def test_load_pipeline_supports_both_token_kwargs():
    # Newer pyannote: from_pretrained(model, *, token=...)
    class NewPipeline:
        @classmethod
        def from_pretrained(cls, model, *, token=None):
            return ("new", model, token)

    # Older pyannote: from_pretrained(model, *, use_auth_token=...)
    class OldPipeline:
        @classmethod
        def from_pretrained(cls, model, *, use_auth_token=None):
            return ("old", model, use_auth_token)

    assert _load_pipeline(NewPipeline, "m", "tok") == ("new", "m", "tok")
    assert _load_pipeline(OldPipeline, "m", "tok") == ("old", "m", "tok")


def test_shim_torchaudio_metadata_adds_missing_attr(monkeypatch):
    # torchaudio >= 2.9 dropped AudioMetaData, which pyannote.audio 3.x references
    # at import time. The shim must define a stand-in so the import succeeds.
    import sys
    import types

    fake = types.ModuleType("torchaudio")  # no AudioMetaData attribute
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    assert not hasattr(fake, "AudioMetaData")

    _shim_torchaudio_metadata()

    assert hasattr(fake, "AudioMetaData")
    md = fake.AudioMetaData(sample_rate=16000, num_frames=100, num_channels=1)
    assert md.sample_rate == 16000 and md.num_channels == 1


def test_shim_torchaudio_metadata_noop_when_present(monkeypatch):
    # When torchaudio still ships the class, the shim must leave it untouched.
    import sys
    import types

    fake = types.ModuleType("torchaudio")
    sentinel = object()
    fake.AudioMetaData = sentinel
    monkeypatch.setitem(sys.modules, "torchaudio", fake)

    _shim_torchaudio_metadata()

    assert fake.AudioMetaData is sentinel


def test_shim_hf_hub_translates_use_auth_token(monkeypatch):
    # pyannote 3.x passes hf_hub_download(use_auth_token=...), removed in
    # huggingface_hub 1.0; the shim must translate it to the new 'token' kwarg.
    import sys
    import types

    seen = {}

    def fake_download(*args, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return "cached/path"

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    _shim_hf_hub_use_auth_token()
    fake_hf.hf_hub_download(repo_id="x", use_auth_token="tok")

    assert "use_auth_token" not in seen
    assert seen.get("token") == "tok"


def test_shim_hf_hub_is_idempotent(monkeypatch):
    # Applying the shim twice must not double-wrap the download function.
    import sys
    import types

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda *a, **k: "p"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    _shim_hf_hub_use_auth_token()
    wrapped_once = fake_hf.hf_hub_download
    _shim_hf_hub_use_auth_token()

    assert fake_hf.hf_hub_download is wrapped_once


def test_trusted_torch_load_defaults_weights_only_false_and_restores(monkeypatch):
    # PyTorch 2.6 defaults torch.load(weights_only=True), which rejects pyannote
    # checkpoints. The context manager must default it to False inside its scope
    # and restore the original torch.load afterwards.
    import sys
    import types

    seen = {}

    def fake_load(*args, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return "ckpt"

    fake_torch = types.ModuleType("torch")
    fake_torch.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    original = fake_torch.load

    with _trusted_torch_load():
        fake_torch.load("model.ckpt")
        assert seen.get("weights_only") is False
        # Must override an explicit True/None too: pyannote's loader passes
        # weights_only=None explicitly, which torch>=2.6 treats as True.
        fake_torch.load("model.ckpt", weights_only=True)
        assert seen.get("weights_only") is False
        fake_torch.load("model.ckpt", weights_only=None)
        assert seen.get("weights_only") is False

    assert fake_torch.load is original  # restored on exit


class _FakePipeline:
    """Stands in for a pyannote pipeline: apply() signature + hyper-params."""

    def __init__(self, params=None):
        self._params = params if params is not None else {}
        self.instantiated_with = None

    def apply(self, file, num_speakers=None, min_speakers=None, max_speakers=None,
              return_embeddings=False):
        return ("diar", "emb")

    def parameters(self, instantiated=False):
        return self._params

    def instantiate(self, params):
        self.instantiated_with = params


def _dcfg(settings, **over):
    for k, v in over.items():
        setattr(settings.diarization, k, v)
    return settings.diarization


def test_speaker_count_kwargs_forwards_supported_hints(settings):
    p = _FakePipeline()
    assert _speaker_count_kwargs(p, _dcfg(settings, min_speakers=0, max_speakers=0)) == {}
    assert _speaker_count_kwargs(p, _dcfg(settings, min_speakers=2, max_speakers=4)) == {
        "min_speakers": 2, "max_speakers": 4,
    }
    # equal non-zero min/max collapse to the stronger num_speakers hint
    assert _speaker_count_kwargs(p, _dcfg(settings, min_speakers=3, max_speakers=3)) == {
        "num_speakers": 3,
    }


def test_speaker_count_kwargs_skips_unsupported_params(settings):
    class NoHints:
        def apply(self, file, return_embeddings=False):  # older/other API
            return ("diar", "emb")

    d = _dcfg(settings, min_speakers=2, max_speakers=4)
    assert _speaker_count_kwargs(NoHints(), d) == {}


def test_apply_segmentation_threshold_when_exposed(settings):
    p = _FakePipeline(params={"segmentation": {"threshold": 0.5, "min_duration_off": 0.0}})
    assert _apply_segmentation_threshold(p, 0.7) is True
    assert p.instantiated_with["segmentation"]["threshold"] == 0.7
    assert p.instantiated_with["segmentation"]["min_duration_off"] == 0.0  # preserved


def test_apply_segmentation_threshold_noop_when_absent(settings):
    # pyannote 3.1's powerset segmentation exposes no threshold — must not crash
    p = _FakePipeline(params={"segmentation": {"min_duration_off": 0.0}})
    assert _apply_segmentation_threshold(p, 0.7) is False
    assert p.instantiated_with is None

    class NoParams:
        pass

    assert _apply_segmentation_threshold(NoParams(), 0.7) is False


def test_deterministic_embedding_is_stable_and_normalized():
    a = deterministic_embedding("alice", 16)
    b = deterministic_embedding("alice", 16)
    c = deterministic_embedding("bob", 16)
    assert a == b                       # stable
    assert a != c                       # distinct seeds differ
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6  # L2-normalized


def test_mock_diarizer_default_single_speaker():
    res = MockDiarizer(dim=16).diarize(Path("/tmp/x.flac"))
    assert len(res.clusters) == 1
    assert res.turns[0].local_label == "SPEAKER_00"


def test_mock_diarizer_scripted_clusters():
    res = MockDiarizer(
        turns=[(0.0, 2.0, "S0"), (2.0, 4.0, "S1"), (4.0, 5.0, "S0")], dim=8
    ).diarize(Path("/tmp/x.flac"))
    labels = {c.local_label for c in res.clusters}
    assert labels == {"S0", "S1"}
    s0 = next(c for c in res.clusters if c.local_label == "S0")
    assert s0.total_speech_s == 3.0     # 2.0 + 1.0


def test_best_overlap_picks_dominant_speaker():
    from secondbrain.pipeline.diarize import SpeakerTurn

    turns = [SpeakerTurn(0.0, 1.0, "A"), SpeakerTurn(1.0, 4.0, "B")]
    label, frac = best_overlap(turns, 1.0, 3.0)   # fully inside B
    assert label == "B" and frac == 1.0
    label, frac = best_overlap(turns, 0.0, 2.0)   # 1s A + 1s B → tie broken by max
    assert label in {"A", "B"} and 0.0 < frac <= 1.0


def test_best_overlap_no_overlap_returns_none():
    from secondbrain.pipeline.diarize import SpeakerTurn

    label, frac = best_overlap([SpeakerTurn(0.0, 1.0, "A")], 5.0, 6.0)
    assert label is None and frac == 0.0
