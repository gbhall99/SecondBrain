from pathlib import Path

from secondbrain.pipeline.diarize import MockDiarizer, _load_pipeline, deterministic_embedding
from secondbrain.speaker.attribution import best_overlap


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
