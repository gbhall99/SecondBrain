"""Config validation — misconfiguration fails fast with a clear message."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from secondbrain.config import (
    ApiConfig,
    DiarizationConfig,
    LLMConfig,
    LoggingConfig,
    ProactiveConfig,
    Settings,
    TasksConfig,
    TranscriptionConfig,
)


def test_valid_config_accepted():
    s = Settings()
    assert s.transcription.backend in {"parakeet", "whisper", "mock"}
    assert 1 <= s.api.port <= 65535


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TranscriptionConfig(backend="bogus"),
        lambda: DiarizationConfig(backend="bogus"),
        lambda: LLMConfig(backend="bogus"),
        lambda: TasksConfig(autonomy="whenever"),
    ],
)
def test_bad_enum_rejected(factory):
    with pytest.raises(ValidationError):
        factory()


def test_logging_level_normalised():
    assert LoggingConfig(level="debug").level == "DEBUG"
    with pytest.raises(ValidationError):
        LoggingConfig(level="LOUD")


@pytest.mark.parametrize("port", [0, 70000, -1])
def test_bad_port_rejected(port):
    with pytest.raises(ValidationError):
        ApiConfig(port=port)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_threshold_out_of_range_rejected(value):
    with pytest.raises(ValidationError):
        DiarizationConfig(match_threshold=value)


def test_digest_hour_and_weekday_bounds():
    with pytest.raises(ValidationError):
        ProactiveConfig(digest_hour=24)
    with pytest.raises(ValidationError):
        ProactiveConfig(weekly_review_weekday=7)
    # valid extremes accepted
    assert ProactiveConfig(digest_hour=0, weekly_review_weekday=6).digest_hour == 0


def test_llm_temperature_bounds():
    with pytest.raises(ValidationError):
        LLMConfig(temperature=3.0)
    assert LLMConfig(temperature=0.7).temperature == 0.7


def test_settings_env_validation_propagates(monkeypatch):
    monkeypatch.setenv("SB_API__PORT", "999999")
    with pytest.raises(ValidationError):
        Settings()


# --- new validators (batch 3) -------------------------------------------------


def test_llm_request_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        LLMConfig(request_timeout_s=0)
    with pytest.raises(ValidationError):
        LLMConfig(request_timeout_s=-5.0)
    assert LLMConfig(request_timeout_s=30.0).request_timeout_s == 30.0


def test_llm_host_must_be_http_url():
    with pytest.raises(ValidationError):
        LLMConfig(host="127.0.0.1:11434")
    with pytest.raises(ValidationError):
        LLMConfig(host="tcp://somewhere")
    assert LLMConfig(host="https://127.0.0.1:11434").host.startswith("https://")


def test_llm_keep_alive_default():
    assert LLMConfig().keep_alive == "30m"


def test_security_session_age_minimum():
    from secondbrain.config import SecurityConfig

    with pytest.raises(ValidationError):
        SecurityConfig(session_max_age_days=0)
    assert SecurityConfig(session_max_age_days=1).session_max_age_days == 1


def test_encrypt_db_requires_passphrase():
    from secondbrain.config import SecurityConfig

    with pytest.raises(ValidationError) as exc_info:
        SecurityConfig(encrypt_db=True, db_passphrase="")
    assert "db_passphrase" in str(exc_info.value)
    ok = SecurityConfig(encrypt_db=True, db_passphrase="hunter2hunter2")
    assert ok.encrypt_db is True


def test_capture_rates_must_be_positive():
    from secondbrain.config import CaptureConfig

    with pytest.raises(ValidationError):
        CaptureConfig(sample_rate=0)
    with pytest.raises(ValidationError):
        CaptureConfig(chunk_seconds=0)
    assert CaptureConfig(sample_rate=44100, chunk_seconds=30).sample_rate == 44100


def test_diarization_threshold_ordering_enforced():
    # documented ordering: reattribute > match > owner_match > low_confidence
    with pytest.raises(ValidationError) as exc_info:
        DiarizationConfig(match_threshold=0.85)  # >= reattribute (0.80)
    assert "reattribute_threshold > match_threshold" in str(exc_info.value)
    with pytest.raises(ValidationError):
        DiarizationConfig(owner_match_threshold=0.71)  # >= match (0.70)
    with pytest.raises(ValidationError):
        DiarizationConfig(low_confidence_threshold=0.65)  # >= owner_match (0.65)
    # defaults respect the ordering
    d = DiarizationConfig()
    assert (d.reattribute_threshold > d.match_threshold
            > d.owner_match_threshold > d.low_confidence_threshold)


def test_logging_file_enabled_default_true():
    assert LoggingConfig().file_enabled is True
