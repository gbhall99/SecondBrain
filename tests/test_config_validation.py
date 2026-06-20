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
