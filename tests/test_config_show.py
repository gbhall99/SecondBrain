"""redacted_dict — effective config is printable without leaking secrets."""

from __future__ import annotations

from secondbrain.config import Settings, redacted_dict


def test_redacts_set_secrets():
    s = Settings(
        security={"db_passphrase": "hunter2"},
        diarization={"hf_token": "hf_abc"},
    )
    d = redacted_dict(s)
    assert d["security"]["db_passphrase"] == "***redacted***"
    assert d["diarization"]["hf_token"] == "***redacted***"


def test_empty_secrets_left_empty():
    d = redacted_dict(Settings())
    assert d["security"]["db_passphrase"] == ""
    assert d["diarization"]["hf_token"] == ""


def test_non_secret_values_preserved():
    s = Settings(api={"port": 9001})
    d = redacted_dict(s)
    assert d["api"]["port"] == 9001
    assert d["api"]["host"] == "127.0.0.1"


def test_redacted_dict_does_not_mutate_settings():
    s = Settings(security={"db_passphrase": "secret"})
    redacted_dict(s)
    assert s.security.db_passphrase == "secret"  # original untouched
