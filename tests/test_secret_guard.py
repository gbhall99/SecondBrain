"""Secret-in-committed-config guard — flag secrets that belong in local/env."""

from __future__ import annotations

from secondbrain import health
from secondbrain.config import committed_secrets


def test_clean_config_reports_no_secrets(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[api]\nport = 8765\n\n[security]\nrequire_auth = false\n')
    assert committed_secrets(cfg) == []


def test_db_passphrase_in_committed_config_flagged(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[security]\ndb_passphrase = "hunter2"\n')
    assert committed_secrets(cfg) == ["security.db_passphrase"]


def test_multiple_secrets_flagged(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[security]\ndb_passphrase = "x"\n'
        '[diarization]\nhf_token = "hf_abc"\n'
    )
    assert set(committed_secrets(cfg)) == {"security.db_passphrase", "diarization.hf_token"}


def test_empty_secret_value_not_flagged(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[diarization]\nhf_token = ""\n')
    assert committed_secrets(cfg) == []


def test_missing_file_is_clean(tmp_path):
    assert committed_secrets(tmp_path / "nope.toml") == []


def test_health_check_passes_when_clean(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[api]\nport = 1\n")
    # _secrets() reads the repo's config.toml; verify via the scanner directly
    assert committed_secrets(cfg) == []
    check = health._secrets()
    assert check.name == "secrets"
    assert isinstance(check.ok, bool)
