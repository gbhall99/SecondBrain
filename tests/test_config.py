from secondbrain.config import Settings, load_settings


def test_defaults_and_derived_paths(tmp_path):
    s = Settings(paths={"data_dir": str(tmp_path)})
    assert s.api.host == "127.0.0.1"  # never bind publicly by default
    assert s.audio_raw_dir == tmp_path / "audio" / "raw"
    assert s.db_path == tmp_path / "secondbrain.db"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SB_API__PORT", "9999")
    monkeypatch.setenv("SB_TRANSCRIPTION__BACKEND", "mock")
    s = Settings()
    assert s.api.port == 9999
    assert s.transcription.backend == "mock"


def test_loads_repo_config_toml():
    # The committed config.toml should parse and bind locally by default.
    s = load_settings()
    assert s.api.host == "127.0.0.1"
    assert s.capture.sample_rate == 16000
