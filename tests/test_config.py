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


def test_diarization_config_defaults_and_env(monkeypatch):
    s = Settings()
    assert s.diarization.enabled is False           # off by default
    assert s.diarization.embedding_dim == 256
    assert s.conversation.max_gap_minutes == 5.0
    monkeypatch.setenv("SB_DIARIZATION__ENABLED", "true")
    monkeypatch.setenv("SB_DIARIZATION__MATCH_THRESHOLD", "0.8")
    s2 = Settings()
    assert s2.diarization.enabled is True
    assert s2.diarization.match_threshold == 0.8


def test_llm_and_extraction_config_defaults_and_env(monkeypatch):
    s = Settings()
    assert s.llm.backend == "mock"          # CI default
    assert s.extraction.enabled is False     # off by default
    monkeypatch.setenv("SB_LLM__BACKEND", "ollama")
    monkeypatch.setenv("SB_EXTRACTION__ENABLED", "true")
    s2 = Settings()
    assert s2.llm.backend == "ollama"
    assert s2.extraction.enabled is True


def test_proactive_config_defaults_and_env(monkeypatch):
    s = Settings()
    assert s.proactive.enabled is False
    assert s.proactive.top_n == 5
    monkeypatch.setenv("SB_PROACTIVE__ENABLED", "true")
    monkeypatch.setenv("SB_PROACTIVE__DIGEST_HOUR", "8")
    s2 = Settings()
    assert s2.proactive.enabled is True
    assert s2.proactive.digest_hour == 8


def test_security_config_defaults_and_env(monkeypatch):
    s = Settings()
    assert s.security.require_auth is False
    assert s.security.encrypt_db is False
    monkeypatch.setenv("SB_SECURITY__REQUIRE_AUTH", "true")
    monkeypatch.setenv("SB_SECURITY__DB_PASSPHRASE", "hunter2")
    s2 = Settings()
    assert s2.security.require_auth is True
    assert s2.security.db_passphrase == "hunter2"


def test_tasks_config_defaults_and_env(monkeypatch):
    s = Settings()
    assert s.tasks.enabled is False
    assert s.tasks.web_research_enabled is False
    assert s.tasks.daily_capacity_minutes == 240
    monkeypatch.setenv("SB_TASKS__ENABLED", "true")
    monkeypatch.setenv("SB_TASKS__DAILY_CAPACITY_MINUTES", "120")
    s2 = Settings()
    assert s2.tasks.enabled is True
    assert s2.tasks.daily_capacity_minutes == 120


def test_diarization_quality_config_defaults(monkeypatch):
    s = Settings()
    assert s.diarization.exemplar_k == 3
    assert s.diarization.reattribute_threshold == 0.80
    assert s.diarization.overlap_flag is True
    monkeypatch.setenv("SB_DIARIZATION__EXEMPLAR_K", "5")
    assert Settings().diarization.exemplar_k == 5


def test_loads_repo_config_toml():
    # The committed config.toml should parse and bind locally by default.
    s = load_settings()
    assert s.api.host == "127.0.0.1"
    assert s.capture.sample_rate == 16000
