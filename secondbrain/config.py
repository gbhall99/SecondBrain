"""Typed configuration loaded from config.toml (+ config.local.toml) and env vars.

Precedence (highest first): environment variables (prefix ``SB_``), then
``config.local.toml``, then ``config.toml``, then defaults defined here. Nested
fields use ``__`` as the env delimiter, e.g. ``SB_API__PORT=9000``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    data_dir: str = "data"


class CaptureConfig(BaseModel):
    input_device: str = ""
    sample_rate: int = 16000
    channels: int = 1
    chunk_seconds: int = 60
    min_free_disk_gb: float = 5.0


class ConsentConfig(BaseModel):
    recording_enabled: bool = True
    paused: bool = False
    raw_audio_retention_hours: int = 168
    speaker_opt_out: list[str] = Field(default_factory=list)


class VadConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 700


class TranscriptionConfig(BaseModel):
    backend: str = "parakeet"  # parakeet | whisper | mock
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v2"
    language: str = ""


class SearchConfig(BaseModel):
    semantic_enabled: bool = True
    embedding_model: str = "bge-small"


class ConversationConfig(BaseModel):
    # Chunks within this gap of each other belong to the same conversation; a
    # larger idle gap closes the open conversation (→ enqueue diarization).
    max_gap_minutes: float = 5.0
    min_conversation_seconds: float = 5.0


class DiarizationConfig(BaseModel):
    # Disabled by default so Phase 1 behaviour (and CI) is unchanged until opted
    # in. The Mac mini config.toml enables it.
    enabled: bool = False
    backend: str = "pyannote"  # pyannote | mock
    model: str = "pyannote/speaker-diarization-3.1"
    # One-time HuggingFace token for the gated models. Prefer config.local.toml
    # or the HF_TOKEN / HUGGINGFACE_TOKEN env var over committing it here.
    hf_token: str = ""
    embedding_dim: int = 256
    match_threshold: float = 0.70          # cosine sim to auto-label a known voice
    owner_match_threshold: float = 0.65    # owner checked first, slightly looser
    centroid_update_threshold: float = 0.75  # only fold confident obs into centroid
    cluster_distance_threshold: float = 0.30  # nightly agglomerative (cosine dist)
    low_confidence_threshold: float = 0.5  # below this a label is flagged
    min_cluster_speech_s: float = 1.0      # ignore clusters too short to embed


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class ProactiveConfig(BaseModel):
    # Proactive engine: nightly brief + weekly review, goals, nudges. OFF by
    # default so earlier phases/CI are unchanged until enabled on the Mac.
    enabled: bool = False
    event_triggers: bool = False       # real-time nudges for urgent commitments
    coaching_enabled: bool = False     # candid, transcript-grounded coaching (opt-in)
    digest_hour: int = 6               # local hour the morning brief is generated
    weekly_review_weekday: int = 0     # 0=Monday … 6=Sunday
    top_n: int = 5                     # daily cap on surfaced suggestions
    per_kind_cap: int = 2
    recent_days: int = 1
    lookback_days: int = 30
    connection_threshold: float = 0.78
    goal_link_threshold: float = 0.72
    due_soon_days: int = 3
    stale_goal_days: int = 14
    stale_days: int = 21
    confidence_floor: float = 0.4
    suppress_days: int = 30
    urgent_due_hours: int = 24


class LLMConfig(BaseModel):
    # Local LLM for knowledge extraction + Q&A. "mock" is the CI/dev default;
    # "ollama" talks to a local Ollama server (fully offline).
    backend: str = "mock"  # mock | ollama
    model: str = "llama3.1:8b-instruct"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    request_timeout_s: float = 120.0


class ExtractionConfig(BaseModel):
    # Knowledge graph extraction + chat. Disabled by default so Phase 1/2 (and
    # CI) behaviour is unchanged until opted in on the Mac.
    enabled: bool = False
    max_context_chars: int = 24000        # ~6k tokens; char heuristic, no tokenizer
    overlap_segments: int = 1
    entity_match_threshold: float = 0.82  # cosine: auto-link to existing node
    entity_review_threshold: float = 0.70  # in [review, match): LLM disambiguation
    chat_max_hops: int = 1
    chat_max_facts: int = 40
    chat_max_context_chars: int = 32000


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="SB_",
        env_nested_delimiter="__",
        extra="ignore",
        # Both files are read; config.local.toml (listed last) wins over
        # config.toml, and env vars override both (see settings_customise_sources).
        toml_file=[str(REPO_ROOT / "config.toml"), str(REPO_ROOT / "config.local.toml")],
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    # --- derived paths -------------------------------------------------------

    @property
    def data_path(self) -> Path:
        p = Path(self.paths.data_dir)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def audio_raw_dir(self) -> Path:
        return self.data_path / "audio" / "raw"

    @property
    def audio_processed_dir(self) -> Path:
        return self.data_path / "audio" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.data_path / "models"

    @property
    def db_path(self) -> Path:
        return self.data_path / "secondbrain.db"

    def ensure_dirs(self) -> None:
        for d in (self.audio_raw_dir, self.audio_processed_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first: explicit kwargs > env vars > .env > TOML files.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )


def load_settings() -> Settings:
    """Build :class:`Settings` from TOML files + environment overrides."""
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached process-wide settings. Call :func:`reload_settings` after edits."""
    return load_settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
