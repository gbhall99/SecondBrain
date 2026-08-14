"""Typed configuration loaded from config.toml (+ config.local.toml) and env vars.

Precedence (highest first): environment variables (prefix ``SB_``), then
``config.local.toml``, then ``config.toml``, then defaults defined here. Nested
fields use ``__`` as the env delimiter, e.g. ``SB_API__PORT=9000``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
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

    @field_validator("sample_rate")
    @classmethod
    def _check_sample_rate(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"capture.sample_rate must be > 0, got {v}")
        return v

    @field_validator("chunk_seconds")
    @classmethod
    def _check_chunk_seconds(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"capture.chunk_seconds must be > 0, got {v}")
        return v


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


def _one_of(field: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {value!r}")
    return value


class TranscriptionConfig(BaseModel):
    backend: str = "parakeet"  # parakeet | whisper | mock
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v2"
    language: str = ""
    # Skip transcription when VAD found less total speech than this (seconds).
    # 0.0 keeps the historical behavior (any speech at all is transcribed).
    min_speech_seconds: float = 0.0

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        return _one_of("transcription.backend", v, {"parakeet", "whisper", "mock"})


class SearchConfig(BaseModel):
    semantic_enabled: bool = True
    embedding_model: str = "bge-small"
    # Nearest-neighbour ceiling (L2 distance between normalized embeddings,
    # 0..2). vec0 KNN always returns `k` rows however unrelated; anything above
    # this is noise, not a match. Calibrated on real transcripts: verbatim ≈0.55,
    # paraphrase ≈0.7-0.77, unrelated ≥0.9. Raise it if semantic recall feels
    # too strict, lower it if unrelated lines slip in.
    semantic_max_distance: float = 0.8


class ConversationConfig(BaseModel):
    # Chunks within this gap of each other belong to the same conversation; a
    # larger idle gap closes the open conversation (→ enqueue diarization).
    max_gap_minutes: float = 5.0
    min_conversation_seconds: float = 5.0
    # A chunk that would stretch a conversation past this closes it and starts a
    # new one, so an all-day open mic can't accrete one giant "meeting".
    max_conversation_minutes: float = 120.0


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
    # An exemplar match whose top1−top2 similarity margin is below this is
    # labeled but flagged low-confidence instead of confidently auto-labeled
    # (two candidate voices were nearly tied). 0.0 disables the gate.
    min_match_margin: float = 0.05
    # Owner enrollment needs at least this much total speech across the clips.
    min_enroll_speech_s: float = 10.0
    # Phase 7 — quality/self-correction
    exemplar_k: int = 3                    # match vs k nearest stored exemplars
    max_exemplars_per_speaker: int = 50    # cap kept exemplars (prune beyond)
    reattribute_threshold: float = 0.80    # HIGH bar to relabel past low-conf segs
    prune_min_confidence: float = 0.3      # drop exemplars below this quality
    overlap_flag: bool = True              # flag overlapped segments low-confidence
    # Mac-side pyannote knobs (passed through; tune on device)
    segmentation_threshold: float = 0.5
    min_speakers: int = 0                  # 0 = auto
    max_speakers: int = 0                  # 0 = auto

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        return _one_of("diarization.backend", v, {"pyannote", "mock"})

    @field_validator(
        "match_threshold", "owner_match_threshold", "centroid_update_threshold",
        "cluster_distance_threshold", "low_confidence_threshold", "reattribute_threshold",
        "prune_min_confidence", "segmentation_threshold", "min_match_margin",
    )
    @classmethod
    def _check_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _check_threshold_ordering(self) -> DiarizationConfig:
        # Documented in docs/OPERATIONS.md: relabeling past lines needs a HIGHER
        # bar than live matching, the owner is checked slightly looser, and the
        # low-confidence flag sits below all of them. A violation silently
        # breaks attribution quality, so fail fast with the expected ordering.
        ordered = (
            ("reattribute_threshold", self.reattribute_threshold),
            ("match_threshold", self.match_threshold),
            ("owner_match_threshold", self.owner_match_threshold),
            ("low_confidence_threshold", self.low_confidence_threshold),
        )
        for (hi_name, hi), (lo_name, lo) in zip(ordered, ordered[1:], strict=False):
            if hi <= lo:
                raise ValueError(
                    "diarization thresholds must satisfy reattribute_threshold > "
                    "match_threshold > owner_match_threshold > low_confidence_threshold; "
                    f"got {hi_name}={hi} <= {lo_name}={lo}"
                )
        return self


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765

    @field_validator("port")
    @classmethod
    def _check_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"api.port must be in [1, 65535], got {v}")
        return v


class SecurityConfig(BaseModel):
    # Auth + at-rest encryption. All OFF by default so local use & CI are
    # unchanged. Enable when exposing the UI beyond localhost (e.g. Tailscale).
    require_auth: bool = False
    username: str = "owner"
    session_max_age_days: int = 14
    # SQLCipher at-rest encryption of the database (needs the `secure` extra +
    # a passphrase; put the passphrase in config.local.toml or SB_SECURITY__DB_PASSPHRASE).
    encrypt_db: bool = False
    db_passphrase: str = ""

    @field_validator("session_max_age_days")
    @classmethod
    def _check_session_age(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"security.session_max_age_days must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def _check_encrypt_db_has_passphrase(self) -> SecurityConfig:
        # Fail at load time, not on the first DB open deep inside the daemon.
        if self.encrypt_db and not self.db_passphrase:
            raise ValueError(
                "security.encrypt_db is true but security.db_passphrase is empty — "
                "set it in config.local.toml or the SB_SECURITY__DB_PASSPHRASE env var"
            )
        return self


class BackupConfig(BaseModel):
    # Automatic daily DB snapshots from the daemon's maintenance loop.
    auto_enabled: bool = True
    keep: int = 7          # prune to the newest N snapshots after each auto backup


class LoggingConfig(BaseModel):
    level: str = "INFO"
    # Also write logs to <data>/logs/secondbrain.log (rotated) besides stdout.
    file_enabled: bool = True

    @field_validator("level")
    @classmethod
    def _check_level(cls, v: str) -> str:
        up = v.upper()
        return _one_of("logging.level", up, {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class TasksConfig(BaseModel):
    # Goal decomposition + tasks + daily planning (Phase 6). OFF by default.
    enabled: bool = False
    daily_capacity_minutes: int = 240
    # Length of a working day; the planner suggests capacity = workday minus
    # today's recorded meeting minutes (floored at 30).
    workday_minutes: int = 480
    urgent_days: int = 2               # due within N days → "urgent" quadrant
    important_value: int = 4           # value ≥ this → "important" quadrant
    # Opt-in web research per task (local graph-RAG research is always available).
    web_research_enabled: bool = False
    web_search_url: str = ""           # e.g. a SearXNG JSON endpoint; user-supplied
    autonomy: str = "propose"          # propose (approve) | auto

    @field_validator("autonomy")
    @classmethod
    def _check_autonomy(cls, v: str) -> str:
        return _one_of("tasks.autonomy", v, {"propose", "auto"})


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
    # Jaccard keyword overlap is a much coarser signal than embedding cosine,
    # so the keyword fallback gets its own (lower) linking threshold.
    goal_link_keyword_threshold: float = 0.3
    due_soon_days: int = 3
    stale_goal_days: int = 14
    stale_days: int = 21
    confidence_floor: float = 0.4
    suppress_days: int = 30
    urgent_due_hours: int = 24
    reconnect_days: int = 30           # flag a known person not seen in N days
    snooze_default_days: int = 7       # per-item snooze length when none chosen
    goal_at_risk_days: int = 14        # target date within N days + low progress → at risk

    @field_validator("digest_hour")
    @classmethod
    def _check_hour(cls, v: int) -> int:
        if not 0 <= v <= 23:
            raise ValueError(f"proactive.digest_hour must be in [0, 23], got {v}")
        return v

    @field_validator("weekly_review_weekday")
    @classmethod
    def _check_weekday(cls, v: int) -> int:
        if not 0 <= v <= 6:
            raise ValueError(f"proactive.weekly_review_weekday must be in [0, 6], got {v}")
        return v

    @field_validator(
        "connection_threshold", "goal_link_threshold", "goal_link_keyword_threshold",
        "confidence_floor",
    )
    @classmethod
    def _check_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {v}")
        return v


class LLMConfig(BaseModel):
    # Local LLM for knowledge extraction + Q&A. "mock" is the CI/dev default;
    # "ollama" talks to a local Ollama server (fully offline).
    backend: str = "mock"  # mock | ollama
    model: str = "llama3.1:8b-instruct"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    request_timeout_s: float = 120.0
    # How long Ollama keeps the model loaded after a request ("30m", "1h", 0=unload).
    keep_alive: str = "30m"

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        return _one_of("llm.backend", v, {"mock", "ollama"})

    @field_validator("host")
    @classmethod
    def _check_host(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"llm.host must start with http:// or https://, got {v!r}")
        return v

    @field_validator("request_timeout_s")
    @classmethod
    def _check_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"llm.request_timeout_s must be > 0, got {v}")
        return v

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"llm.temperature must be in [0.0, 2.0], got {v}")
        return v


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
    # How many retrieval hits feed the chat context (each expanded with a few
    # neighboring lines from the same conversation, within the char budget).
    chat_max_excerpts: int = 24


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
    backup: BackupConfig = Field(default_factory=BackupConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)

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
        # data_path first so it exists before launchd writes its logs into it.
        for d in (self.data_path, self.audio_raw_dir, self.audio_processed_dir, self.models_dir):
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


# Secret fields that must never be committed to the version-controlled
# config.toml — they belong in config.local.toml (gitignored) or the
# environment. Each entry is (section, field).
SECRET_FIELDS: tuple[tuple[str, str], ...] = (
    ("security", "db_passphrase"),
    ("diarization", "hf_token"),
    ("tasks", "web_search_url"),  # may embed an API key / token
)


_REDACTED = "***redacted***"


def redacted_dict(settings: Settings) -> dict:
    """The effective settings as a dict with secret fields masked.

    Safe to print/log — masks every value listed in :data:`SECRET_FIELDS` that
    is non-empty (empty stays empty, so it's clear nothing is set).
    """
    data = settings.model_dump()
    for section, field in SECRET_FIELDS:
        sec = data.get(section)
        if isinstance(sec, dict) and sec.get(field):
            sec[field] = _REDACTED
    return data


def committed_secrets(config_path: Path | None = None) -> list[str]:
    """Return secret field paths that have a non-empty value in committed config.toml.

    Reads ONLY the version-controlled ``config.toml`` (never config.local.toml or
    env), so this flags secrets a user accidentally placed where git would track
    them. Returns dotted paths like ``"security.db_passphrase"``; empty if clean
    or the file is absent/unreadable.
    """
    import tomllib

    path = config_path or (REPO_ROOT / "config.toml")
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return []
    found = []
    for section, field in SECRET_FIELDS:
        value = data.get(section, {}).get(field) if isinstance(data.get(section), dict) else None
        if value:
            found.append(f"{section}.{field}")
    return found


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
