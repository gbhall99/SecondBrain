"""Shared test fixtures.

Tests run fully on Linux/CI: VAD and transcription use mock backends, and
semantic search is disabled (no sqlite-vec / embedding model required).
"""

from __future__ import annotations

import pytest

from secondbrain.config import Settings
from secondbrain.storage.db import init_db


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        paths={"data_dir": str(tmp_path / "data")},
        # input_device pinned to "" (system default) so a device name in this
        # machine's config.local.toml can't leak in and fail the microphone
        # health check whenever that hardware is unplugged — tests are hermetic.
        capture={"sample_rate": 16000, "channels": 1, "chunk_seconds": 1,
                 "min_free_disk_gb": 0.0, "input_device": ""},
        vad={"enabled": False},
        transcription={"backend": "mock"},
        search={"semantic_enabled": False},
        # Explicit mock so tests never talk to a real Ollama configured in this
        # machine's config.local.toml (init kwargs beat the TOML sources).
        llm={"backend": "mock"},
    )


@pytest.fixture
def conn(settings):
    settings.ensure_dirs()
    c = init_db(settings=settings)
    try:
        yield c
    finally:
        c.close()
