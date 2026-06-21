"""The seeded config.local.toml.example must be valid and enable the AI features.

Guards against a typo in the example silently breaking a user's deployment config.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from secondbrain.config import Settings

EXAMPLE = Path(__file__).resolve().parent.parent / "config.local.toml.example"


def test_example_parses_and_builds_settings():
    data = tomllib.loads(EXAMPLE.read_text())
    s = Settings(**data)  # raises if the example has an invalid shape
    assert s.diarization.enabled is True
    assert s.extraction.enabled is True
    assert s.proactive.enabled is True
    assert s.llm.backend == "ollama"
