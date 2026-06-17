"""Local LLM behind a backend interface.

Mirrors ``pipeline/transcribe.py`` / ``pipeline/diarize.py``: the real backend
(Ollama over local HTTP) and a deterministic ``MockLLM`` for CI, selected by a
factory. Fully local — Ollama runs on 127.0.0.1; nothing leaves the machine.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from secondbrain.config import Settings, get_settings


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    raw: dict | None = None


class LLM(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...


class MockLLM(LLM):
    """Deterministic LLM for CI/dev.

    Resolution order: a scripted FIFO queue, then substring-keyed responses, then
    a deterministic default (empty JSON object when a schema is requested, else a
    short hash-seeded string). This lets extraction and chat be unit-tested with
    zero model downloads.
    """

    backend_name = "mock"

    def __init__(
        self,
        responses: list[str] | None = None,
        by_substring: dict[str, str] | None = None,
        default: str | None = None,
    ):
        self._responses = list(responses or [])
        self._by_substring = by_substring or {}
        self._default = default

    def complete(self, *, system, prompt, schema=None, temperature=0.0, max_tokens=None):
        text = self._resolve(prompt, schema)
        return LLMResponse(text=text, model="mock", backend=self.backend_name)

    def _resolve(self, prompt: str, schema: dict | None) -> str:
        if self._responses:
            return self._responses.pop(0)
        for needle, resp in self._by_substring.items():
            if needle in prompt:
                return resp
        if self._default is not None:
            return self._default
        if schema is not None:
            return "{}"  # schema-valid empty object
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock-llm:{digest}]"


class OllamaLLM(LLM):
    """Local Ollama server (127.0.0.1:11434). Supports schema-constrained output."""

    backend_name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm.model

    def complete(self, *, system, prompt, schema=None, temperature=0.0, max_tokens=None):
        import httpx  # already a dependency; imports cleanly on any OS

        options: dict = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": options,
        }
        if schema is not None:
            payload["format"] = schema  # Ollama constrains generation to the schema
        resp = httpx.post(
            f"{self.settings.llm.host}/api/chat",
            json=payload,
            timeout=self.settings.llm.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            backend=self.backend_name,
            raw=data,
        )


def get_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    backend = settings.llm.backend.lower()
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        return OllamaLLM(settings)
    raise ValueError(f"Unknown LLM backend: {backend!r}")


def dumps_schema(model_json_schema: dict) -> str:
    """Helper to serialize a JSON schema for prompts/debugging."""
    return json.dumps(model_json_schema, separators=(",", ":"))
