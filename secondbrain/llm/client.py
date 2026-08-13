"""Local LLM behind a backend interface.

Mirrors ``pipeline/transcribe.py`` / ``pipeline/diarize.py``: the real backend
(Ollama over local HTTP) and a deterministic ``MockLLM`` for CI, selected by a
factory. Fully local — Ollama runs on 127.0.0.1; nothing leaves the machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from secondbrain.config import Settings, get_settings

log = logging.getLogger(__name__)


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

    async def astream(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield the completion incrementally (used by the chat streaming API).

        Default implementation for backends without native streaming: run the
        blocking ``complete`` in a worker thread (so it can't stall the event
        loop) and yield the whole answer as one chunk.
        """
        import anyio  # starlette/httpx dependency; imports cleanly everywhere

        def _call() -> LLMResponse:
            return self.complete(
                system=system, prompt=prompt, temperature=temperature, max_tokens=max_tokens
            )

        resp = await anyio.to_thread.run_sync(_call)
        if resp.text:
            yield resp.text


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

    # num_ctx buckets: Ollama's default context window (often 2k-4k) silently
    # truncates long prompts *from the start*, which would drop the system
    # prompt (citation + general-knowledge labeling rules) exactly when the
    # retrieval context is rich. We size the window to the request using a
    # conservative ~3 chars/token estimate, in coarse buckets so the model
    # isn't reloaded for every small variation in prompt size.
    _CTX_BUCKETS = (8192, 16384, 32768)

    # Transient-failure retries on the blocking path: connection refused, read
    # timeout, or a 5xx from Ollama (model still loading, restart). 4xx is never
    # retried — a missing model won't appear by asking again.
    _RETRY_BACKOFF_S = (0.5, 2.0)

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm.model
        self._http = None  # lazily-created httpx.Client, reused across calls

    def _client(self):
        """Reused blocking HTTP client (connection pooling across calls)."""
        import httpx

        if self._http is None:
            self._http = httpx.Client(
                timeout=httpx.Timeout(self.settings.llm.request_timeout_s, connect=5.0)
            )
        return self._http

    def _options(
        self, system: str, prompt: str, temperature: float, max_tokens: int | None
    ) -> dict:
        need = (len(system) + len(prompt)) // 3 + (max_tokens or 2048) + 512
        if need > self._CTX_BUCKETS[-1]:
            log.warning(
                "prompt needs ~%d ctx tokens but the largest num_ctx bucket is %d; "
                "the start of the prompt may be truncated by the model",
                need, self._CTX_BUCKETS[-1],
            )
        num_ctx = next((b for b in self._CTX_BUCKETS if b >= need), self._CTX_BUCKETS[-1])
        options: dict = {"temperature": temperature, "num_ctx": num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens
        return options

    def _payload(self, system: str, prompt: str, options: dict, *, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
            # Keep the model resident between calls so the next request doesn't
            # pay the multi-second load again.
            "keep_alive": self.settings.llm.keep_alive,
            "options": options,
        }

    @staticmethod
    def _raise_for_status(resp) -> None:
        """raise_for_status, but with the response body in the message so
        Ollama's actual complaint ("model not found, try pulling it") reaches
        the user instead of a bare status code."""
        import httpx

        if resp.status_code < 400:
            return
        body = (resp.text or "")[:200]
        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code} from Ollama: {body}",
            request=resp.request,
            response=resp,
        )

    def complete(self, *, system, prompt, schema=None, temperature=0.0, max_tokens=None):
        import httpx  # already a dependency; imports cleanly on any OS

        payload = self._payload(
            system, prompt, self._options(system, prompt, temperature, max_tokens), stream=False
        )
        if schema is not None:
            payload["format"] = schema  # Ollama constrains generation to the schema
        url = f"{self.settings.llm.host}/api/chat"
        started = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(len(self._RETRY_BACKOFF_S) + 1):
            if attempt:
                time.sleep(self._RETRY_BACKOFF_S[attempt - 1])
            try:
                resp = self._client().post(url, json=payload)
                self._raise_for_status(resp)
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                log.warning("Ollama request failed (%s), attempt %d", exc, attempt + 1)
                continue
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500:  # transient server side
                    last_exc = exc
                    log.warning("Ollama request failed (%s), attempt %d", exc, attempt + 1)
                    continue
                raise  # 4xx: never retry
            data = resp.json()
            log.debug(
                "ollama call model=%s duration_ms=%s eval_count=%s",
                self.model,
                round((data.get("total_duration") or 0) / 1e6)
                or round((time.monotonic() - started) * 1000),
                data.get("eval_count"),
            )
            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=self.model,
                backend=self.backend_name,
                raw=data,
            )
        raise last_exc  # both retries exhausted

    async def astream(self, *, system, prompt, temperature=0.0, max_tokens=None):
        """Token stream from Ollama (``stream: true``), yielded as text chunks.

        The read timeout applies *between* chunks, so a long generation streams
        for as long as tokens keep arriving; a stalled model still times out.
        Cancelling the surrounding task closes the HTTP connection, which makes
        Ollama abort the generation instead of finishing it for nobody.
        """
        import httpx

        payload = self._payload(
            system, prompt, self._options(system, prompt, temperature, max_tokens), stream=True
        )
        timeout = httpx.Timeout(self.settings.llm.request_timeout_s, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client, client.stream(
            "POST", f"{self.settings.llm.host}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue  # ignore malformed keep-alive noise
                piece = (data.get("message") or {}).get("content") or ""
                if piece:
                    yield piece
                if data.get("done"):
                    return


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
