"""Human-readable failure messages for local-LLM (Ollama) transport errors.

Shared by the web API and the CLI so "Ollama isn't running" reads the same
everywhere instead of surfacing as an httpx traceback.
"""

from __future__ import annotations

from secondbrain.config import Settings


def llm_failure_detail(exc: Exception, settings: Settings) -> str | None:
    """Actionable message for an Ollama transport failure, else None.

    None means the exception is not a recognised LLM transport problem and the
    caller should re-raise (or handle it its own way).
    """
    import httpx  # already a dependency; imports cleanly on any OS

    if isinstance(exc, httpx.ConnectError):
        return (
            "Couldn't reach the local model — is Ollama running? "
            f"(expected at {settings.llm.host})"
        )
    if isinstance(exc, httpx.TimeoutException):
        return (
            f"The local model didn't answer within {int(settings.llm.request_timeout_s)}s. "
            "It may be busy loading — try again, or ask a simpler question."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        hint = (
            f" Model '{settings.llm.model}' may not be pulled — try `ollama pull "
            f"{settings.llm.model}`." if exc.response.status_code == 404 else ""
        )
        return f"The local model returned an error (HTTP {exc.response.status_code}).{hint}"
    return None
