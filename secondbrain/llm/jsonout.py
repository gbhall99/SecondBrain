"""Defensive JSON parsing for LLM structured output."""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)


class LLMJSONError(ValueError):
    """Raised when an LLM response can't be parsed as the expected JSON."""


# Only a fence that opens the reply (optionally after whitespace); fence-lines
# inside string content must never be stripped.
_FENCE_OPEN = re.compile(r"\A\s*```[a-zA-Z0-9_-]*[ \t]*\r?\n?")

# Reprompt used by complete_json when the first reply wasn't valid JSON.
REPROMPT = "Your last reply was not valid JSON. Return ONLY the JSON object."


def _strip_fences(text: str) -> str:
    """Strip one leading code fence and its matching trailing fence, if present."""
    m = _FENCE_OPEN.match(text)
    if m is None:
        return text
    body = text[m.end():]
    end = body.rfind("```")
    if end != -1 and not body[end + 3:].strip():
        body = body[:end]
    return body


def _close_truncated(candidate: str) -> str | None:
    """Best-effort completion of truncated JSON: close open strings, drop a
    trailing comma, and append the missing closing brackets/braces.

    Nesting is tracked outside strings only. Returns None when the input isn't
    a plausibly-truncated JSON value (nothing open, or mismatched closers).
    """
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in candidate:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None  # mismatched — not simple truncation
            stack.pop()
    if not stack and not in_str:
        return None  # nothing to repair
    out = candidate
    if in_str:
        out += '"'
    out = out.rstrip()
    if out.endswith(","):
        out = out[:-1]
    return out + "".join(reversed(stack))


def _loads_object(s: str) -> dict:
    value = json.loads(s)
    if not isinstance(value, dict):
        raise LLMJSONError(f"expected a JSON object, got {type(value).__name__}")
    return value


def parse_json(text: str) -> dict:
    """Parse a JSON *object* from an LLM response, tolerating code fences.

    Falls back to extracting the outermost ``{...}`` span if there's leading or
    trailing prose, then to repairing plausibly-truncated output (a generation
    cut off mid-object) by closing open strings/brackets. Raises
    :class:`LLMJSONError` on failure — including when the reply parses but the
    top-level value isn't an object — so the caller (the extraction worker) can
    record it and let the queue retry.
    """
    candidate = _strip_fences(text).strip()
    try:
        return _loads_object(candidate)
    except LLMJSONError:
        raise  # parsed fine but isn't an object — not repairable
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return _loads_object(candidate[start : end + 1])
        except json.JSONDecodeError:
            pass
    if start != -1:
        repaired = _close_truncated(candidate[start:])
        if repaired is not None:
            try:
                return _loads_object(repaired)
            except (json.JSONDecodeError, LLMJSONError):
                pass
    log.warning("LLM JSON parse failed; response starts: %r", text[:200])
    if start == -1:
        raise LLMJSONError("no JSON object found in response")
    raise LLMJSONError("could not parse JSON object from response")


def complete_json(
    llm,
    *,
    system: str,
    prompt: str,
    schema: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    retries: int = 1,
) -> dict:
    """``llm.complete(...)`` + :func:`parse_json`, with one reprompt on failure.

    When the first reply isn't parseable JSON, the model is asked once more with
    an explicit "return ONLY the JSON object" nudge before the
    :class:`LLMJSONError` propagates.
    """
    attempt_prompt = prompt
    for remaining in range(retries, -1, -1):
        resp = llm.complete(
            system=system,
            prompt=attempt_prompt,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return parse_json(resp.text)
        except LLMJSONError:
            if remaining == 0:
                raise
            attempt_prompt = f"{prompt}\n\n{REPROMPT}"
    raise LLMJSONError("unreachable")  # pragma: no cover
