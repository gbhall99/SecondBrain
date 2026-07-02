"""Defensive JSON parsing for LLM structured output."""

from __future__ import annotations

import json
import re


class LLMJSONError(ValueError):
    """Raised when an LLM response can't be parsed as the expected JSON."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str) -> dict:
    """Parse a JSON object from an LLM response, tolerating code fences.

    Falls back to extracting the outermost ``{...}`` span if there's leading or
    trailing prose. Raises :class:`LLMJSONError` on failure so the caller (the
    extraction worker) can record it and let the queue retry.
    """
    candidate = _FENCE.sub("", text).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMJSONError(f"could not parse JSON object: {exc}") from exc
    raise LLMJSONError("no JSON object found in response")
