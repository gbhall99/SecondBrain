"""Safe, targeted edits to ``config.local.toml`` (used by ``deploy/install.sh``).

Deliberately tiny: we only need to set ``[diarization].hf_token`` during install,
so rather than depend on a TOML *writer* we do a precise, tested edit that leaves
the rest of the file untouched. The value is escaped via :func:`json.dumps` (TOML
basic strings accept the same escapes for this charset), so the result re-parses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ASSIGN = re.compile(r'^(\s*)hf_token\s*=.*$', re.MULTILINE)
_SECTION = re.compile(r'^\[diarization\]\s*$', re.MULTILINE)
_ANY_SECTION = re.compile(r'^\[[^\]]+\]\s*$', re.MULTILINE)

# Secrets land in this file: owner read/write only.
SECRET_FILE_MODE = 0o600


def _diarization_span(text: str) -> tuple[int, int] | None:
    """(start, end) offsets of the ``[diarization]`` section body, or None.

    The body runs from just after the section header to the next section header
    (or EOF), so an ``hf_token`` key in some *other* section is never touched.
    """
    m = _SECTION.search(text)
    if m is None:
        return None
    nxt = _ANY_SECTION.search(text, m.end())
    return (m.end(), nxt.start() if nxt else len(text))


def set_hf_token(text: str, token: str) -> str:
    """Return ``text`` with ``[diarization].hf_token`` set to ``token``.

    Three cases, in order: replace an existing ``hf_token = …`` assignment
    *inside the [diarization] section* (preserving indentation); else insert one
    just after a ``[diarization]`` header; else append a new ``[diarization]``
    section. An ``hf_token`` key in any other section is left alone.
    """
    quoted = json.dumps(token)
    span = _diarization_span(text)
    if span is not None:
        start, end = span
        body = text[start:end]
        if _ASSIGN.search(body):
            body = _ASSIGN.sub(lambda m: f"{m.group(1)}hf_token = {quoted}", body, count=1)
            return text[:start] + body + text[end:]
        return text[:start] + f"\nhf_token = {quoted}" + text[start:]
    # No [diarization] section. A bare top-level `hf_token = …` (no section
    # header before it) is still the same key — replace it in place.
    first_section = _ANY_SECTION.search(text)
    top = text[: first_section.start()] if first_section else text
    if _ASSIGN.search(top):
        return _ASSIGN.sub(lambda m: f"{m.group(1)}hf_token = {quoted}", text, count=1)
    sep = "" if text == "" or text.endswith("\n") else "\n"
    return f"{text}{sep}\n[diarization]\nhf_token = {quoted}\n"


def write_hf_token(path: str | Path, token: str) -> None:
    """Set ``[diarization].hf_token`` to ``token`` in the file at ``path``.

    The file is chmod'd to owner-only (0600) afterwards — it holds a secret.
    """
    p = Path(path)
    original = p.read_text() if p.exists() else ""
    p.write_text(set_hf_token(original, token))
    p.chmod(SECRET_FILE_MODE)
