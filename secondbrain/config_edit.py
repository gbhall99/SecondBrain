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


def set_hf_token(text: str, token: str) -> str:
    """Return ``text`` with ``[diarization].hf_token`` set to ``token``.

    Three cases, in order: replace an existing ``hf_token = …`` assignment
    (preserving indentation); else insert one just after a ``[diarization]`` header;
    else append a new ``[diarization]`` section.
    """
    quoted = json.dumps(token)
    if _ASSIGN.search(text):
        return _ASSIGN.sub(lambda m: f"{m.group(1)}hf_token = {quoted}", text, count=1)
    m = _SECTION.search(text)
    if m:
        return text[: m.end()] + f"\nhf_token = {quoted}" + text[m.end():]
    sep = "" if text == "" or text.endswith("\n") else "\n"
    return f"{text}{sep}\n[diarization]\nhf_token = {quoted}\n"


def write_hf_token(path: str | Path, token: str) -> None:
    """Set ``[diarization].hf_token`` to ``token`` in the file at ``path``."""
    p = Path(path)
    original = p.read_text() if p.exists() else ""
    p.write_text(set_hf_token(original, token))
