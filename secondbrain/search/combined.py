"""Unified search: fuse full-text and (optional) semantic results.

Uses Reciprocal Rank Fusion (RRF), which combines rankings without needing the
two scoring scales (bm25 vs cosine distance) to be comparable.
"""

from __future__ import annotations

import logging
import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.search import fulltext, semantic
from secondbrain.speaker import registry
from secondbrain.storage.models import SearchHit

log = logging.getLogger(__name__)

_RRF_K = 60
# Over-fetch multiplier when we'll be dropping opted-out hits, so the final LIMIT
# still yields ~limit visible results instead of an under-filled page.
_OPTOUT_MARGIN = 4


def _safe_semantic(
    conn: sqlite3.Connection, query: str, limit: int, settings: Settings
) -> list[SearchHit]:
    """Semantic search that degrades to [] on any runtime error (dim mismatch,
    corrupt vec table, model failure) so it never takes down full-text search."""
    if not settings.search.semantic_enabled:
        return []
    try:
        return semantic.search(conn, query, limit, settings)
    except Exception:  # noqa: BLE001 - best-effort; fall back to full-text
        log.warning("semantic search failed; falling back to full-text only", exc_info=True)
        return []


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    *,
    settings: Settings | None = None,
    mode: str = "auto",  # auto | fulltext | semantic
) -> list[SearchHit]:
    settings = settings or get_settings()
    opted = registry.opted_out_speaker_ids(conn, settings)
    # Over-fetch before opt-out filtering so LIMIT counts only visible rows.
    fetch = limit * _OPTOUT_MARGIN if opted else limit

    def drop(hits: list[SearchHit]) -> list[SearchHit]:
        if not opted:
            return hits
        spk = registry.segments_speaker_map(conn, [h.segment_id for h in hits])
        return [h for h in hits if spk.get(h.segment_id) not in opted]

    if mode == "fulltext":
        return drop(fulltext.search(conn, query, fetch))[:limit]
    if mode == "semantic":
        return drop(_safe_semantic(conn, query, fetch, settings))[:limit]

    ft = fulltext.search(conn, query, fetch)
    sem = _safe_semantic(conn, query, fetch, settings)
    if not sem:
        return drop(ft)[:limit]

    scores: dict[int, float] = {}
    hits: dict[int, SearchHit] = {}
    for ranking in (ft, sem):
        for rank, hit in enumerate(ranking):
            scores[hit.segment_id] = scores.get(hit.segment_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            hits.setdefault(hit.segment_id, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[SearchHit] = []
    for seg_id, fused in ordered:
        hit = hits[seg_id]
        hit.score = round(fused, 6)  # RRF: higher is better
        out.append(hit)
    return drop(out)[:limit]
