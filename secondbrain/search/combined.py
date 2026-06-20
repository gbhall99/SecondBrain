"""Unified search: fuse full-text and (optional) semantic results.

Uses Reciprocal Rank Fusion (RRF), which combines rankings without needing the
two scoring scales (bm25 vs cosine distance) to be comparable.
"""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.search import fulltext, semantic
from secondbrain.speaker import registry
from secondbrain.storage.models import SearchHit

_RRF_K = 60


def _drop_opted_out(
    conn: sqlite3.Connection, hits: list[SearchHit], settings: Settings
) -> list[SearchHit]:
    """Filter out hits whose segment belongs to an opted-out speaker (privacy)."""
    opted = registry.opted_out_speaker_ids(conn, settings)
    if not opted:
        return hits
    spk = registry.segments_speaker_map(conn, [h.segment_id for h in hits])
    return [h for h in hits if spk.get(h.segment_id) not in opted]


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    *,
    settings: Settings | None = None,
    mode: str = "auto",  # auto | fulltext | semantic
) -> list[SearchHit]:
    settings = settings or get_settings()

    if mode == "fulltext":
        return _drop_opted_out(conn, fulltext.search(conn, query, limit), settings)
    if mode == "semantic":
        return _drop_opted_out(conn, semantic.search(conn, query, limit, settings), settings)

    ft = fulltext.search(conn, query, limit)
    sem = semantic.search(conn, query, limit, settings) if settings.search.semantic_enabled else []
    if not sem:
        return _drop_opted_out(conn, ft, settings)

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
    return _drop_opted_out(conn, out, settings)[:limit]
