"""Unified search: fuse full-text and (optional) semantic results.

Uses Reciprocal Rank Fusion (RRF), which combines rankings without needing the
two scoring scales (bm25 vs cosine distance) to be comparable.
"""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.search import fulltext, semantic
from secondbrain.storage.models import SearchHit

_RRF_K = 60


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
        return fulltext.search(conn, query, limit)
    if mode == "semantic":
        return semantic.search(conn, query, limit, settings)

    ft = fulltext.search(conn, query, limit)
    sem = semantic.search(conn, query, limit, settings) if settings.search.semantic_enabled else []
    if not sem:
        return ft

    scores: dict[int, float] = {}
    hits: dict[int, SearchHit] = {}
    for ranking in (ft, sem):
        for rank, hit in enumerate(ranking):
            scores[hit.segment_id] = scores.get(hit.segment_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            hits.setdefault(hit.segment_id, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[SearchHit] = []
    for seg_id, fused in ordered[:limit]:
        hit = hits[seg_id]
        hit.score = round(fused, 6)  # RRF: higher is better
        out.append(hit)
    return out
