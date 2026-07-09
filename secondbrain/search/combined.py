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
# When the strict AND-of-all-tokens query finds fewer hits than this, retry with
# the relaxed OR query so natural-language questions still ground to something.
_MIN_STRICT_HITS = 3


def _fulltext_with_fallback(
    conn: sqlite3.Connection, query: str, fetch: int, flt: dict
) -> list[SearchHit]:
    """Strict FTS first (exact phrases win), relaxed OR recall when it's thin.

    The strict query ANDs every token, which is right for keyword lookups
    ("canada pilot") but returns nothing for full questions ("What was the plan
    for the Canada pilot?"). If strict yields fewer than _MIN_STRICT_HITS, the
    stopword-stripped OR variant tops the list up — strict hits keep their rank
    ahead of relaxed-only ones.
    """
    strict = fulltext.search(conn, query, fetch, **flt)
    if len(strict) >= _MIN_STRICT_HITS:
        return strict
    relaxed = fulltext.search(conn, query, fetch, relaxed=True, **flt)
    have = {h.segment_id for h in strict}
    return strict + [h for h in relaxed if h.segment_id not in have]


def _safe_semantic(
    conn: sqlite3.Connection, query: str, limit: int, settings: Settings, flt: dict
) -> list[SearchHit]:
    """Semantic search that degrades to [] on any runtime error (dim mismatch,
    corrupt vec table, model failure) so it never takes down full-text search."""
    if not settings.search.semantic_enabled:
        return []
    try:
        return semantic.search(conn, query, limit, settings, **flt)
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
    since_utc: str | None = None,
    until_utc: str | None = None,
    speaker_id: int | None = None,
) -> list[SearchHit]:
    """Fused search. Optional filters (UTC time window, merge-resolved speaker)
    are pushed down into both engines — SQL WHERE for FTS, candidate widening
    for KNN — so a filtered page is exact rather than a post-filter over a
    capped pool that silently drops matches once the corpus outgrows it.
    Opted-out voices are excluded the same way (unattributed lines stay)."""
    settings = settings or get_settings()
    flt = {
        "since_utc": since_utc,
        "until_utc": until_utc,
        "speaker_id": speaker_id,
        "exclude_speaker_ids": registry.opted_out_speaker_ids(conn, settings),
    }

    if mode == "fulltext":
        return _fulltext_with_fallback(conn, query, limit, flt)[:limit]
    if mode == "semantic":
        return _safe_semantic(conn, query, limit, settings, flt)[:limit]

    ft = _fulltext_with_fallback(conn, query, limit, flt)
    sem = _safe_semantic(conn, query, limit, settings, flt)
    if not sem:
        return ft[:limit]

    scores: dict[int, float] = {}
    hits: dict[int, SearchHit] = {}
    for ranking in (ft, sem):
        for rank, hit in enumerate(ranking):
            scores[hit.segment_id] = scores.get(hit.segment_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            hits.setdefault(hit.segment_id, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ft_ids = {h.segment_id for h in ft}
    out: list[SearchHit] = []
    for seg_id, fused in ordered:
        hit = hits[seg_id]
        hit.score = round(fused, 6)  # RRF: higher is better
        if seg_id not in ft_ids:
            # Found by meaning only, not by the words themselves — lets the UI
            # label these "related" and say so when *no* literal match exists.
            hit.extra["semantic_only"] = True
        out.append(hit)
    return out[:limit]
