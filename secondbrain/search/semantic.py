"""Optional semantic search via sqlite-vec + a local embedding model.

Everything here degrades gracefully: if sqlite-vec or the embedding model is
unavailable (e.g. on a minimal CI box), :func:`is_available` returns False and
callers fall back to full-text search only. No data ever leaves the machine.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading

from secondbrain.config import Settings, get_settings
from secondbrain.storage.db import try_load_sqlite_vec
from secondbrain.storage.models import SearchHit

log = logging.getLogger(__name__)

_EMBED_DIM = 384  # bge-small / all-MiniLM family


def _serialize(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class Embedder:
    """Lazily-loaded local sentence embedder. Singleton-ish per process."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            name = "BAAI/bge-small-en-v1.5" if self.model_name == "bge-small" else self.model_name
            self._model = SentenceTransformer(name)
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


_embedder: Embedder | None = None
_embedder_unavailable_logged = False


def _get_embedder(settings: Settings) -> Embedder | None:
    global _embedder, _embedder_unavailable_logged
    if not settings.search.semantic_enabled:
        return None
    try:
        import sentence_transformers  # noqa: F401
    except Exception:  # noqa: BLE001 - optional backend: a broken native dep
        # (e.g. torchcodec failing to load its shared libs) raises RuntimeError/
        # OSError at import, not just ImportError. Semantic search is optional
        # (full-text still works), so degrade quietly and log once rather than
        # spamming a traceback for every processed chunk.
        if not _embedder_unavailable_logged:
            log.warning(
                "semantic search unavailable (embedding backend failed to load); "
                "using full-text search only",
                exc_info=True,
            )
            _embedder_unavailable_logged = True
        return None
    if _embedder is None:
        _embedder = Embedder(settings.search.embedding_model)
    return _embedder


def get_embedder(settings: Settings | None = None) -> Embedder | None:
    """Public accessor for the local text embedder (None if unavailable).

    Used by knowledge entity-resolution as well as semantic search. Independent of
    sqlite-vec (which is only needed for the transcript vector index).
    """
    return _get_embedder(settings or get_settings())


def is_available(conn: sqlite3.Connection, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return try_load_sqlite_vec(conn) and _get_embedder(settings) is not None


def ensure_index(conn: sqlite3.Connection) -> bool:
    """Create the vec0 virtual table if sqlite-vec is loadable. Returns success."""
    if not try_load_sqlite_vec(conn):
        return False
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS segment_vectors USING vec0(
            segment_id INTEGER PRIMARY KEY,
            embedding FLOAT[{_EMBED_DIM}]
        )
        """
    )
    return True


def index_segments(
    conn: sqlite3.Connection,
    segment_ids: list[int],
    texts: list[str],
    settings: Settings | None = None,
) -> int:
    """Embed and store vectors for the given segments. No-op if unavailable."""
    settings = settings or get_settings()
    embedder = _get_embedder(settings)
    if embedder is None or not ensure_index(conn):
        return 0
    vectors = embedder.encode(texts)
    conn.executemany(
        "INSERT OR REPLACE INTO segment_vectors(segment_id, embedding) VALUES (?, ?)",
        [(sid, _serialize(v)) for sid, v in zip(segment_ids, vectors, strict=False)],
    )
    return len(segment_ids)


def backfill_missing(
    conn: sqlite3.Connection,
    settings: Settings | None = None,
    *,
    batch_size: int = 64,
) -> int:
    """Embed transcript segments that predate the vector index. Idempotent.

    The worker indexes new segments as they're transcribed, but anything
    recorded before semantic search was available (or while the embedding
    backend was broken) has no vector, so natural-language questions can't
    ground against it. Returns how many segments were indexed; 0 when the
    backend/extension is unavailable or nothing is missing.
    """
    settings = settings or get_settings()
    if _get_embedder(settings) is None or not ensure_index(conn):
        return 0
    total = 0
    while True:
        rows = conn.execute(
            """
            SELECT id, text FROM transcript_segments
            WHERE id NOT IN (SELECT segment_id FROM segment_vectors)
            ORDER BY id LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
        if not rows:
            break
        done = index_segments(conn, [r["id"] for r in rows], [r["text"] for r in rows], settings)
        if not done:  # backend vanished mid-run: stop rather than spin
            break
        total += done
    return total


def start_background_backfill(settings: Settings | None = None) -> threading.Thread | None:
    """Kick off :func:`backfill_missing` on a daemon thread (own connection).

    Called at web-server startup so historical transcripts become semantically
    searchable without blocking startup or requiring a manual command. Safe to
    call unconditionally: it no-ops when semantic search is disabled and
    swallows every failure (full-text search keeps working regardless).
    """
    settings = settings or get_settings()
    if not settings.search.semantic_enabled:
        return None

    def _run() -> None:
        try:
            from secondbrain.storage.db import db_session

            with db_session(settings=settings) as conn:
                n = backfill_missing(conn, settings)
            if n:
                log.info("semantic backfill: indexed %d segment(s)", n)
        except Exception:  # noqa: BLE001 - best-effort background task
            log.warning("semantic backfill failed; full-text search unaffected", exc_info=True)

    t = threading.Thread(target=_run, name="sb-semantic-backfill", daemon=True)
    t.start()
    return t


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    settings: Settings | None = None,
    *,
    since_utc: str | None = None,
    until_utc: str | None = None,
    speaker_id: int | None = None,
    exclude_speaker_ids: set[int] | frozenset[int] | None = None,
) -> list[SearchHit]:
    """KNN search, optionally filtered by time window / speaker.

    vec0 KNN can't push arbitrary WHERE clauses into the index, so filters are
    applied to the k nearest rows and — when the page doesn't fill — the query
    retries with a widened k until either ``limit`` hits pass the filters, the
    whole index has been scanned, or the tail of the candidates is already
    beyond the distance ceiling (every later neighbour is farther still). A
    filtered page is therefore exact, not "whatever survived from the first k".
    """
    settings = settings or get_settings()
    embedder = _get_embedder(settings)
    if embedder is None or not ensure_index(conn):
        return []
    qvec = _serialize(embedder.encode([query])[0])
    # KNN always returns k rows no matter how far away they are; without a
    # ceiling an unmatched query "finds" arbitrary segments and the caller can
    # never tell found from not-found. (vec0 can't filter on distance in SQL.)
    ceiling = settings.search.semantic_max_distance
    exclude = frozenset(exclude_speaker_ids or ())
    filtered = bool(since_utc or until_utc or exclude) or speaker_id is not None

    def keep(r: sqlite3.Row) -> bool:
        if float(r["distance"]) > ceiling:
            return False
        if since_utc and (r["start_at"] is None or r["start_at"] < since_utc):
            return False
        if until_utc and (r["start_at"] is None or r["start_at"] >= until_utc):
            return False
        spk = r["speaker_id"]
        if speaker_id is not None and spk != speaker_id:
            return False
        return spk is None or spk not in exclude

    # Start above `limit` when filters will drop rows so one query usually
    # suffices; each retry quadruples k (few round-trips even on big corpora).
    k = max(limit * 4, 16) if filtered else limit
    while True:
        rows = conn.execute(
            """
            SELECT v.segment_id, v.distance, s.audio_file_id, s.text,
                   s.start_offset_s, s.end_offset_s, s.start_at, s.speaker_id
            FROM segment_vectors v
            JOIN transcript_segments s ON s.id = v.segment_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (qvec, k),
        ).fetchall()
        kept = [r for r in rows if keep(r)]
        if len(kept) >= limit or len(rows) < k:
            break  # page full, or the entire index has been considered
        if rows and float(rows[-1]["distance"]) > ceiling:
            break  # candidates already past the ceiling; the rest are farther
        k *= 4
    return [
        SearchHit(
            segment_id=r["segment_id"],
            audio_file_id=r["audio_file_id"],
            text=r["text"],
            start_offset_s=r["start_offset_s"],
            end_offset_s=r["end_offset_s"],
            start_at=r["start_at"],
            score=float(r["distance"]),  # embedding distance: lower is better
            extra={"distance": round(float(r["distance"]), 4)},
        )
        for r in kept[:limit]
    ]
