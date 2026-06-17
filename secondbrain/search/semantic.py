"""Optional semantic search via sqlite-vec + a local embedding model.

Everything here degrades gracefully: if sqlite-vec or the embedding model is
unavailable (e.g. on a minimal CI box), :func:`is_available` returns False and
callers fall back to full-text search only. No data ever leaves the machine.
"""

from __future__ import annotations

import sqlite3
import struct

from secondbrain.config import Settings, get_settings
from secondbrain.storage.db import try_load_sqlite_vec
from secondbrain.storage.models import SearchHit

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


def _get_embedder(settings: Settings) -> Embedder | None:
    global _embedder
    if not settings.search.semantic_enabled:
        return None
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
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


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    settings: Settings | None = None,
) -> list[SearchHit]:
    settings = settings or get_settings()
    embedder = _get_embedder(settings)
    if embedder is None or not ensure_index(conn):
        return []
    qvec = _serialize(embedder.encode([query])[0])
    rows = conn.execute(
        """
        SELECT v.segment_id, v.distance, s.audio_file_id, s.text,
               s.start_offset_s, s.end_offset_s, s.start_at
        FROM segment_vectors v
        JOIN transcript_segments s ON s.id = v.segment_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (qvec, limit),
    ).fetchall()
    return [
        SearchHit(
            segment_id=r["segment_id"],
            audio_file_id=r["audio_file_id"],
            text=r["text"],
            start_offset_s=r["start_offset_s"],
            end_offset_s=r["end_offset_s"],
            start_at=r["start_at"],
            score=float(r["distance"]),  # cosine distance: lower is better
        )
        for r in rows
    ]
