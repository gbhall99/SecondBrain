"""service.search date-range filtering (post-filter over search hits)."""

from __future__ import annotations

from secondbrain.query import service
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def _seed(conn):
    af = models.insert_audio_file(
        conn, AudioFile(path="/a.flac", started_at="2026-06-10T09:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [
            Segment(t, af, 0.0, 2.0, "pricing model discussion early",
                    start_at="2026-06-10T09:00:00.000Z"),
            Segment(t, af, 2.0, 4.0, "pricing model discussion late",
                    start_at="2026-06-20T09:00:00.000Z"),
        ],
    )


def test_search_without_range_returns_all(conn, settings):
    _seed(conn)
    hits = service.search(conn, "pricing", settings=settings)
    assert len(hits) == 2


def test_search_since_filters_earlier(conn, settings):
    _seed(conn)
    hits = service.search(conn, "pricing", settings=settings, since="2026-06-15")
    texts = [h["text"] for h in hits]
    assert texts == ["pricing model discussion late"]


def test_search_until_filters_later(conn, settings):
    _seed(conn)
    hits = service.search(conn, "pricing", settings=settings, until="2026-06-15")
    texts = [h["text"] for h in hits]
    assert texts == ["pricing model discussion early"]


def test_search_window_excludes_both_ends(conn, settings):
    _seed(conn)
    hits = service.search(conn, "pricing", settings=settings, since="2026-06-12", until="2026-06-18")
    assert hits == []
