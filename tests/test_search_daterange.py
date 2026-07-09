"""service.search date-range / speaker filtering (pushed into the search SQL,
so filtered pages are exact at any corpus size — never a post-filter over a
capped candidate pool)."""

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


def test_filtered_search_is_exact_at_corpus_scale(conn, settings):
    """Regression: filters must not silently drop matches once better-ranked
    unfiltered hits outnumber the old limit*4 candidate pool."""
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (9, 'Dana', 'known', 0)")
    af = models.insert_audio_file(
        conn, AudioFile(path="/big.flac", started_at="2026-06-01T12:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    long_text = "we talked about pricing while wandering through many other unrelated topics"
    models.insert_segments(
        conn,
        # 100 keyword-dense decoys own the top bm25 ranks…
        [Segment(t, af, i, i + 1.0, "pricing", start_at="2026-06-01T12:00:00.000Z")
         for i in range(100)]
        # …while every hit the filters actually want ranks below them.
        + [Segment(t, af, 300.0 + i, 301.0 + i, f"{long_text} take {i}",
                   start_at="2026-06-20T12:00:00.000Z", speaker_id=9)
           for i in range(6)],
    )
    hits = service.search(conn, "pricing", limit=20, settings=settings, speaker=9)
    assert len(hits) == 6
    assert all(h["speaker"] == "Dana" for h in hits)
    hits = service.search(conn, "pricing", limit=20, settings=settings, since="2026-06-15")
    assert len(hits) == 6
    # And the unfiltered page still fills to its limit.
    assert len(service.search(conn, "pricing", limit=20, settings=settings)) == 20
