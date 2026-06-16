from secondbrain.search import combined, fulltext
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def _seed(conn):
    af = models.insert_audio_file(
        conn, AudioFile(path="/a.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [
            Segment(t, af, 0.0, 2.0, "we should revisit the pricing model next quarter",
                    start_at="2026-06-16T09:00:00.000Z"),
            Segment(t, af, 2.0, 4.0, "the vendor demo is scheduled for monday",
                    start_at="2026-06-16T09:00:02.000Z"),
        ],
    )


def test_fulltext_search_finds_terms(conn):
    _seed(conn)
    hits = fulltext.search(conn, "pricing")
    assert len(hits) == 1
    assert "pricing" in hits[0].text


def test_fulltext_query_is_injection_safe(conn):
    _seed(conn)
    # Punctuation / FTS operators in user input must not raise.
    assert fulltext.search(conn, 'pricing"; DROP') is not None


def test_combined_falls_back_to_fulltext_when_semantic_disabled(conn, settings):
    _seed(conn)
    hits = combined.search(conn, "vendor demo", settings=settings)
    assert any("vendor" in h.text for h in hits)
