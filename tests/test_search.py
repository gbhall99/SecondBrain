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


def test_semantic_error_does_not_break_search(conn, settings, monkeypatch):
    _seed(conn)
    settings.search.semantic_enabled = True
    from secondbrain.search import semantic

    def boom(*a, **k):
        raise RuntimeError("vec dim mismatch")

    monkeypatch.setattr(semantic, "search", boom)
    # Semantic blows up, but full-text results still come back (no exception).
    hits = combined.search(conn, "pricing", settings=settings)
    assert any("pricing" in h.text for h in hits)


def test_optout_does_not_shrink_results_below_available(conn, settings):
    # An opted-out speaker's segment ranks first; a non-opted match exists too.
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (1,'P','known',0,1)")
    af = models.insert_audio_file(
        conn, AudioFile(path="/b.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [
        Segment(t, af, 0.0, 1.0, "pricing secret", start_at="2026-06-16T09:00:00.000Z", speaker_id=1),
        Segment(t, af, 1.0, 2.0, "pricing public", start_at="2026-06-16T09:00:01.000Z"),
    ])
    hits = combined.search(conn, "pricing", limit=1, settings=settings)
    # The opted-out hit is filtered but the non-opted one still fills the page.
    assert len(hits) == 1 and "public" in hits[0].text
