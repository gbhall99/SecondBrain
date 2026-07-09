import pytest

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


def test_natural_language_question_falls_back_to_or_query(conn, settings):
    _seed(conn)
    # Strict AND-of-all-tokens matches nothing for a full question; the relaxed
    # OR fallback must still ground it against the transcript.
    assert fulltext.search(conn, "What was the plan for the pricing model?") == []
    hits = combined.search(conn, "What was the plan for the pricing model?", settings=settings)
    assert any("pricing" in h.text for h in hits)


def test_relaxed_query_strips_stopwords_and_dedupes():
    q = fulltext._relaxed_fts_query("What did I say about the vendor, the vendor demo?")
    assert q == '"vendor" OR "demo"'
    # All-stopword questions keep the raw tokens rather than matching nothing.
    assert fulltext._relaxed_fts_query("what did i do") == '"what" OR "did" OR "i" OR "do"'
    assert fulltext._relaxed_fts_query("!!!") == '""'


def test_relaxed_search_is_injection_safe(conn):
    _seed(conn)
    assert fulltext.search(conn, 'pricing"; DROP OR NEAR(', relaxed=True) is not None


def test_strict_hits_rank_ahead_of_relaxed_fallback(conn, settings):
    _seed(conn)
    hits = combined.search(conn, "vendor demo", settings=settings)
    # Both tokens hit the vendor segment first; OR-recall may add the other.
    assert hits and "vendor" in hits[0].text


def test_semantic_distance_ceiling_drops_unrelated(conn, settings, monkeypatch):
    """KNN always returns k rows; the ceiling keeps 'no matches' honest."""
    pytest.importorskip("sqlite_vec")
    from secondbrain.search import semantic

    if not semantic.ensure_index(conn):
        pytest.skip("sqlite-vec extension not loadable in this sqlite build")
    _seed(conn)
    dim = semantic._EMBED_DIM

    class FakeEmbedder:
        def encode(self, texts):
            out = []
            for t in texts:
                v = [0.0] * dim
                # "pricing" texts (and the query) share a direction (d=0);
                # everything else is orthogonal (d=sqrt(2) > ceiling).
                v[0 if "pricing" in t else 1] = 1.0
                out.append(v)
            return out

    settings.search.semantic_enabled = True
    monkeypatch.setattr(semantic, "_get_embedder", lambda s: FakeEmbedder())
    rows = conn.execute("SELECT id, text FROM transcript_segments").fetchall()
    semantic.index_segments(conn, [r["id"] for r in rows], [r["text"] for r in rows], settings)

    hits = semantic.search(conn, "pricing", 10, settings)
    assert [h.text for h in hits] == ["we should revisit the pricing model next quarter"]
    assert hits[0].extra["distance"] == 0.0  # surfaced for clients
    # Loosening the ceiling brings distant neighbours back (it's tunable).
    settings.search.semantic_max_distance = 2.0
    assert len(semantic.search(conn, "pricing", 10, settings)) == 2


def test_auto_mode_flags_semantic_only_hits(conn, settings, monkeypatch):
    _seed(conn)
    settings.search.semantic_enabled = True
    from secondbrain.search import semantic
    from secondbrain.storage.models import SearchHit

    sem_hit = SearchHit(
        segment_id=2, audio_file_id=1, text="the vendor demo is scheduled for monday",
        start_offset_s=2.0, end_offset_s=4.0, start_at="2026-06-16T09:00:02.000Z",
        score=0.4, extra={"distance": 0.4},
    )
    monkeypatch.setattr(semantic, "search", lambda *a, **k: [sem_hit])
    hits = combined.search(conn, "pricing", settings=settings)
    by_id = {h.segment_id: h for h in hits}
    assert by_id[2].extra.get("semantic_only") is True  # matched by meaning only
    assert "semantic_only" not in by_id[1].extra  # literal word match keeps clean


def test_backfill_missing_embeds_unindexed_segments(conn, settings, monkeypatch):
    pytest.importorskip("sqlite_vec")
    from secondbrain.search import semantic

    if not semantic.ensure_index(conn):
        pytest.skip("sqlite-vec extension not loadable in this sqlite build")
    _seed(conn)

    class FakeEmbedder:
        def encode(self, texts):
            return [[0.1] * semantic._EMBED_DIM for _ in texts]

    settings.search.semantic_enabled = True
    monkeypatch.setattr(semantic, "_get_embedder", lambda s: FakeEmbedder())
    assert semantic.backfill_missing(conn, settings, batch_size=1) == 2
    assert semantic.backfill_missing(conn, settings) == 0  # idempotent
    n = conn.execute("SELECT COUNT(*) AS n FROM segment_vectors").fetchone()["n"]
    assert n == 2


def test_backfill_noop_when_backend_unavailable(conn, settings, monkeypatch):
    from secondbrain.search import semantic

    _seed(conn)
    monkeypatch.setattr(semantic, "_get_embedder", lambda s: None)
    assert semantic.backfill_missing(conn, settings) == 0


def test_fulltext_filters_run_in_sql_before_limit(conn):
    """A filtered page is exact even when unfiltered hits own every top rank.

    The old design post-filtered a capped candidate pool (limit*4), so at
    corpus scale a speaker/date filter silently dropped matches. Filters now
    run in the SQL WHERE, before LIMIT.
    """
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (9, 'Dana', 'known', 0)")
    af = models.insert_audio_file(
        conn, AudioFile(path="/c.flac", started_at="2026-06-01T12:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    # 100 keyword-dense decoys (best bm25 ranks, no speaker) drown out 5 long
    # lines from Dana that an over-fetch-then-filter approach would never see.
    long_text = "we talked about pricing while wandering through many other unrelated topics"
    models.insert_segments(
        conn,
        [Segment(t, af, i, i + 1.0, "pricing", start_at="2026-06-01T12:00:00.000Z")
         for i in range(100)]
        + [Segment(t, af, 300.0 + i, 301.0 + i, f"{long_text} take {i}",
                   start_at="2026-06-20T12:00:00.000Z", speaker_id=9)
           for i in range(5)],
    )
    hits = fulltext.search(conn, "pricing", 20, speaker_id=9)
    assert len(hits) == 5
    # Time window: UTC bounds, [since, until) — exact under the same crowding.
    hits = fulltext.search(conn, "pricing", 20, since_utc="2026-06-10T00:00:00")
    assert len(hits) == 5
    assert fulltext.search(conn, "pricing", 20, until_utc="2026-06-01T12:00:00") == []
    # Excluding a voice keeps unattributed (NULL-speaker) lines.
    hits = fulltext.search(conn, "pricing", 200, exclude_speaker_ids={9})
    assert len(hits) == 100


def test_semantic_filtered_search_widens_k_until_page_fills(conn, settings, monkeypatch):
    """KNN pre-filtering must not stop at the first k candidates: when a filter
    rejects them all, the query retries with a wider k until the page fills or
    the index is exhausted."""
    pytest.importorskip("sqlite_vec")
    import math

    from secondbrain.search import semantic

    if not semantic.ensure_index(conn):
        pytest.skip("sqlite-vec extension not loadable in this sqlite build")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (9, 'Dana', 'known', 0)")
    af = models.insert_audio_file(
        conn, AudioFile(path="/d.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    # 29 nearer segments from nobody; Dana's line is the FARTHEST (rank 30),
    # well beyond the initial filtered candidate page of max(limit*4, 16).
    models.insert_segments(
        conn,
        [Segment(t, af, float(i), i + 1.0, f"filler {i}", start_at="2026-06-16T09:00:00.000Z")
         for i in range(29)]
        + [Segment(t, af, 40.0, 41.0, "dana target",
                   start_at="2026-06-16T09:00:00.000Z", speaker_id=9)],
    )
    dim = semantic._EMBED_DIM

    class FakeEmbedder:
        def encode(self, texts):
            out = []
            for txt in texts:
                v = [0.0] * dim
                if txt.startswith("filler"):
                    ang = 0.01 * (int(txt.split()[1]) + 1)   # distances ≈ 0.01…0.29
                elif txt == "dana target":
                    ang = 0.5                                # ≈ 0.49, inside the 0.8 ceiling
                else:  # the query
                    ang = 0.0
                v[0], v[1] = math.cos(ang), math.sin(ang)
                out.append(v)
            return out

    settings.search.semantic_enabled = True
    monkeypatch.setattr(semantic, "_get_embedder", lambda s: FakeEmbedder())
    rows = conn.execute("SELECT id, text FROM transcript_segments ORDER BY id").fetchall()
    semantic.index_segments(conn, [r["id"] for r in rows], [r["text"] for r in rows], settings)

    hits = semantic.search(conn, "q", 1, settings, speaker_id=9)
    assert [h.text for h in hits] == ["dana target"]
    # Date window pre-filtering follows the same path.
    assert semantic.search(conn, "q", 1, settings, until_utc="2026-06-01T00:00:00") == []
    hits = semantic.search(conn, "q", 40, settings, since_utc="2026-06-16T00:00:00")
    assert len(hits) == 30  # everything, not just the first candidate page


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
