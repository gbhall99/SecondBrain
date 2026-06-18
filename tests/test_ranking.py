from datetime import UTC, datetime

from secondbrain.proactive import ranking, store
from secondbrain.proactive.detectors import Suggestion

NOW = datetime(2026, 6, 16, 12, 0, 0, tzinfo=UTC)


def _sug(kind, conf, key):
    return Suggestion(kind=kind, title=kind, detail="", confidence=conf, payload={"key": key})


def test_confidence_floor_drops_weak(conn, settings):
    settings.proactive.confidence_floor = 0.5
    out = ranking.rank(conn, [_sug("connection", 0.2, {"a": 1})], settings, now=NOW)
    assert out == []


def test_top_n_and_per_kind_cap(conn, settings):
    settings.proactive.top_n = 2
    settings.proactive.per_kind_cap = 1
    sugs = [
        _sug("commitment_overdue", 0.9, {"e": 1}),
        _sug("commitment_overdue", 0.9, {"e": 2}),  # capped (per_kind=1)
        _sug("goal_alignment", 0.9, {"g": 1}),
        _sug("connection", 0.9, {"c": 1}),          # cut by top_n=2
    ]
    out = ranking.rank(conn, sugs, settings, now=NOW)
    assert len(out) == 2
    assert [s.kind for s in out] == ["commitment_overdue", "goal_alignment"]


def test_snooze_kind_excludes(conn, settings):
    store.snooze_kind(conn, "connection", days=7)
    out = ranking.rank(conn, [_sug("connection", 0.9, {"a": 1})], settings, now=NOW)
    assert out == []


def test_cross_day_suppression(conn, settings):
    s = _sug("connection", 0.9, {"a": 1})
    # a previously-dismissed suggestion with the same dedupe hash suppresses it
    conn.execute(
        "INSERT INTO suggestions (digest_date, kind, title, importance, confidence, status, dedupe_hash) "
        "VALUES ('2026-06-10','connection','x',0.5,0.9,'dismissed',?)",
        (s.dedupe_hash,),
    )
    out = ranking.rank(conn, [s], settings, now=NOW)
    assert out == []


def test_feedback_weight_changes_order(conn, settings):
    settings.proactive.top_n = 5
    settings.proactive.per_kind_cap = 5
    # down-vote connections so they rank below an equal-base goal item
    store.bump_feedback_weight(conn, "connection", "down")
    store.bump_feedback_weight(conn, "connection", "down")
    out = ranking.rank(
        conn,
        [_sug("connection", 0.9, {"a": 1}), _sug("connection", 0.9, {"a": 2})],
        settings, now=NOW,
    )
    assert all(s.importance < 0.6 for s in out)  # base 0.6 * <1 weight
