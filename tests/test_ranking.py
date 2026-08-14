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


def test_rank_returns_all_scored_and_caps_apply_at_render(conn, settings):
    settings.proactive.top_n = 2
    settings.proactive.per_kind_cap = 1
    sugs = [
        _sug("goal_alignment", 0.9, {"g": 1}),
        _sug("goal_alignment", 0.9, {"g": 2}),  # over per_kind cap → overflow
        _sug("connection", 0.9, {"c": 1}),
        _sug("stale_goal", 0.9, {"s": 1}),      # over top_n → overflow
    ]
    out = ranking.rank(conn, sugs, settings, now=NOW)
    assert len(out) == 4  # nothing is silently dropped any more
    visible, overflow = ranking.apply_caps(out, settings)
    assert [s.kind for s in visible] == ["goal_alignment", "connection"]
    assert len(overflow) == 2


def test_commitment_kinds_get_loose_per_kind_cap(conn, settings):
    settings.proactive.top_n = 10
    settings.proactive.per_kind_cap = 2
    sugs = [_sug("commitment_overdue", 0.9, {"e": i}) for i in range(6)]
    sugs += [_sug("connection", 0.9, {"c": i}) for i in range(4)]
    out = ranking.rank(conn, sugs, settings, now=NOW)
    visible, overflow = ranking.apply_caps(out, settings)
    # all 6 commitments are visible (cap 10); connections stop at per_kind_cap=2
    assert sum(1 for s in visible if s.kind == "commitment_overdue") == 6
    assert sum(1 for s in visible if s.kind == "connection") == 2
    assert len(overflow) == 2


def test_apply_caps_works_on_dict_rows(conn, settings):
    settings.proactive.top_n = 1
    rows = [{"kind": "connection", "id": 1}, {"kind": "connection", "id": 2}]
    visible, overflow = ranking.apply_caps(rows, settings)
    assert visible == [rows[0]] and overflow == [rows[1]]


def test_base_weights_cover_new_kinds_and_drop_dead_key():
    assert "stale_commitment" not in ranking.BASE_WEIGHT
    for kind in ("tasks_due", "plan_carryover", "goal_at_risk", "commitment_undated"):
        assert kind in ranking.BASE_WEIGHT


def test_snooze_kind_excludes(conn, settings):
    store.snooze_kind(conn, "connection", days=7)
    out = ranking.rank(conn, [_sug("connection", 0.9, {"a": 1})], settings, now=NOW)
    assert out == []


def test_snooze_hash_excludes_single_item(conn, settings):
    a = _sug("connection", 0.9, {"a": 1})
    b = _sug("connection", 0.9, {"a": 2})
    store.snooze_hash(conn, a.dedupe_hash, days=3)
    out = ranking.rank(conn, [a, b], settings, now=NOW)
    # only the snoozed item is hidden; its sibling survives
    assert [s.dedupe_hash for s in out] == [b.dedupe_hash]


def test_snoozed_kinds_parses_timestamps_not_lexical(conn):
    from secondbrain.storage import state

    # A stored no-milliseconds value sorts lexically AFTER a %f-style now
    # ("...00Z" > "...00.000Z"), which the old string compare read as "still
    # snoozed". Parsed properly, a past expiry is not snoozed.
    state.set_state(conn, store.SNOOZE_PREFIX + "connection", "2020-01-01T00:00:00Z")
    now_iso = NOW.strftime("%Y-%m-%dT%H:%M:%fZ")
    assert store.snoozed_kinds(conn, now_iso) == set()
    # and a genuinely-future expiry (any format) still snoozes
    state.set_state(conn, store.SNOOZE_PREFIX + "connection", "2999-01-01T00:00:00Z")
    assert store.snoozed_kinds(conn, now_iso) == {"connection"}


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
