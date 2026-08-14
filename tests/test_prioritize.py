from datetime import date

from secondbrain.tasks import prioritize


def _task(**kw):
    base = {"id": 1, "value": 3, "effort": 3, "due_date": None, "goal_id": None}
    base.update(kw)
    return base


TODAY = date(2026, 6, 16)


def test_quadrants(conn, settings):
    urgent_important = _task(value=5, due_date="2026-06-17")
    assert prioritize.quadrant(conn, urgent_important, settings, TODAY) == prioritize.DO
    important_only = _task(value=5, due_date="2026-09-01")
    assert prioritize.quadrant(conn, important_only, settings, TODAY) == prioritize.SCHEDULE
    urgent_only = _task(value=2, due_date="2026-06-17")
    assert prioritize.quadrant(conn, urgent_only, settings, TODAY) == prioritize.DELEGATE
    neither = _task(value=2, due_date=None)
    assert prioritize.quadrant(conn, neither, settings, TODAY) == prioritize.ELIMINATE


def test_score_orders_urgent_important_first(conn, settings):
    hi = _task(value=5, due_date="2026-06-16")
    lo = _task(value=2, due_date=None)
    assert prioritize.score(conn, hi, settings, TODAY) > prioritize.score(conn, lo, settings, TODAY)


def test_graduated_importance(conn, settings):
    # value >= 4 alone
    assert prioritize._is_important(conn, _task(value=4), settings)
    assert not prioritize._is_important(conn, _task(value=3), settings)
    # priority-1 goal
    g1 = conn.execute("INSERT INTO goals (title, priority) VALUES ('p1', 1)").lastrowid
    assert prioritize._is_important(conn, _task(value=1, goal_id=g1), settings)
    # priority-2 goal needs decent value
    g2 = conn.execute("INSERT INTO goals (title, priority) VALUES ('p2', 2)").lastrowid
    assert prioritize._is_important(conn, _task(value=3, goal_id=g2), settings)
    assert not prioritize._is_important(conn, _task(value=2, goal_id=g2), settings)
    # a spoken commitment (promoted from a conversation edge) is important
    assert prioritize._is_important(conn, _task(value=1, source_edge_id=7), settings)


def test_commitment_sourced_task_lands_in_do_when_urgent(conn, settings):
    t = _task(value=2, due_date="2026-06-17", source_edge_id=7)
    assert prioritize.quadrant(conn, t, settings, TODAY) == prioritize.DO


def test_quick_win_bonus_is_multiplicative(conn, settings):
    slow = _task(value=3, effort=3)
    quick = _task(value=3, effort=2)
    s_slow = prioritize.score(conn, slow, settings, TODAY)
    s_quick = prioritize.score(conn, quick, settings, TODAY)
    assert s_quick == round(s_slow * prioritize.QUICK_WIN_FACTOR, 4)


def test_overdue_urgency_grows_mildly_and_caps(conn, settings):
    day = _task(due_date="2026-06-15")       # 1 day overdue
    week = _task(due_date="2026-06-06")      # 10 days overdue
    ancient = _task(due_date="2020-01-01")   # years overdue
    f1 = prioritize._urgency_factor(day, TODAY)
    f10 = prioritize._urgency_factor(week, TODAY)
    fmax = prioritize._urgency_factor(ancient, TODAY)
    assert f1 == 1.32 and f10 == 1.5
    assert fmax == 1.6  # capped


def test_score_breakdown_explains_the_score(conn, settings):
    t = _task(value=5, effort=2, due_date="2026-06-16")
    parts = prioritize.score_breakdown(conn, t, settings, TODAY)
    assert {"base", "goal", "quadrant", "quadrant_weight", "urgency",
            "quick_win", "score"} <= set(parts)
    expected = round(parts["base"] * parts["goal"] * parts["quadrant_weight"]
                     * parts["urgency"] * parts["quick_win"], 4)
    assert parts["score"] == expected
    assert parts["score"] == prioritize.score(conn, t, settings, TODAY)
