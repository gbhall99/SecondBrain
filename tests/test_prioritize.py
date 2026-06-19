from datetime import date

from secondbrain.tasks import prioritize


def _task(**kw):
    base = dict(id=1, value=3, effort=3, due_date=None, goal_id=None)
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
