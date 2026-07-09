from secondbrain.tasks import planner, store


def test_propose_fits_capacity_and_accept_schedules(conn, settings):
    settings.tasks.daily_capacity_minutes = 60
    a = store.create_task(conn, title="A", estimate_minutes=30, value=5)
    b = store.create_task(conn, title="B", estimate_minutes=30, value=4)
    store.create_task(conn, title="C", estimate_minutes=30, value=1)  # over capacity → dropped

    day = planner.propose_day(conn, date="2026-06-16", settings=settings)
    assert day["status"] == "proposed"
    assert len(day["task_ids"]) == 2          # 60m capacity / 30m each
    assert a in day["task_ids"] and b in day["task_ids"]

    accepted = planner.accept_day(conn, "2026-06-16")
    assert accepted["status"] == "accepted"
    assert store.get_task(conn, a)["scheduled_for"] == "2026-06-16"
    assert store.get_task(conn, a)["status"] == "scheduled"


def test_blocked_tasks_excluded_from_plan(conn, settings):
    a = store.create_task(conn, title="A", estimate_minutes=10)
    b = store.create_task(conn, title="B", estimate_minutes=10)
    store.add_dependency(conn, b, a)  # B blocked until A done
    day = planner.propose_day(conn, date="2026-06-16", settings=settings)
    assert b not in day["task_ids"]
    assert a in day["task_ids"]


def test_status_blocked_task_not_scheduled(conn, settings):
    a = store.create_task(conn, title="A", estimate_minutes=10)
    b = store.create_task(conn, title="B", estimate_minutes=10)
    store.set_status(conn, b, "blocked")  # explicitly held back by the user
    day = planner.propose_day(conn, date="2026-06-16", settings=settings)
    assert a in day["task_ids"] and b not in day["task_ids"]


def test_accept_preserves_in_progress(conn, settings):
    a = store.create_task(conn, title="A", estimate_minutes=10)
    store.set_status(conn, a, "in_progress")
    planner.propose_day(conn, date="2026-06-16", settings=settings)
    planner.accept_day(conn, "2026-06-16")
    t = store.get_task(conn, a)
    assert t["status"] == "in_progress"          # not clobbered back to 'scheduled'
    assert t["scheduled_for"] == "2026-06-16"    # but the day is still set


def test_accept_writes_wellformed_updated_at(conn, settings):
    from datetime import datetime

    a = store.create_task(conn, title="A", estimate_minutes=10)
    planner.propose_day(conn, date="2026-06-16", settings=settings)
    planner.accept_day(conn, "2026-06-16")
    ts = store.get_task(conn, a)["updated_at"]
    # The old inline strftime ('%H:%M:%fZ') dropped the seconds field and
    # produced timestamps fromisoformat rejects; utcnow_iso must round-trip.
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.year == datetime.now().year


def test_repropose_returns_dropped_tasks_to_backlog(conn, settings):
    a = store.create_task(conn, title="A", estimate_minutes=30, value=5)
    b = store.create_task(conn, title="B", estimate_minutes=30, value=4)
    planner.propose_day(conn, date="2026-06-16", capacity_minutes=60, settings=settings)
    planner.accept_day(conn, "2026-06-16")
    assert store.get_task(conn, b)["status"] == "scheduled"

    # Shrink capacity: only A fits, so B must lose its stale 'scheduled' state.
    day = planner.propose_day(conn, date="2026-06-16", capacity_minutes=30, settings=settings)
    assert day["task_ids"] == [a]
    t = store.get_task(conn, b)
    assert t["status"] == "backlog" and t["scheduled_for"] is None
    # A stayed planned and untouched.
    assert store.get_task(conn, a)["status"] == "scheduled"

    # An in-progress task dropped from the plan keeps its status and day.
    store.set_status(conn, b, "in_progress")
    store.update_task(conn, b, scheduled_for="2026-06-16")
    planner.propose_day(conn, date="2026-06-16", capacity_minutes=30, settings=settings)
    t = store.get_task(conn, b)
    assert t["status"] == "in_progress" and t["scheduled_for"] == "2026-06-16"


def test_release_stale_scheduled_only_touches_past_scheduled(conn, settings):
    past = store.create_task(conn, title="left behind")
    today_t = store.create_task(conn, title="planned today")
    prog = store.create_task(conn, title="still working on it")
    store.update_task(conn, past, status="scheduled", scheduled_for="2026-06-15")
    store.update_task(conn, today_t, status="scheduled", scheduled_for="2026-06-16")
    store.update_task(conn, prog, status="in_progress", scheduled_for="2026-06-15")

    assert store.release_stale_scheduled(conn, "2026-06-16") == 1
    t = store.get_task(conn, past)
    assert t["status"] == "backlog" and t["scheduled_for"] is None
    # today's schedule and in-progress work are untouched
    assert store.get_task(conn, today_t)["status"] == "scheduled"
    p = store.get_task(conn, prog)
    assert p["status"] == "in_progress" and p["scheduled_for"] == "2026-06-15"


def test_propose_releases_previous_days_unfinished_plan(conn, settings):
    a = store.create_task(conn, title="A", estimate_minutes=10)
    planner.propose_day(conn, date="2026-06-16", settings=settings)
    planner.accept_day(conn, "2026-06-16")
    assert store.get_task(conn, a)["status"] == "scheduled"

    # Next morning: proposing releases yesterday's leftover before planning,
    # so the task competes for today instead of staying pinned to the past.
    day = planner.propose_day(conn, date="2026-06-17", settings=settings)
    assert a in day["task_ids"]
    t = store.get_task(conn, a)
    assert t["status"] == "backlog" and t["scheduled_for"] is None


def test_remove_from_day_releases_task_and_keeps_plan(conn, settings):
    a = store.create_task(conn, title="A", estimate_minutes=10)
    b = store.create_task(conn, title="B", estimate_minutes=10)
    planner.propose_day(conn, date="2026-06-16", settings=settings)
    planner.accept_day(conn, "2026-06-16")

    day = planner.remove_from_day(conn, b, date="2026-06-16")
    assert day["task_ids"] == [a]
    assert day["status"] == "accepted"          # plan status survives the edit
    t = store.get_task(conn, b)
    assert t["status"] == "backlog" and t["scheduled_for"] is None

    # In-progress tasks keep their status — only the day pin clears.
    store.set_status(conn, a, "in_progress")
    day = planner.remove_from_day(conn, a, date="2026-06-16")
    assert day["task_ids"] == []
    t = store.get_task(conn, a)
    assert t["status"] == "in_progress" and t["scheduled_for"] is None

    # A day with no plan: nothing to remove from.
    assert planner.remove_from_day(conn, a, date="2031-01-01") is None
