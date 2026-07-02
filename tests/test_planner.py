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
