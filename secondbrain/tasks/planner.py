"""Daily planner: propose a capacity-fitted Today list (you approve)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.storage.models import utcnow_iso
from secondbrain.tasks import prioritize, store

_DEFAULT_TASK_MINUTES = 30
# Suggested capacity never drops below this, however meeting-packed the day is.
MIN_SUGGESTED_CAPACITY = 30


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _local_day_utc_bounds(day: str) -> tuple[str, str]:
    """UTC ISO bounds [start, end) covering the *local* calendar day ``day``."""
    start_local = datetime.strptime(day, "%Y-%m-%d")  # naive == system local time
    fmt = "%Y-%m-%dT%H:%M:%S"
    return (
        start_local.astimezone(UTC).strftime(fmt),
        (start_local + timedelta(days=1)).astimezone(UTC).strftime(fmt),
    )


def meeting_minutes(conn: sqlite3.Connection, day: str | None = None) -> int:
    """Minutes of recorded conversation (meetings) on the local day so far.

    Open conversations count up to now, so a mid-meeting check-in still sees
    the time already spent.
    """
    day = day or datetime.now().astimezone().strftime("%Y-%m-%d")
    lo, hi = _local_day_utc_bounds(day)
    now_iso = utcnow_iso()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(
            (julianday(MIN(COALESCE(ended_at, ?), ?)) - julianday(started_at)) * 1440.0
        ), 0) AS m
        FROM conversations
        WHERE started_at >= ? AND started_at < ?
          AND started_at IS NOT NULL
        """,
        (now_iso, hi, lo, hi),
    ).fetchone()
    return max(0, int(round(row["m"] or 0.0)))


def suggested_capacity(settings: Settings, meeting_min: int) -> int:
    """Working-day length minus recorded meeting minutes, floored at 30."""
    return max(MIN_SUGGESTED_CAPACITY, settings.tasks.workday_minutes - max(0, meeting_min))


def propose_day(
    conn: sqlite3.Connection,
    date: str | None = None,
    capacity_minutes: int | None = None,
    settings: Settings | None = None,
) -> dict:
    """Build (and store as 'proposed') a capacity-fitted ranked Today plan.

    The returned plan additionally carries ``rolled_over`` (tasks released
    from earlier unfinished plans this pass) and ``big_rock_task_id`` when the
    top-ranked task is bigger than the whole capacity — it is included anyway,
    flagged, instead of silently packing small tasks around it.
    """
    settings = settings or get_settings()
    date = date or _today()
    capacity = capacity_minutes or settings.tasks.daily_capacity_minutes
    today = datetime.strptime(date, "%Y-%m-%d").date()

    # Earlier days' unfinished plans must not pin tasks to the past: release
    # anything still 'scheduled' for a day before this one back to the backlog
    # so it competes for today like everything else. Capture what slipped
    # first so the proposal can say so.
    rolled = [
        {"id": r["id"], "title": r["title"]}
        for r in conn.execute(
            "SELECT id, title FROM tasks WHERE status='scheduled' "
            "AND scheduled_for IS NOT NULL AND scheduled_for < ? ORDER BY id",
            (date,),
        ).fetchall()
    ]
    store.release_stale_scheduled(conn, date)

    ranked = sorted(
        store.ready_tasks(conn),
        key=lambda t: prioritize.score(conn, t, settings, today),
        reverse=True,
    )

    def est(t: dict) -> int:
        return t.get("estimate_minutes") or _DEFAULT_TASK_MINUTES

    chosen: list[int] = []
    used = 0
    big_rock: int | None = None
    if ranked and est(ranked[0]) > capacity:
        # Big-rock protection: the single most important task doesn't fit the
        # day at all. Surface it at the top with a flag ("won't fit — schedule
        # a block?") instead of hiding it behind a pile of small tasks.
        big_rock = ranked[0]["id"]
        chosen.append(big_rock)
        used += est(ranked[0])
        ranked = ranked[1:]
    for t in ranked:
        e = est(t)
        if used + e > capacity and chosen:
            continue
        chosen.append(t["id"])
        used += e
        if used >= capacity:
            break

    conn.execute(
        """
        INSERT INTO day_plans (date, capacity_minutes, status, task_ids)
        VALUES (?, ?, 'proposed', ?)
        ON CONFLICT(date) DO UPDATE SET
            capacity_minutes=excluded.capacity_minutes,
            status='proposed', task_ids=excluded.task_ids
        """,
        (date, capacity, json.dumps(chosen)),
    )
    # Re-proposing can shrink the list: tasks accepted into an earlier version
    # of this day's plan but not chosen now would otherwise keep a stale
    # 'scheduled' pill in the backlog forever. Send those back to the backlog
    # (in-progress and done/dropped tasks are left untouched).
    sql = (
        "UPDATE tasks SET status='backlog', scheduled_for=NULL, updated_at=? "
        "WHERE scheduled_for=? AND status='scheduled'"
    )
    if chosen:
        sql += f" AND id NOT IN ({','.join('?' * len(chosen))})"
    conn.execute(sql, (utcnow_iso(), date, *chosen))
    plan = get_day(conn, date)
    plan["rolled_over"] = rolled
    plan["big_rock_task_id"] = big_rock
    return plan


def accept_day(conn: sqlite3.Connection, date: str | None = None) -> dict | None:
    date = date or _today()
    plan = get_day(conn, date)
    if plan is None:
        return None
    for tid in plan["task_ids"]:
        # Don't clobber an in-progress task back to 'scheduled'; just set the day.
        conn.execute(
            "UPDATE tasks SET scheduled_for=?, "
            "status=CASE WHEN status='in_progress' THEN status ELSE 'scheduled' END, "
            "updated_at=? WHERE id=? AND status NOT IN ('done','dropped')",
            (date, utcnow_iso(), tid),
        )
    conn.execute("UPDATE day_plans SET status='accepted' WHERE date=?", (date,))
    return get_day(conn, date)


def add_to_day(
    conn: sqlite3.Connection,
    task_id: int,
    date: str | None = None,
    settings: Settings | None = None,
) -> dict | None:
    """Pin one task into the day's plan ("Do today") without re-proposing.

    Creates a proposed plan if the day has none yet. On an accepted plan the
    task is scheduled immediately (mirroring accept_day for that one task).
    Returns the updated plan, or None when the task doesn't exist or is
    already finished.
    """
    settings = settings or get_settings()
    date = date or _today()
    task = store.get_task(conn, task_id)
    if task is None or task["status"] in store.DONE_STATUSES:
        return None
    row = conn.execute("SELECT status, task_ids FROM day_plans WHERE date=?", (date,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO day_plans (date, capacity_minutes, status, task_ids) "
            "VALUES (?, ?, 'proposed', ?)",
            (date, settings.tasks.daily_capacity_minutes, json.dumps([task_id])),
        )
        return get_day(conn, date)
    ids = json.loads(row["task_ids"] or "[]")
    if task_id not in ids:
        ids.append(task_id)
        conn.execute("UPDATE day_plans SET task_ids=? WHERE date=?", (json.dumps(ids), date))
    if row["status"] == "accepted":
        conn.execute(
            "UPDATE tasks SET scheduled_for=?, "
            "status=CASE WHEN status='in_progress' THEN status ELSE 'scheduled' END, "
            "updated_at=? WHERE id=? AND status NOT IN ('done','dropped')",
            (date, utcnow_iso(), task_id),
        )
    return get_day(conn, date)


def remove_from_day(
    conn: sqlite3.Connection, task_id: int, date: str | None = None
) -> dict | None:
    """Take one task out of the day's plan ("not today") without re-proposing.

    The plan keeps its status (proposed/accepted) and its other tasks. The
    removed task goes back to the backlog if the plan had it 'scheduled';
    in-progress / done tasks keep their status — only the day pin clears.
    Returns the updated plan, or None when that day has no plan.
    """
    date = date or _today()
    plan = get_day(conn, date)
    if plan is None:
        return None
    if task_id in plan["task_ids"]:
        ids = [tid for tid in plan["task_ids"] if tid != task_id]
        conn.execute(
            "UPDATE day_plans SET task_ids=? WHERE date=?", (json.dumps(ids), date)
        )
    conn.execute(
        "UPDATE tasks SET status=CASE WHEN status='scheduled' THEN 'backlog' ELSE status END, "
        "scheduled_for=NULL, updated_at=? WHERE id=? AND scheduled_for=?",
        (utcnow_iso(), task_id, date),
    )
    return get_day(conn, date)


def get_day(conn: sqlite3.Connection, date: str | None = None) -> dict | None:
    date = date or _today()
    row = conn.execute("SELECT * FROM day_plans WHERE date=?", (date,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    stored = json.loads(d["task_ids"] or "[]")
    # Done/dropped (or deleted) tasks leave the plan: set_status prunes the
    # stored list, and hydration filters defensively for rows written before
    # that behaviour existed.
    tasks = []
    ids = []
    for tid in stored:
        t = store.get_task(conn, tid)
        if t is None or t["status"] in store.DONE_STATUSES:
            continue
        ids.append(tid)
        tasks.append(t)
    d["task_ids"] = ids
    d["tasks"] = tasks
    return d
