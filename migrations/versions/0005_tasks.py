"""tasks + daily planning (Phase 6)

Revision ID: 0005_tasks
Revises: 0004_proactive
Create Date: 2026-06-19

Adds tasks, task_deps, task_research, day_plans. Runs the same DDL the
direct-init path uses (secondbrain.storage.schema) so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0005, STATEMENTS_0005_CREATE

revision = "0005_tasks"
down_revision = "0004_proactive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0005:
        op.execute(stmt)
    for stmt in STATEMENTS_0005_CREATE:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in (
        "DROP TABLE IF EXISTS day_plans",
        "DROP INDEX IF EXISTS idx_task_research_task",
        "DROP TABLE IF EXISTS task_research",
        "DROP TABLE IF EXISTS task_deps",
        "DROP INDEX IF EXISTS idx_tasks_scheduled",
        "DROP INDEX IF EXISTS idx_tasks_goal",
        "DROP INDEX IF EXISTS idx_tasks_status",
        "DROP TABLE IF EXISTS tasks",
    ):
        op.execute(stmt)
