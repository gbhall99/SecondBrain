"""meeting-aware planner + weekly digest stats (Phase 12)

Revision ID: 0010_planner
Revises: 0009_decisions
Create Date: 2026-08-14

Adds visible-rollover columns to tasks (rollover_count, last_planned_for) so
the planner can show how often a task slipped, and a JSON payload column on
digests carrying the weekly review's deterministic stats. Runs the same DDL
the direct-init path uses so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0010, STATEMENTS_0010_CREATE

revision = "0010_planner"
down_revision = "0009_decisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0010:
        op.execute(stmt)
    for stmt in STATEMENTS_0010_CREATE:
        op.execute(stmt)


def downgrade() -> None:
    # SQLite can't easily drop columns; the additive columns are harmless.
    pass
