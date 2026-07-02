"""proactivity + goals (Phase 4)

Revision ID: 0004_proactive
Revises: 0003_knowledge
Create Date: 2026-06-17

Adds goals, goal_links, suggestions, suggestion_feedback, digests. Runs the same
DDL the direct-init path uses (secondbrain.storage.schema) so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0004, STATEMENTS_0004_CREATE

revision = "0004_proactive"
down_revision = "0003_knowledge"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0004:
        op.execute(stmt)
    for stmt in STATEMENTS_0004_CREATE:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in (
        "DROP TABLE IF EXISTS digests",
        "DROP INDEX IF EXISTS idx_feedback_dedupe",
        "DROP TABLE IF EXISTS suggestion_feedback",
        "DROP INDEX IF EXISTS idx_suggestions_dedupe",
        "DROP INDEX IF EXISTS idx_suggestions_date_status",
        "DROP TABLE IF EXISTS suggestions",
        "DROP INDEX IF EXISTS idx_goal_links_goal",
        "DROP TABLE IF EXISTS goal_links",
        "DROP INDEX IF EXISTS idx_goals_status",
        "DROP TABLE IF EXISTS goals",
    ):
        op.execute(stmt)
