"""pipeline reliability (Phase 10)

Revision ID: 0008_reliability
Revises: 0007_perf_indexes
Create Date: 2026-08-13

Adds capture-observability columns to audio_files (speech_seconds, rms_level,
overflow_count) and an index on jobs(state, finished_at) for the done-job pruning
sweep. Runs the same DDL the direct-init path uses so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0008, STATEMENTS_0008_CREATE

revision = "0008_reliability"
down_revision = "0007_perf_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0008:
        op.execute(stmt)
    for stmt in STATEMENTS_0008_CREATE:
        op.execute(stmt)


def downgrade() -> None:
    # SQLite can't easily drop columns; only drop the added index.
    op.execute("DROP INDEX IF EXISTS idx_jobs_state_finished")
