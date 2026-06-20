"""speaker quality / self-correction (Phase 7)

Revision ID: 0006_speaker_quality
Revises: 0005_tasks
Create Date: 2026-06-19

Adds exemplar metadata to speaker_observations and locking/provenance columns to
transcript_segments. Runs the same DDL the direct-init path uses so the two can't
drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0006, STATEMENTS_0006_CREATE

revision = "0006_speaker_quality"
down_revision = "0005_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0006:
        op.execute(stmt)
    for stmt in STATEMENTS_0006_CREATE:
        op.execute(stmt)


def downgrade() -> None:
    # SQLite can't easily drop columns; this migration only adds columns.
    pass
