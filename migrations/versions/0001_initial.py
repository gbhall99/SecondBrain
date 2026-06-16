"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-16

Runs the canonical DDL from secondbrain.storage.schema so the migration and the
direct-init path (used in tests) can never drift.
"""

from alembic import op

from secondbrain.storage.schema import STATEMENTS

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for table in (
        "app_state",
        "transcript_segments_fts",
        "speakers",
        "jobs",
        "transcript_segments",
        "transcripts",
        "audio_files",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table}")
