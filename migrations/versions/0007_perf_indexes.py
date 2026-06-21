"""performance indexes (Phase 9)

Revision ID: 0007_perf_indexes
Revises: 0006_speaker_quality
Create Date: 2026-06-21

Adds composite indexes for hot interactive read paths (project/person dossiers,
timeline, opt-out filtering at scale). Additive + IF NOT EXISTS; runs the same DDL
the direct-init path uses so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import STATEMENTS_0007_CREATE

revision = "0007_perf_indexes"
down_revision = "0006_speaker_quality"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in STATEMENTS_0007_CREATE:
        op.execute(stmt)


def downgrade() -> None:
    for name in (
        "idx_goal_links_kind_ref",
        "idx_kg_edges_src_kind_valid",
        "idx_kg_edges_dst_kind_valid",
        "idx_segments_speaker_conf",
    ):
        op.execute(f"DROP INDEX IF EXISTS {name}")
