"""decision tracking + edge full-text search (Phase 11)

Revision ID: 0009_decisions
Revises: 0008_reliability
Create Date: 2026-08-13

Adds an FTS5 index over kg_edges.object_text (decisions and commitments become
searchable), sync triggers mirroring the transcript_segments_fts pattern, a
one-time backfill of existing edges, and a normalized due-date column
(kg_edges.due_date_norm) for action items. Runs the same DDL the direct-init
path uses so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0009, STATEMENTS_0009_CREATE

revision = "0009_decisions"
down_revision = "0008_reliability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0009:
        op.execute(stmt)
    for stmt in STATEMENTS_0009_CREATE:
        op.execute(stmt)
    # Backfill the new FTS table from edges written before the triggers existed.
    op.execute("INSERT INTO kg_edges_fts(kg_edges_fts) VALUES('rebuild')")


def downgrade() -> None:
    # SQLite can't easily drop columns; drop the FTS table and its triggers.
    op.execute("DROP TRIGGER IF EXISTS trg_kg_edges_ai")
    op.execute("DROP TRIGGER IF EXISTS trg_kg_edges_ad")
    op.execute("DROP TRIGGER IF EXISTS trg_kg_edges_au")
    op.execute("DROP TABLE IF EXISTS kg_edges_fts")
