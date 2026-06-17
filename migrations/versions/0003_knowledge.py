"""knowledge graph (Phase 3)

Revision ID: 0003_knowledge
Revises: 0002_speakers
Create Date: 2026-06-17

Adds kg_nodes, kg_aliases, kg_edges, knowledge_extractions, and
conversations.knowledge_status. Runs the same DDL the direct-init path uses
(secondbrain.storage.schema) so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0003, STATEMENTS_0003_CREATE

revision = "0003_knowledge"
down_revision = "0002_speakers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0003:  # clean DB at revision 0002 → raw ADD COLUMN is safe
        op.execute(stmt)
    for stmt in STATEMENTS_0003_CREATE:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in (
        "DROP INDEX IF EXISTS idx_kg_edges_conversation",
        "DROP INDEX IF EXISTS idx_kg_edges_kind_valid",
        "DROP INDEX IF EXISTS idx_kg_edges_dst",
        "DROP INDEX IF EXISTS idx_kg_edges_src",
        "DROP TABLE IF EXISTS kg_edges",
        "DROP INDEX IF EXISTS idx_kg_aliases_norm",
        "DROP TABLE IF EXISTS kg_aliases",
        "DROP INDEX IF EXISTS idx_kg_nodes_speaker",
        "DROP INDEX IF EXISTS idx_kg_nodes_type_name",
        "DROP TABLE IF EXISTS kg_nodes",
        "DROP TABLE IF EXISTS knowledge_extractions",
    ):
        op.execute(stmt)
