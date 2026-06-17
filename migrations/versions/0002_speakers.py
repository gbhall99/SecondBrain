"""speakers + diarization (Phase 2)

Revision ID: 0002_speakers
Revises: 0001_initial
Create Date: 2026-06-17

Adds conversations, speaker_observations, speaker profile columns, and
transcript_segments.speaker_confidence. Runs the same DDL the direct-init path
uses (secondbrain.storage.schema) so the two can't drift.
"""

from alembic import op

from secondbrain.storage.schema import ALTERS_0002, STATEMENTS_0002_CREATE

revision = "0002_speakers"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in ALTERS_0002:  # clean DB at revision 0001 → raw ADD COLUMN is safe
        op.execute(stmt)
    for stmt in STATEMENTS_0002_CREATE:  # some indices reference the new columns
        op.execute(stmt)


def downgrade() -> None:
    # SQLite can't easily drop columns; drop the new tables/indices only.
    for stmt in (
        "DROP INDEX IF EXISTS idx_audio_conversation",
        "DROP INDEX IF EXISTS idx_segments_speaker",
        "DROP INDEX IF EXISTS idx_speaker_obs_speaker",
        "DROP TABLE IF EXISTS speaker_observations",
        "DROP INDEX IF EXISTS idx_conversations_status",
        "DROP TABLE IF EXISTS conversations",
    ):
        op.execute(stmt)
