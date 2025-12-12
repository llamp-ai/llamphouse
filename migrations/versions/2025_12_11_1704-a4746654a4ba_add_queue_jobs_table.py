"""add queue_jobs table

Revision ID: a4746654a4ba
Revises: a96292c5d628
Create Date: 2025-12-11 17:04:30.794931+07:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a4746654a4ba'
down_revision: Union[str, None] = 'a96292c5d628'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "queue",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("assistant_id", sa.String, index=True, nullable=True),
        sa.Column("thread_id", sa.String, index=True, nullable=True),
        sa.Column("run_id", sa.String, index=True, nullable=True),
        sa.Column("payload", postgresql.JSONB, nullable=True),
        sa.Column("status", sa.String, nullable=False, server_default="queued"),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_error", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), onupdate=sa.text("now()")),
    )
    op.create_index("idx_queue_ready", "queue", ["status", "ready_at"])
    op.create_index("idx_queue_lease", "queue", ["lease_until"])


def downgrade() -> None:
    op.drop_index("idx_queue_ready", table_name="queue")
    op.drop_index("idx_queue_lease", table_name="queue")
    op.drop_table("queue")
