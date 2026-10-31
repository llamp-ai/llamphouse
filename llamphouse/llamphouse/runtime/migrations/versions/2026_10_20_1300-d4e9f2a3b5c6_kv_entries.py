"""key-value records (kv_entries)

Small JSON records grouped by namespace, for what the framework keeps
besides threads, messages and runs - LangGraph checkpoints first
(wrappers/langgraph_checkpoint.py). Records tied to a thread go with it.

Revision ID: d4e9f2a3b5c6
Revises: c3d8e1f2a4b5
Create Date: 2026-10-20 13:00:00+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d4e9f2a3b5c6"
down_revision: Union[str, Sequence[str], None] = "c3d8e1f2a4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "kv_entries",
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("thread_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("namespace", "key"),
    )
    op.create_index("ix_kv_entries_thread_id", "kv_entries", ["thread_id"])


def downgrade() -> None:
    op.drop_index("ix_kv_entries_thread_id", table_name="kv_entries")
    op.drop_table("kv_entries")
