"""add heartbeat_at to runs; joins the two heads

A running run's worker records here, every few seconds, that it is alive
(workers/run_lifecycle.py): a run whose heartbeat stopped had its worker
die. Revises both heads, so the history is one line again.

Revision ID: b7e2c9a4d1f3
Revises: a1b2c3d4e5f6, f3a91c5d2e10
Create Date: 2026-10-14 12:00:00+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7e2c9a4d1f3"
down_revision: Union[str, Sequence[str], None] = ("a1b2c3d4e5f6", "f3a91c5d2e10")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("heartbeat_at", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "heartbeat_at")
