"""llamphouse 2.0: sessions listed by agent and by when they changed (Compass's Sessions)

Revision ID: d3f6a7b8c9e0
Revises: c2e5a6b7d8f9
Create Date: 2026-11-01 10:00:00+00:00
"""
from typing import Sequence, Union

from alembic import op

revision: str = "d3f6a7b8c9e0"
down_revision: Union[str, None] = "c2e5a6b7d8f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UPGRADE = [
    "alter table llamphouse_sessions add column if not exists agent text",
    "alter table llamphouse_sessions add column if not exists updated_at timestamptz",
    "create index if not exists llamphouse_sessions_updated on llamphouse_sessions (updated_at desc)",
]

DOWNGRADE = [
    "drop index if exists llamphouse_sessions_updated",
    "alter table llamphouse_sessions drop column if exists updated_at",
    "alter table llamphouse_sessions drop column if exists agent",
]


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
