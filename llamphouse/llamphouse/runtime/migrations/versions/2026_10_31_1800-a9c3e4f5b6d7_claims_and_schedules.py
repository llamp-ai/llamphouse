"""claims (idempotent requests) and schedules (cron and one-off runs)

Revision ID: a9c3e4f5b6d7
Revises: f8b2d3e4a5c6
Create Date: 2026-10-31 18:00:00+00:00
"""
from typing import Sequence, Union

from alembic import op

revision: str = "a9c3e4f5b6d7"
down_revision: Union[str, None] = "f8b2d3e4a5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""create table if not exists llamphouse_claims (
        key text primary key,
        value text,
        created_at timestamptz not null default now()
    )""")
    op.execute("""create table if not exists llamphouse_schedules (
        id text primary key,
        agent text not null,
        definition text not null,
        next_at timestamptz,
        data json not null,
        updated_at timestamptz not null default now()
    )""")
    op.execute("create index if not exists llamphouse_schedules_due on llamphouse_schedules (next_at)")


def downgrade() -> None:
    op.execute("drop table if exists llamphouse_claims, llamphouse_schedules")
