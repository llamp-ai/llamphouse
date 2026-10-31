"""runs' times (for statistics and dashboards) and documents (Compass's dashboards and charts)

Revision ID: c2e5a6b7d8f9
Revises: b1d4f5a6c7e8
Create Date: 2026-11-01 09:00:00+00:00
"""
from typing import Sequence, Union

from alembic import op

revision: str = "c2e5a6b7d8f9"
down_revision: Union[str, None] = "b1d4f5a6c7e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("alter table llamphouse_runs add column if not exists created_at timestamptz")
    op.execute("alter table llamphouse_runs add column if not exists ended_at timestamptz")
    op.execute("create index if not exists llamphouse_runs_created on llamphouse_runs (created_at desc)")
    op.execute("""create table if not exists llamphouse_documents (
        kind text not null,
        id text not null,
        data json not null,
        updated_at timestamptz not null default now(),
        primary key (kind, id)
    )""")


def downgrade() -> None:
    op.execute("drop table if exists llamphouse_documents")
    op.execute("drop index if exists llamphouse_runs_created")
    op.execute("alter table llamphouse_runs drop column if exists ended_at")
    op.execute("alter table llamphouse_runs drop column if exists created_at")
