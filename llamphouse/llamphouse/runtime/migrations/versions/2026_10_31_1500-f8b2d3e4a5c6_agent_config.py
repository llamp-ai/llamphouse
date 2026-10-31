"""agent config: values changed at runtime, shared by every process (llamphouse.runtime.config_store)

Revision ID: f8b2d3e4a5c6
Revises: e7a1c2d3f4b5
Create Date: 2026-10-31 15:00:00+00:00
"""
from typing import Sequence, Union

from alembic import op

revision: str = "f8b2d3e4a5c6"
down_revision: Union[str, None] = "e7a1c2d3f4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""create table if not exists llamphouse_agent_config (
        agent text primary key,
        config_values jsonb not null default '{}',
        version integer not null default 0,
        updated_by text,
        updated_at timestamptz not null default now()
    )""")


def downgrade() -> None:
    op.execute("drop table if exists llamphouse_agent_config")
