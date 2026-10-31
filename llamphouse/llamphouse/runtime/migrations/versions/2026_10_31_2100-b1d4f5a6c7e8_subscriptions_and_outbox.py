"""subscriptions (external parties hearing about runs) and the outbox their deliveries go through

Revision ID: b1d4f5a6c7e8
Revises: a9c3e4f5b6d7
Create Date: 2026-10-31 21:00:00+00:00
"""
from typing import Sequence, Union

from alembic import op

revision: str = "b1d4f5a6c7e8"
down_revision: Union[str, None] = "a9c3e4f5b6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""create table if not exists llamphouse_subscriptions (
        id text primary key,
        data json not null,
        created_at timestamptz not null default now()
    )""")
    op.execute("""create table if not exists llamphouse_outbox (
        id text primary key,
        seq bigserial,
        subscription_id text not null,
        payload json not null,
        attempts integer not null default 0,
        next_at timestamptz not null,
        delivered_at timestamptz,
        dead boolean not null default false,
        last_error text,
        created_at timestamptz not null default now()
    )""")
    op.execute("create index if not exists llamphouse_outbox_due on llamphouse_outbox (next_at) "
               "where delivered_at is null and not dead")
    op.execute("create index if not exists llamphouse_outbox_subscription on llamphouse_outbox "
               "(subscription_id, created_at desc)")


def downgrade() -> None:
    op.execute("drop table if exists llamphouse_subscriptions, llamphouse_outbox")
