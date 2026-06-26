"""add run stream/provider_config and float timestamps

Revision ID: d8f4c2a91b67
Revises: 9e7371e978bd
Create Date: 2026-04-28 13:06:06.899582+07:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d8f4c2a91b67"
down_revision: Union[str, None] = "9e7371e978bd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "runs",
        "expires_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "started_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "cancelled_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "failed_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "completed_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )

    op.alter_column(
        "run_steps",
        "expired_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "run_steps",
        "cancelled_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "run_steps",
        "failed_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "run_steps",
        "completed_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )

    op.alter_column(
        "messages",
        "completed_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        "messages",
        "incomplete_at",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )

    op.add_column(
        "runs",
        sa.Column("stream", sa.Boolean(), server_default="false", nullable=False),
    )
    op.add_column(
        "runs",
        sa.Column("provider_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "provider_config")
    op.drop_column("runs", "stream")

    op.alter_column(
        "messages",
        "incomplete_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "messages",
        "completed_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )

    op.alter_column(
        "run_steps",
        "completed_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "run_steps",
        "failed_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "run_steps",
        "cancelled_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "run_steps",
        "expired_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )

    op.alter_column(
        "runs",
        "completed_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "failed_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "cancelled_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "started_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        "runs",
        "expires_at",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
