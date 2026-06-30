"""convert run and message timestamps to float

Revision ID: d8f4c2a91b67
Revises: fb9d6bd2410c
Create Date: 2026-04-28 13:10:00.000000+07:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd8f4c2a91b67'
down_revision: Union[str, None] = 'fb9d6bd2410c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'runs',
        'expires_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'started_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'cancelled_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'failed_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'completed_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )

    op.alter_column(
        'run_steps',
        'expired_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'run_steps',
        'cancelled_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'run_steps',
        'failed_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'run_steps',
        'completed_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )

    op.alter_column(
        'messages',
        'completed_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'messages',
        'incomplete_at',
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'messages',
        'incomplete_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'messages',
        'completed_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )

    op.alter_column(
        'run_steps',
        'completed_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'run_steps',
        'failed_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'run_steps',
        'cancelled_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'run_steps',
        'expired_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )

    op.alter_column(
        'runs',
        'completed_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'failed_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'cancelled_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'started_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.alter_column(
        'runs',
        'expires_at',
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
