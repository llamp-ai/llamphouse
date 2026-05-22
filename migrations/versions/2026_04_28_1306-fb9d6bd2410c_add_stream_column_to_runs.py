"""add stream and provider_config columns to runs

Revision ID: fb9d6bd2410c
Revises: 9e7371e978bd
Create Date: 2026-04-28 13:06:06.899582+07:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'fb9d6bd2410c'
down_revision: Union[str, None] = '9e7371e978bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('runs', sa.Column('stream', sa.Boolean(), server_default='false', nullable=False))
    op.add_column(
        'runs',
        sa.Column('provider_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('runs', 'provider_config')
    op.drop_column('runs', 'stream')
