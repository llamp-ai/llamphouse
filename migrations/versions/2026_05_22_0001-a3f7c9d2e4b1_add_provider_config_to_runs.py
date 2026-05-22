"""add provider_config to runs

Revision ID: a3f7c9d2e4b1
Revises: fb9d6bd2410c
Create Date: 2026-05-22 00:01:00.000000+07:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a3f7c9d2e4b1'
down_revision: Union[str, None] = 'fb9d6bd2410c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'runs',
        sa.Column('provider_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('runs', 'provider_config')
