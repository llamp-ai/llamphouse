"""add reasoning_effort to runs

Revision ID: a96292c5d628
Revises: 95e461947b69
Create Date: 2025-12-01 11:57:30.791326+07:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a96292c5d628'
down_revision: Union[str, None] = '95e461947b69'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("reasoning_effort", sa.String(), nullable=False, server_default="medium"))


def downgrade() -> None:
    op.drop_column("runs", "reasoning_effort")
