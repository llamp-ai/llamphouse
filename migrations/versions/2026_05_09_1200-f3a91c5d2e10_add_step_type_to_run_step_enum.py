"""add step type to run_step_type_enum

Revision ID: f3a91c5d2e10
Revises: c4f1d2e3a5b8
Create Date: 2026-05-09 12:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'f3a91c5d2e10'
down_revision: Union[str, None] = 'c4f1d2e3a5b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Postgres requires ALTER TYPE ... ADD VALUE outside a transaction block
    # for new enum values. autocommit_block() handles that.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE run_step_type_enum ADD VALUE IF NOT EXISTS 'step'")
    # SQLite stores enums as plain strings — no schema change required.


def downgrade() -> None:
    # Postgres does not support removing enum values without recreating the
    # type. Downgrade is intentionally a no-op; rows of type 'step' would
    # need to be migrated/removed manually before dropping the value.
    pass
