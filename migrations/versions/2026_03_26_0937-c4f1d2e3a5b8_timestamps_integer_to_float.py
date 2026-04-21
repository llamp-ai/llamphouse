"""timestamps integer to float for millisecond precision

Revision ID: c4f1d2e3a5b8
Revises: 9e7371e978bd
Create Date: 2026-03-26 09:37:00.000000+00:00

Changes lifecycle timestamp columns from Integer to Double Precision (Float)
in the runs, run_steps, and messages tables so that sub-second (millisecond)
precision can be stored.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c4f1d2e3a5b8'
down_revision: Union[str, None] = '9e7371e978bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── runs ──────────────────────────────────────────────────────────────────
    op.alter_column('runs', 'expires_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('runs', 'started_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('runs', 'cancelled_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('runs', 'failed_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('runs', 'completed_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)

    # ── run_steps ─────────────────────────────────────────────────────────────
    op.alter_column('run_steps', 'expired_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('run_steps', 'cancelled_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('run_steps', 'failed_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('run_steps', 'completed_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)

    # ── messages ──────────────────────────────────────────────────────────────
    op.alter_column('messages', 'completed_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)
    op.alter_column('messages', 'incomplete_at',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=True)


def downgrade() -> None:
    # ── messages ──────────────────────────────────────────────────────────────
    op.alter_column('messages', 'incomplete_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('messages', 'completed_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)

    # ── run_steps ─────────────────────────────────────────────────────────────
    op.alter_column('run_steps', 'completed_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('run_steps', 'failed_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('run_steps', 'cancelled_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('run_steps', 'expired_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)

    # ── runs ──────────────────────────────────────────────────────────────────
    op.alter_column('runs', 'completed_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('runs', 'failed_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('runs', 'cancelled_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('runs', 'started_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
    op.alter_column('runs', 'expires_at',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=True)
