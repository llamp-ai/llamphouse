"""durable runs: run leases, wake time, signals

A running run is held by one worker under a lease its heartbeat renews
(worker_id, lease_expires_at; lease_epoch counts the claims). A waiting run
holds no worker and is due again at wake_at, or when its signal arrives -
signals are kept in their own table until the run consumes them.
See docs/DURABLE_RUNS_PLAN.md.

Revision ID: c3d8e1f2a4b5
Revises: b7e2c9a4d1f3
Create Date: 2026-10-20 12:00:00+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c3d8e1f2a4b5"
down_revision: Union[str, Sequence[str], None] = "b7e2c9a4d1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("worker_id", sa.String(), nullable=True))
    op.add_column("runs", sa.Column("lease_expires_at", sa.Float(), nullable=True))
    op.add_column("runs", sa.Column("lease_epoch", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("wake_at", sa.Float(), nullable=True))
    # What the recovery loop scans: running runs by lease, waiting runs by wake time.
    op.create_index(
        "idx_runs_recover_lease", "runs", ["lease_expires_at"],
        postgresql_where=sa.text("status = 'in_progress'"),
    )
    op.create_index(
        "idx_runs_recover_wake", "runs", ["wake_at"],
        postgresql_where=sa.text("status = 'requires_action' AND wake_at IS NOT NULL"),
    )

    op.create_table(
        "signals",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("correlation_id", sa.String(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_signals_pending", "signals", ["run_id", "name", "correlation_id"],
        postgresql_where=sa.text("consumed_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_signals_pending", table_name="signals")
    op.drop_table("signals")
    op.drop_index("idx_runs_recover_wake", table_name="runs")
    op.drop_index("idx_runs_recover_lease", table_name="runs")
    op.drop_column("runs", "wake_at")
    op.drop_column("runs", "lease_epoch")
    op.drop_column("runs", "lease_expires_at")
    op.drop_column("runs", "worker_id")
