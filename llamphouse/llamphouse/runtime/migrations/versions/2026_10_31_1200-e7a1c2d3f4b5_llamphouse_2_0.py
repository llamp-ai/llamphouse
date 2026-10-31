"""llamphouse 2.0: the runtime's tables

The base of the `v2` branch - next to the 1.x schema (`v1`), not after it: a new 2.0 database gets only these
tables; a 1.x database keeps its own and gains these.

Trees (a thread and every thread its runs create) are the unit of a transaction: every row carries its tree
and the tree's version when written. Written to adopt a database the 2.0 prototype's built-in migrator set up
(`if not exists` throughout; its `llamphouse_schema` table is dropped).

Revision ID: e7a1c2d3f4b5
Revises:
Create Date: 2026-10-31 12:00:00+00:00
"""
from typing import Sequence, Union

from alembic import op

revision: str = "e7a1c2d3f4b5"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = ("v2",)
depends_on: Union[str, Sequence[str], None] = None

UPGRADE = [
    # trees: one row per tree - its version, and how many events it wrote (their order)
    """create table if not exists llamphouse_trees (
        id text primary key,
        version bigint not null,
        written bigint not null,
        created_at timestamptz not null default now()
    )""",
    """create table if not exists llamphouse_threads (
        id text primary key,
        tree_id text not null,
        version bigint not null,
        agent text not null,
        created_by text,
        forked_from text,
        closed boolean not null default false,
        created_at timestamptz,
        metadata jsonb not null default '{}',
        data json not null
    )""",
    "alter table llamphouse_threads add column if not exists metadata jsonb not null default '{}'",
    "create index if not exists llamphouse_threads_tree on llamphouse_threads (tree_id, version)",
    "create index if not exists llamphouse_threads_created on llamphouse_threads (created_at desc)",
    "create index if not exists llamphouse_threads_metadata on llamphouse_threads using gin (metadata jsonb_path_ops)",
    """create table if not exists llamphouse_events (
        thread_id text not null,
        position integer not null,
        tree_id text not null,
        version bigint not null,
        written bigint not null,
        key text,
        data json not null,
        primary key (thread_id, position)
    )""",
    "create index if not exists llamphouse_events_tree on llamphouse_events (tree_id, version)",
    """create table if not exists llamphouse_runs (
        id text primary key,
        tree_id text not null,
        version bigint not null,
        thread_id text not null,
        agent text not null,
        status text not null,
        wake_at timestamptz,
        lease_until timestamptz,
        ring_at timestamptz,
        data json not null
    )""",
    "create index if not exists llamphouse_runs_tree on llamphouse_runs (tree_id, version)",
    "create index if not exists llamphouse_runs_thread on llamphouse_runs (thread_id)",
    """create unique index if not exists llamphouse_one_active_run on llamphouse_runs (thread_id)
        where status not in ('completed', 'failed', 'cancelled', 'rejected')""",
    "create index if not exists llamphouse_runs_wake on llamphouse_runs (wake_at) where status = 'waiting'",
    "create index if not exists llamphouse_runs_lease on llamphouse_runs (lease_until) where status = 'running'",
    "create index if not exists llamphouse_runs_queued on llamphouse_runs (ring_at) where status = 'queued'",
    """create table if not exists llamphouse_sessions (
        id text primary key,
        tree_id text not null,
        version bigint not null,
        thread_id text not null,
        revising boolean not null default false,
        revision_due timestamptz,
        data json not null
    )""",
    "create index if not exists llamphouse_sessions_tree on llamphouse_sessions (tree_id, version)",
    "create index if not exists llamphouse_sessions_due on llamphouse_sessions (revision_due) where revising",
    """create table if not exists llamphouse_run_events (
        run_id text not null,
        seq integer not null,
        tree_id text not null,
        at timestamptz not null default now(),
        data json not null,
        primary key (run_id, seq)
    )""",
    # files: their bytes once per sha256 (PostgresFileStore)
    """create table if not exists llamphouse_blobs (
        sha256 text primary key,
        data bytea not null
    )""",
    """create table if not exists llamphouse_files (
        id text primary key,
        thread_id text,
        sha256 text not null references llamphouse_blobs(sha256),
        size bigint not null,
        mime_type text not null,
        name text,
        created_by text not null,
        created_at timestamptz not null,
        expires_at timestamptz
    )""",
    "create index if not exists llamphouse_files_thread on llamphouse_files (thread_id)",
    # spans (PostgresTracingStore)
    """create table if not exists llamphouse_traces (
        timestamp timestamptz not null,
        trace_id char(32) not null,
        span_id char(16) not null,
        parent_span_id varchar(16) not null default '',
        span_name text not null,
        span_kind smallint not null default 0,
        duration_ns bigint not null default 0,
        status_code varchar(30) not null default '',
        status_message text not null default '',
        span_attributes jsonb not null default '{}',
        events jsonb not null default '[]',
        primary key (span_id)
    )""",
    "create index if not exists idx_llh_traces_trace_id on llamphouse_traces (trace_id)",
    "create index if not exists idx_llh_traces_ts on llamphouse_traces (timestamp desc)",
    "drop table if exists llamphouse_schema",  # the prototype's own migrator: Alembic keeps the history now
]

DOWNGRADE = [
    "drop table if exists llamphouse_run_events, llamphouse_sessions, llamphouse_runs, llamphouse_events, "
    "llamphouse_threads, llamphouse_trees, llamphouse_files, llamphouse_blobs, llamphouse_traces cascade",
]


def upgrade() -> None:
    for statement in UPGRADE:  # one at a time: a driver may not take several in one go
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
