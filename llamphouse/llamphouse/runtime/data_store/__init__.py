"""The data store: where the engine's state lives - the truth, shared by every process.

    base.py       Store, StoreTx and the stored forms (Row, EventRow, Rows, Due)
    memory.py     MemoryStore - in this process
    postgres.py   PostgresStore - asyncpg, its schema under Alembic

The runtime file names one: `data_store: {postgres: {dsn: ...}}`, or your own `{my_pkg:MyStore: {...}}`.
"""

from .base import TERMINAL, Due, EventRow, Row, Rows, Store, StoreTx
from .memory import MemoryStore


def __getattr__(name: str):  # the Postgres store loads when asked for: asyncpg is an extra
    if name == "PostgresStore":
        from .postgres import PostgresStore

        return PostgresStore
    raise AttributeError(name)


__all__ = ["Store", "StoreTx", "Row", "EventRow", "Rows", "Due", "TERMINAL", "MemoryStore", "PostgresStore"]
