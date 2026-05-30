"""LLAMPHouse tracing stores package.

Tracing stores bridge the gap between the OpenTelemetry SDK and the
Compass dashboard's trace viewer.  Each store has two responsibilities:

1. **Receiving spans** — via a custom ``SpanExporter`` registered with
   the active ``TracerProvider`` (in-memory and Postgres stores), or via
   an external pipeline such as the OTel Collector → ClickHouse.

2. **Querying spans** — async methods used by the Compass API routes to
   serve the ``/api/traces`` and ``/api/traces/{run_id}`` endpoints.

Selecting a store
-----------------
Set the ``TRACING_STORE`` environment variable:

+---------------+-----------------------------+---------------------------+
| Value         | Store                       | Also required             |
+===============+=============================+===========================+
| ``memory``    | :class:`InMemoryTracingStore` | *(none)*                |
+---------------+-----------------------------+---------------------------+
| ``postgres``  | :class:`PostgresTracingStore` | ``DATABASE_URL``        |
+---------------+-----------------------------+---------------------------+
| ``clickhouse``| :class:`ClickHouseTracingStore`| ``CLICKHOUSE_URL``     |
+---------------+-----------------------------+---------------------------+

When ``TRACING_STORE`` is not set, the store is auto-detected:

* ``CLICKHOUSE_URL`` present → :class:`ClickHouseTracingStore`
* Otherwise → :class:`InMemoryTracingStore`
"""

from __future__ import annotations

import logging
import os

from .base_tracing_store import BaseTracingStore
from .in_memory_tracing_store import InMemoryTracingStore
from .postgres_tracing_store import PostgresTracingStore
from .clickhouse_tracing_store import ClickHouseTracingStore

logger = logging.getLogger("llamphouse.tracing")

__all__ = [
    "BaseTracingStore",
    "InMemoryTracingStore",
    "PostgresTracingStore",
    "ClickHouseTracingStore",
    "get_tracing_store_from_env",
]


def get_tracing_store_from_env() -> BaseTracingStore:
    """Create and return the tracing store indicated by environment variables.

    See the module docstring for the full selection logic.
    """
    store_type = os.getenv("TRACING_STORE", "").lower().strip()

    if store_type == "clickhouse":
        url = os.getenv("CLICKHOUSE_URL", "")
        if not url:
            logger.warning(
                "TRACING_STORE=clickhouse but CLICKHOUSE_URL is not set — "
                "falling back to InMemoryTracingStore"
            )
            return InMemoryTracingStore()
        logger.info("Tracing store: ClickHouse (%s)", url)
        return ClickHouseTracingStore(url)

    if store_type == "postgres":
        db_url = os.getenv("DATABASE_URL", "")
        if not db_url:
            logger.warning(
                "TRACING_STORE=postgres but DATABASE_URL is not set — "
                "falling back to InMemoryTracingStore"
            )
            return InMemoryTracingStore()
        logger.info("Tracing store: Postgres")
        return PostgresTracingStore(db_url)

    if store_type == "memory":
        logger.info("Tracing store: in-memory")
        return InMemoryTracingStore()

    # ── Auto-detect ───────────────────────────────────────────────────────
    clickhouse_url = os.getenv("CLICKHOUSE_URL", "")
    if clickhouse_url:
        logger.info("Tracing store: ClickHouse (auto-detected via CLICKHOUSE_URL)")
        return ClickHouseTracingStore(clickhouse_url)

    logger.info("Tracing store: in-memory (default)")
    return InMemoryTracingStore()
