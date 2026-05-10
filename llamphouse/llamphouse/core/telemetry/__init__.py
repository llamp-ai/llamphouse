"""Lightweight, non-blocking, opt-out telemetry for LLAMPHouse.

Design goals:
    * Never block the host application or interfere with normal execution.
    * Never raise — any failure is silently swallowed.
    * Anonymous: a random install_id is generated locally; no user data,
      prompts, agent names, or message content are ever collected.
    * Opt-out via the ``LLAMPHOUSE_TELEMETRY=0`` environment variable.

Events are queued in memory and flushed in a background daemon thread,
either when the queue reaches ``_BATCH_SIZE`` or every ``_FLUSH_INTERVAL``
seconds. Network errors are ignored.
"""

from .client import record, shutdown, is_enabled, bump, observe_run_ms

__all__ = ["record", "shutdown", "is_enabled", "bump", "observe_run_ms"]
