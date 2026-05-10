"""Background, fire-and-forget telemetry client.

The client is a singleton. The first call to :func:`record` lazily starts
a daemon worker thread that drains a bounded in-memory queue and POSTs
batched JSON to the configured endpoint. All errors are silently dropped
so the host application is never affected.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request as _urlreq
from urllib.error import URLError

try:
    from llamphouse import __version__ as _LLAMPHOUSE_VERSION
except Exception:  # pragma: no cover - defensive
    _LLAMPHOUSE_VERSION = "unknown"


_DEFAULT_ENDPOINT = "https://api.llamp.ai/telemetry"
_BATCH_SIZE = 20
_FLUSH_INTERVAL = 30.0          # seconds — flush queued events
_USAGE_FLUSH_INTERVAL = 300.0   # seconds — emit aggregated usage counters every 5 min
_QUEUE_MAX = 500                 # drop oldest if exceeded — never block producers
_HTTP_TIMEOUT = 3.0              # seconds

# Run-duration histogram buckets (seconds, upper bound). The last bucket is +Inf.
_RUN_DUR_BUCKETS_S = (1.0, 10.0, 60.0, 300.0)
_RUN_DUR_BUCKET_LABELS = ("lt_1s", "lt_10s", "lt_60s", "lt_300s", "gte_300s")


def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def _falsy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("0", "false", "no", "off")


_USAGE_TIER_VALUES = ("usage", "full", "detailed", "all", "2")
_LIFECYCLE_TIER_VALUES = ("lifecycle", "minimal", "basic")


def _tier() -> str:
    """Return the active telemetry tier: ``"off"``, ``"lifecycle"`` or ``"usage"``.

    * ``LLAMPHOUSE_TELEMETRY=0`` (or any falsy value) / ``NO_TRACKING=1`` → off
    * ``LLAMPHOUSE_TELEMETRY=lifecycle`` (or ``minimal``/``basic``)
      → lifecycle events only
    * Anything else (including unset, ``1``/``true``, or
      ``usage``/``full``/``detailed``/``all``) → usage (default)
    """
    explicit = os.environ.get("LLAMPHOUSE_TELEMETRY")
    if explicit is not None:
        normalised = explicit.strip().lower()
        if normalised in _LIFECYCLE_TIER_VALUES:
            return "lifecycle"
        if normalised in _USAGE_TIER_VALUES:
            return "usage"
        if _falsy(explicit):
            return "off"
        if _truthy(explicit):
            return "usage"
    if _truthy(os.environ.get("NO_TRACKING")):
        return "off"
    return "usage"


def is_enabled() -> bool:
    """True when telemetry is on at any tier (``lifecycle`` or ``usage``)."""
    return _tier() != "off"


def _usage_enabled() -> bool:
    return _tier() == "usage"


def _validate_uuid(val: Optional[str]) -> str:
    """Return a normalised UUID string, or "" if ``val`` is not a valid UUID.

    Accepts any form ``uuid.UUID`` accepts (with/without hyphens, braces,
    or ``urn:uuid:`` prefix) and always emits the canonical hyphenated
    lower-case representation.
    """
    if not val:
        return ""
    try:
        return str(uuid.UUID(val.strip()))
    except (ValueError, AttributeError, TypeError):
        return ""


def _install_id() -> str:
    """Return a stable, anonymous install id stored at ``~/.llamphouse/telemetry_id``."""
    try:
        path = Path.home() / ".llamphouse" / "telemetry_id"
        if path.exists():
            value = path.read_text().strip()
            if value:
                return value
        path.parent.mkdir(parents=True, exist_ok=True)
        value = uuid.uuid4().hex
        path.write_text(value)
        return value
    except Exception:
        # Fall back to a per-process id if the filesystem is read-only.
        return f"ephemeral-{uuid.uuid4().hex}"


class _TelemetryClient:
    def __init__(self) -> None:
        self._endpoint = os.environ.get(
            "LLAMPHOUSE_TELEMETRY_ENDPOINT", _DEFAULT_ENDPOINT
        )
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._install_id = _install_id()
        self._session_id = uuid.uuid4().hex
        # Optional, opt-in identifier so operators can correlate events
        # from a known deployment/customer. Must be a valid UUID — any
        # other value is silently ignored. Empty string when unset.
        self._tracking_id = _validate_uuid(os.environ.get("LLAMPHOUSE_TRACKING_ID"))
        self._base_props = {
            "llamphouse_version": _LLAMPHOUSE_VERSION,
            "python_version": platform.python_version(),
            "os": platform.system(),
            "arch": platform.machine(),
        }
        # Aggregated usage counters — flushed periodically as a single
        # ``llamphouse_usage`` event. Never holds per-thread / per-run data.
        self._counters: Dict[str, int] = {}
        self._counters_lock = threading.Lock()
        self._usage_window_start = time.time()

    # ── producer side ────────────────────────────────────────────────────
    def record(self, event: str, **props: Any) -> None:
        if not is_enabled():
            return
        try:
            payload = {
                "event": event,
                "ts": time.time(),
                "install_id": self._install_id,
                "session_id": self._session_id,
                **self._base_props,
                "props": props or {},
            }
            if self._tracking_id:
                payload["tracking_id"] = self._tracking_id
            self._ensure_thread()
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                # Drop the oldest event to make room — never block.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(payload)
                except Exception:
                    pass
        except Exception:
            # Telemetry must never raise into the host app.
            pass

    def bump(self, name: str, n: int = 1) -> None:
        """Increment an aggregated counter (no-op unless usage tier is on)."""
        if not _usage_enabled():
            return
        try:
            with self._counters_lock:
                self._counters[name] = self._counters.get(name, 0) + n
            self._ensure_thread()
        except Exception:
            pass

    def observe_run_ms(self, ms: float) -> None:
        """Bucket a run duration (ms) into the run-duration histogram."""
        if not _usage_enabled():
            return
        try:
            seconds = max(0.0, float(ms)) / 1000.0
            label = _RUN_DUR_BUCKET_LABELS[-1]
            for upper, candidate in zip(_RUN_DUR_BUCKETS_S, _RUN_DUR_BUCKET_LABELS):
                if seconds < upper:
                    label = candidate
                    break
            self.bump(f"run_dur_{label}")
        except Exception:
            pass

    def _drain_counters(self) -> Optional[Dict[str, Any]]:
        """Snapshot and reset counters. Returns a usage payload or None."""
        with self._counters_lock:
            if not self._counters:
                return None
            counters = self._counters
            self._counters = {}
            window_start = self._usage_window_start
            self._usage_window_start = time.time()
        return {
            "interval_s": round(self._usage_window_start - window_start, 1),
            "counters": counters,
        }

    def _flush_usage(self) -> None:
        snapshot = self._drain_counters()
        if snapshot is None:
            return
        # Re-enter the standard event path so usage events go through the
        # same queue / batching / shutdown drain as everything else.
        try:
            payload = {
                "event": "llamphouse_usage",
                "ts": time.time(),
                "install_id": self._install_id,
                "session_id": self._session_id,
                **self._base_props,
                "props": snapshot,
            }
            if self._tracking_id:
                payload["tracking_id"] = self._tracking_id
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(payload)
                except Exception:
                    pass
        except Exception:
            pass

    # ── lifecycle ────────────────────────────────────────────────────────
    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            t = threading.Thread(
                target=self._run, name="llamphouse-telemetry", daemon=True
            )
            t.start()
            self._thread = t

    def shutdown(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        try:
            self._thread.join(timeout=timeout)
        except Exception:
            pass

    # ── consumer side ────────────────────────────────────────────────────
    def _run(self) -> None:
        batch: list = []
        last_flush = time.monotonic()
        last_usage_flush = time.monotonic()
        while not self._stop.is_set():
            timeout = max(
                0.1,
                min(
                    _FLUSH_INTERVAL - (time.monotonic() - last_flush),
                    _USAGE_FLUSH_INTERVAL - (time.monotonic() - last_usage_flush),
                ),
            )
            try:
                item = self._queue.get(timeout=timeout)
                batch.append(item)
            except queue.Empty:
                pass

            if (time.monotonic() - last_usage_flush) >= _USAGE_FLUSH_INTERVAL:
                self._flush_usage()
                last_usage_flush = time.monotonic()

            should_flush = (
                len(batch) >= _BATCH_SIZE
                or (batch and (time.monotonic() - last_flush) >= _FLUSH_INTERVAL)
            )
            if should_flush:
                self._send(batch)
                batch = []
                last_flush = time.monotonic()

        # Final usage flush + drain remaining items on shutdown.
        self._flush_usage()
        try:
            while True:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if batch:
            self._send(batch)

    def _send(self, batch: list) -> None:
        if not batch:
            return
        try:
            data = json.dumps({"events": batch}).encode("utf-8")
            req = _urlreq.Request(
                self._endpoint,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"llamphouse/{_LLAMPHOUSE_VERSION}",
                },
            )
            with _urlreq.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                resp.read()
        except (URLError, TimeoutError, OSError, ValueError):
            # Best-effort: drop the batch on failure.
            pass
        except Exception:
            pass


_client: Optional[_TelemetryClient] = None
_client_lock = threading.Lock()


def _get_client() -> _TelemetryClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _TelemetryClient()
    return _client


def record(event: str, **props: Any) -> None:
    """Record a telemetry event. Non-blocking and never raises."""
    if not is_enabled():
        return
    try:
        _get_client().record(event, **props)
    except Exception:
        pass


def bump(name: str, n: int = 1) -> None:
    """Increment an aggregated usage counter. No-op unless usage tier is on."""
    if not _usage_enabled():
        return
    try:
        _get_client().bump(name, n)
    except Exception:
        pass


def observe_run_ms(ms: float) -> None:
    """Bucket a run duration (ms) into the histogram. No-op unless usage tier is on."""
    if not _usage_enabled():
        return
    try:
        _get_client().observe_run_ms(ms)
    except Exception:
        pass


def shutdown(timeout: float = 2.0) -> None:
    """Flush any queued events and stop the worker thread."""
    if _client is None:
        return
    try:
        _client.shutdown(timeout=timeout)
    except Exception:
        pass
