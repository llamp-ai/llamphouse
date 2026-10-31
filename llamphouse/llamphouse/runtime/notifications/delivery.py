"""Delivering the outbox: POST each delivery, again with a growing wait until 2xx, dead after `attempts`."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from .base import Subscription

log = logging.getLogger("llamphouse.runtime.notifications")


def backoff(attempts: int, base: float = 5.0, cap: float = 3600.0) -> timedelta:
    return timedelta(seconds=min(cap, base * (2 ** max(0, attempts - 1))))


async def deliver_due(store: Any, now: datetime, *, attempts: int = 8, timeout: float = 10.0,
                      client: Any = None) -> int:
    """One round: the due deliveries, POSTed. Returns how many arrived."""
    due = await store.due_deliveries(now, hold=timeout * 3)  # in the order they were written
    if not due:
        return 0
    subscriptions = {s["id"]: Subscription.from_dict(s) for s in await store.subscriptions()}
    own = client is None
    if own:
        import httpx

        client = httpx.AsyncClient(timeout=timeout)
    arrived = 0
    try:
        for d in due:
            sub = subscriptions.get(d["subscription_id"])
            if sub is None:  # unsubscribed meanwhile: nobody to tell
                await store.delivery_failed(d["id"], attempts=d["attempts"], next_at=now, error="unsubscribed",
                                            dead=True)
                continue
            body = json.dumps(d["payload"], separators=(",", ":")).encode()
            headers = {"Content-Type": "application/json", "X-Llamphouse-Delivery": d["id"], **sub.headers}
            if sub.secret:
                headers["X-Llamphouse-Signature"] = "sha256=" + hmac.new(sub.secret.encode(), body,
                                                                         hashlib.sha256).hexdigest()
            error = None
            try:
                response = await client.post(sub.url, content=body, headers=headers)
                if 200 <= response.status_code < 300:
                    await store.delivered(d["id"], now)
                    arrived += 1
                    continue
                error = f"HTTP {response.status_code}"
            except Exception as e:  # noqa: BLE001 - the receiver is away: again later
                error = repr(e)
            tries = d["attempts"] + 1
            await store.delivery_failed(d["id"], attempts=tries, next_at=now + backoff(tries), error=error or "",
                                        dead=tries >= attempts)
            if tries >= attempts:
                log.warning("delivery %s to %s is dead after %d attempts: %s", d["id"], sub.url, tries, error)
    finally:
        if own:
            await client.aclose()
    return arrived
