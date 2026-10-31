"""What triggers share: a value by its path in a payload, busy policies, thread labels."""

from __future__ import annotations

from typing import Any, Mapping

BUSY = ("send", "skip", "reject")
LABEL_KEY = "llamphouse.label"


def resolve(data: Any, path: str) -> tuple[bool, Any]:
    """A value by its path in a payload: `$.order.id`, `order.items.0.sku`."""
    value = data
    for key in path.removeprefix("$").strip(".").split("."):
        if not key:
            continue
        if isinstance(value, Mapping) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return False, None
    return True, value


def check_busy(busy: str) -> str:
    if busy not in BUSY:
        raise ValueError(f"busy: one of {BUSY}, not {busy!r}")
    return busy


def thread_label(thread: Any) -> str | None:
    if thread is None or thread == "new":
        return None
    if isinstance(thread, Mapping) and set(thread) == {"label"} and isinstance(thread["label"], str):
        return thread["label"]
    raise ValueError(f"thread: {{label: ...}} or 'new', not {thread!r}")
