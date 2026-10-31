"""Backends behind extras: imported when used, with a clear message when the extra isn't installed."""

from __future__ import annotations

import importlib
from typing import Any


def need(module: str, extra: str) -> Any:
    """`module`, or an ImportError that says which extra brings it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"{module} is not installed: pip install 'llamphouse[{extra}]'") from e
