"""Shared filter helpers for data stores.

A ``filter`` is a ``{"field", "operator", "value", "value2"?}`` dict produced
by the Compass FilterBuilder component.  This module knows how to:

* turn one filter into a SQLAlchemy clause given a column reference;
* turn one filter into a Python predicate over a raw value.

Stores declare their own allowlists of (field name → column / extractor) so
unsupported fields are silently dropped — this matches the FilterBuilder
contract (UI is permissive, server is strict).
"""
from __future__ import annotations

import json as _json
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional


# ── Type coercion ──────────────────────────────────────────────────────────

def _to_ts(val: Any) -> Optional[float]:
    """Coerce a filter value (ISO date, epoch number, or numeric string) to a
    POSIX timestamp (seconds since epoch).  Returns None on failure."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        # Accepts "2024-03-15" or full ISO datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _start_of_day(ts: float) -> float:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


# ── SQLAlchemy clause builder ──────────────────────────────────────────────

def to_sqla_clause(column, kind: str, filt: dict):
    """Return a SQLAlchemy clause for the given filter, or None to skip.

    ``column`` is a column reference (``Thread.id`` etc.).
    ``kind`` is the filter type: ``"string"``, ``"date"``, ``"number"``, ``"json_string"``.
    """
    op = filt.get("operator")
    val = filt.get("value")
    val2 = filt.get("value2")

    if kind in ("string", "json_string"):
        # For JSONB metadata, cast to string for LIKE/equals comparisons.
        if kind == "json_string":
            from sqlalchemy import cast, String
            target = cast(column, String)
        else:
            target = column
        if op == "contains":      return target.ilike(f"%{val}%") if val else None
        if op == "not_contains":  return ~target.ilike(f"%{val}%") if val else None
        if op == "equals":        return target == val if val is not None else None
        if op == "not_equals":    return target != val if val is not None else None
        if op == "starts_with":   return target.ilike(f"{val}%") if val else None
        if op == "ends_with":     return target.ilike(f"%{val}") if val else None
        if op == "is_empty":      return (target == None) | (target == "")  # noqa: E711
        if op == "is_not_empty":  return (target != None) & (target != "")  # noqa: E711
        return None

    if kind == "date":
        ts = _to_ts(val)
        ts2 = _to_ts(val2)
        if op == "is_after"  and ts is not None: return column > ts
        if op == "is_before" and ts is not None: return column < ts
        if op == "is_on"     and ts is not None:
            start = _start_of_day(ts)
            return (column >= start) & (column < start + 86400)
        if op == "is_between" and ts is not None and ts2 is not None:
            lo, hi = sorted((ts, ts2))
            return (column >= lo) & (column <= hi)
        return None

    if kind == "number":
        try:
            num = float(val) if val not in (None, "") else None
            num2 = float(val2) if val2 not in (None, "") else None
        except (TypeError, ValueError):
            return None
        if op == "eq"  and num is not None: return column == num
        if op == "neq" and num is not None: return column != num
        if op == "gt"  and num is not None: return column > num
        if op == "gte" and num is not None: return column >= num
        if op == "lt"  and num is not None: return column < num
        if op == "lte" and num is not None: return column <= num
        if op == "between" and num is not None and num2 is not None:
            lo, hi = sorted((num, num2))
            return (column >= lo) & (column <= hi)
        return None

    return None


# ── Python predicate ───────────────────────────────────────────────────────

def matches(value: Any, kind: str, filt: dict) -> bool:
    """Return True if ``value`` satisfies ``filt`` for in-memory filtering."""
    op = filt.get("operator")
    target = filt.get("value")
    target2 = filt.get("value2")

    if kind == "json_string":
        try:
            value = _json.dumps(value or {}, default=str)
        except Exception:
            value = ""

    if kind in ("string", "json_string"):
        s = "" if value is None else str(value)
        q = "" if target is None else str(target)
        sl, ql = s.lower(), q.lower()
        if op == "contains":     return ql in sl
        if op == "not_contains": return ql not in sl
        if op == "equals":       return s == q
        if op == "not_equals":   return s != q
        if op == "starts_with":  return sl.startswith(ql)
        if op == "ends_with":    return sl.endswith(ql)
        if op == "is_empty":     return s == ""
        if op == "is_not_empty": return s != ""
        return True

    if kind == "date":
        if value is None:
            return False
        ts = float(value)
        t1 = _to_ts(target)
        t2 = _to_ts(target2)
        if op == "is_after"  and t1 is not None: return ts > t1
        if op == "is_before" and t1 is not None: return ts < t1
        if op == "is_on"     and t1 is not None:
            start = _start_of_day(t1)
            return start <= ts < start + 86400
        if op == "is_between" and t1 is not None and t2 is not None:
            lo, hi = sorted((t1, t2))
            return lo <= ts <= hi
        return True

    if kind == "number":
        if value is None: return False
        try:
            v = float(value)
            n1 = float(target) if target not in (None, "") else None
            n2 = float(target2) if target2 not in (None, "") else None
        except (TypeError, ValueError):
            return True
        if op == "eq"  and n1 is not None: return v == n1
        if op == "neq" and n1 is not None: return v != n1
        if op == "gt"  and n1 is not None: return v > n1
        if op == "gte" and n1 is not None: return v >= n1
        if op == "lt"  and n1 is not None: return v < n1
        if op == "lte" and n1 is not None: return v <= n1
        if op == "between" and n1 is not None and n2 is not None:
            lo, hi = sorted((n1, n2))
            return lo <= v <= hi
        return True

    return True


def apply_predicates(items: Iterable, predicates: list[Callable[[Any], bool]]) -> list:
    """Return only the items for which every predicate returns True."""
    return [it for it in items if all(p(it) for p in predicates)]
