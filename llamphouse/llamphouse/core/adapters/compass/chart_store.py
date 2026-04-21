"""
Compass Chart Store
~~~~~~~~~~~~~~~~~~~
Global library of reusable chart definitions.

Charts are saved independently of dashboards so the same chart can be
placed on multiple dashboards.  Each chart stores only its *definition*
(SQL, type, column mapping, title) – not its size or position.  Layout
information (col_span, height_px) lives in the dashboard's chart-slot.

Storage location (in order of precedence):
1. ``path`` constructor argument
2. ``COMPASS_CHARTS_FILE`` environment variable
3. ``./compass_charts.json`` in the current working directory
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_FILE = Path(os.getenv("COMPASS_CHARTS_FILE", "compass_charts.json"))


class ChartStore:
    """Simple JSON-file-backed store for Compass chart definitions."""

    def __init__(self, path: Path = _DEFAULT_FILE):
        self._path = path
        self._charts: dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._charts = {c["id"]: c for c in raw}
            except Exception:
                self._charts = {}

    def _save(self):
        try:
            self._path.write_text(
                json.dumps(list(self._charts.values()), indent=2, default=str)
            )
        except Exception:
            pass

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def list(self) -> list[dict]:
        items = list(self._charts.values())
        items.sort(key=lambda c: c.get("created_at", 0))
        return items

    def get(self, chart_id: str) -> Optional[dict]:
        return self._charts.get(chart_id)

    def create(
        self,
        title: str,
        sql: str = "",
        chart_type: str = "table",
        x_column: Optional[str] = None,
        y_columns: Optional[list] = None,
    ) -> dict:
        now = round(datetime.now(timezone.utc).timestamp(), 3)
        c = {
            "id": str(uuid.uuid4()),
            "title": title or "Untitled Chart",
            "sql": sql,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_columns": y_columns or [],
            "created_at": now,
            "updated_at": now,
        }
        self._charts[c["id"]] = c
        self._save()
        return c

    def update(self, chart_id: str, data: dict) -> Optional[dict]:
        c = self._charts.get(chart_id)
        if not c:
            return None
        for k in ("title", "sql", "chart_type", "x_column", "y_columns"):
            if k in data:
                c[k] = data[k]
        c["updated_at"] = round(datetime.now(timezone.utc).timestamp(), 3)
        self._save()
        return c

    def delete(self, chart_id: str) -> bool:
        if chart_id not in self._charts:
            return False
        del self._charts[chart_id]
        self._save()
        return True
