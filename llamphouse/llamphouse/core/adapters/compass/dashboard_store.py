"""
Compass Dashboard Store
~~~~~~~~~~~~~~~~~~~~~~~
Lightweight file-based store for Compass dashboard definitions.

Each dashboard has a title, description, and a list of *chart slots* – lightweight
references that link a global chart definition (by id) to layout information
(col_span, height_px, position).  The actual chart SQL / type / column config
lives in the ChartStore.

Migration: existing dashboards that contain embedded chart objects (old format)
are transparently migrated on first load – the embedded definitions are extracted
into the ChartStore, and the slot list is rewritten with chart_id references.

Storage location (in order of precedence):
1. ``path`` constructor argument
2. ``COMPASS_DASHBOARDS_FILE`` environment variable
3. ``./compass_dashboards.json`` in the current working directory
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .chart_store import ChartStore

_DEFAULT_FILE = Path(os.getenv("COMPASS_DASHBOARDS_FILE", "compass_dashboards.json"))


class DashboardStore:
    """Simple JSON-file-backed store for Compass dashboard definitions."""

    def __init__(self, path: Path = _DEFAULT_FILE, chart_store: "Optional[ChartStore]" = None):
        self._path = path
        self._chart_store = chart_store
        self._dashboards: dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._dashboards = {d["id"]: d for d in raw}
                self._migrate_embedded_charts()
            except Exception:
                self._dashboards = {}

    def _migrate_embedded_charts(self):
        """One-time migration: convert old embedded chart objects into chart_id slots."""
        if self._chart_store is None:
            return
        changed = False
        for d in self._dashboards.values():
            new_slots = []
            for item in d.get("charts", []):
                if "chart_id" in item:
                    # Already a slot reference
                    new_slots.append(item)
                elif "sql" in item:
                    # Old embedded format — extract into chart library
                    existing = self._chart_store.get(item["id"]) if item.get("id") else None
                    if not existing:
                        chart = self._chart_store.create(
                            title=item.get("title", "Untitled Chart"),
                            sql=item.get("sql", ""),
                            chart_type=item.get("chart_type", "table"),
                            x_column=item.get("x_column"),
                            y_columns=item.get("y_columns", []),
                        )
                        chart_id = chart["id"]
                    else:
                        chart_id = existing["id"]
                    new_slots.append({
                        "chart_id": chart_id,
                        "col_span": item.get("col_span", 2),
                        "height_px": item.get("height_px", 280),
                    })
                    changed = True
                # else: skip unknown items
            d["charts"] = new_slots
        if changed:
            self._save()

    def _save(self):
        try:
            self._path.write_text(
                json.dumps(list(self._dashboards.values()), indent=2, default=str)
            )
        except Exception:
            pass

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def list(self) -> list[dict]:
        items = list(self._dashboards.values())
        items.sort(key=lambda d: d.get("created_at", 0))
        return items

    def get(self, dashboard_id: str) -> Optional[dict]:
        return self._dashboards.get(dashboard_id)

    def create(self, title: str, description: str = "") -> dict:
        now = round(datetime.now(timezone.utc).timestamp(), 3)
        d = {
            "id": str(uuid.uuid4()),
            "title": title or "Untitled Dashboard",
            "description": description,
            "charts": [],   # list of {chart_id, col_span, height_px}
            "created_at": now,
            "updated_at": now,
        }
        self._dashboards[d["id"]] = d
        self._save()
        return d

    def update(self, dashboard_id: str, data: dict) -> Optional[dict]:
        d = self._dashboards.get(dashboard_id)
        if not d:
            return None
        for k in ("title", "description", "charts"):
            if k in data:
                d[k] = data[k]
        d["updated_at"] = round(datetime.now(timezone.utc).timestamp(), 3)
        self._save()
        return d

    def delete(self, dashboard_id: str) -> bool:
        if dashboard_id not in self._dashboards:
            return False
        del self._dashboards[dashboard_id]
        self._save()
        return True
