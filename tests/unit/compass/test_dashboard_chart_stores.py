import json

import pytest

from llamphouse.core.adapters.compass.chart_store import ChartStore
from llamphouse.core.adapters.compass.dashboard_store import DashboardStore


pytestmark = pytest.mark.unit


def test_chart_store_persists_utf8_crud_round_trip(tmp_path):
    path = tmp_path / "charts.json"
    store = ChartStore(path=path)

    chart = store.create(
        title="\u0e2a\u0e23\u0e38\u0e1b\u0e23\u0e31\u0e19",
        sql="select status, count(*) as total from runs group by status",
        chart_type="bar",
        x_column="status",
        y_columns=["total"],
    )
    updated = store.update(chart["id"], {"title": "\u0e23\u0e31\u0e19\u0e15\u0e32\u0e21\u0e2a\u0e16\u0e32\u0e19\u0e30"})

    reloaded = ChartStore(path=path)

    assert updated is not None
    assert reloaded.get(chart["id"])["title"] == "\u0e23\u0e31\u0e19\u0e15\u0e32\u0e21\u0e2a\u0e16\u0e32\u0e19\u0e30"
    assert reloaded.get(chart["id"])["y_columns"] == ["total"]
    assert reloaded.delete(chart["id"]) is True
    assert ChartStore(path=path).list() == []


def test_dashboard_store_persists_chart_slots_round_trip(tmp_path):
    path = tmp_path / "dashboards.json"
    chart_id = "chart_runs_by_status"
    store = DashboardStore(path=path)

    dashboard = store.create(
        title="\u0e41\u0e14\u0e0a\u0e1a\u0e2d\u0e23\u0e4c\u0e14\u0e23\u0e31\u0e19",
        description="\u0e20\u0e32\u0e1e\u0e23\u0e27\u0e21",
    )
    updated = store.update(
        dashboard["id"],
        {
            "charts": [
                {
                    "chart_id": chart_id,
                    "col_span": 2,
                    "height_px": 320,
                    "position": 1,
                }
            ]
        },
    )

    reloaded = DashboardStore(path=path)

    assert updated is not None
    assert reloaded.get(dashboard["id"])["title"] == "\u0e41\u0e14\u0e0a\u0e1a\u0e2d\u0e23\u0e4c\u0e14\u0e23\u0e31\u0e19"
    assert reloaded.get(dashboard["id"])["charts"] == [
        {
            "chart_id": chart_id,
            "col_span": 2,
            "height_px": 320,
            "position": 1,
        }
    ]
    assert reloaded.delete(dashboard["id"]) is True
    assert DashboardStore(path=path).list() == []


def test_dashboard_store_migrates_embedded_charts_to_chart_library(tmp_path):
    dashboard_path = tmp_path / "dashboards.json"
    chart_path = tmp_path / "charts.json"
    embedded_dashboard = {
        "id": "dashboard_1",
        "title": "Legacy",
        "description": "",
        "charts": [
            {
                "id": "legacy_chart",
                "title": "Legacy chart",
                "sql": "select 1 as value",
                "chart_type": "table",
                "x_column": "value",
                "y_columns": ["value"],
                "col_span": 3,
                "height_px": 400,
            }
        ],
        "created_at": 1,
        "updated_at": 1,
    }
    dashboard_path.write_text(json.dumps([embedded_dashboard]), encoding="utf-8")

    chart_store = ChartStore(path=chart_path)
    dashboard_store = DashboardStore(path=dashboard_path, chart_store=chart_store)

    migrated_dashboard = dashboard_store.get("dashboard_1")
    charts = chart_store.list()
    assert len(charts) == 1
    assert charts[0]["title"] == "Legacy chart"
    assert charts[0]["sql"] == "select 1 as value"
    assert migrated_dashboard["charts"] == [
        {
            "chart_id": charts[0]["id"],
            "col_span": 3,
            "height_px": 400,
        }
    ]

    persisted = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert persisted[0]["charts"] == migrated_dashboard["charts"]
