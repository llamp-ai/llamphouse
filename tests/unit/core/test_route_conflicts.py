import pytest
from fastapi import APIRouter

from llamphouse.core import Agent, Context, LLAMPHouse, WebhookTrigger
from llamphouse.core.adapters.base import BaseAPIAdapter
from llamphouse.core import llamphouse as llamphouse_module


pytestmark = pytest.mark.unit


class RouteConflictAgent(Agent):
    async def run(self, context: Context):
        return None


class PrefixAdapter(BaseAPIAdapter):
    def get_routers(self):
        return [APIRouter()]


def test_warn_on_route_conflicts_reports_duplicate_webhook_paths(monkeypatch):
    first_agent = RouteConflictAgent(id="first")
    first_agent.triggers = [WebhookTrigger(path="/triggers/report")]
    second_agent = RouteConflictAgent(id="second")
    second_agent.triggers = [WebhookTrigger(path="triggers/report")]
    warnings = []
    monkeypatch.setattr(llamphouse_module.llamphouse_logger, "warning", warnings.append)
    LLAMPHouse(agents=[first_agent, second_agent], adapters=[], tracing_store=None)

    assert len(warnings) == 1
    assert "Route conflict" in warnings[0]
    assert "second" in warnings[0]
    assert "'/triggers/report'" in warnings[0]
    assert "first" in warnings[0]


def test_warn_on_route_conflicts_reports_trigger_under_adapter_prefix(monkeypatch):
    agent = RouteConflictAgent(id="prefixed")
    agent.triggers = [WebhookTrigger(path="/api/hooks/report")]
    warnings = []
    monkeypatch.setattr(llamphouse_module.llamphouse_logger, "warning", warnings.append)
    LLAMPHouse(
        agents=[agent],
        adapters=[PrefixAdapter(prefix="/api")],
        tracing_store=None,
    )

    assert len(warnings) == 1
    assert "Route conflict" in warnings[0]
    assert "prefixed" in warnings[0]
    assert "'/api/hooks/report'" in warnings[0]
    assert "PrefixAdapter's prefix '/api'" in warnings[0]
