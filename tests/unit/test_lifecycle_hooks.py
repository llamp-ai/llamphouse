"""Tests for Agent on_startup / on_shutdown lifecycle hooks."""

import asyncio
import pytest
import pytest_asyncio

from llamphouse.core import LLAMPHouse, Agent, Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.workers.async_worker import AsyncWorker


class LifecycleAgent(Agent):
    """Agent that records lifecycle events for test assertions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = False
        self.stopped = False

    async def on_startup(self):
        self.started = True

    async def on_shutdown(self):
        self.stopped = True

    async def run(self, context: Context):
        await context.insert_message("ok")


class FailingStartupAgent(Agent):
    """Agent whose on_startup raises — must not prevent other agents from starting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopped = False

    async def on_startup(self):
        raise RuntimeError("startup boom")

    async def on_shutdown(self):
        self.stopped = True

    async def run(self, context: Context):
        await context.insert_message("ok")


class FailingShutdownAgent(Agent):
    """Agent whose on_shutdown raises — must not prevent other agents from shutting down."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = False

    async def on_startup(self):
        self.started = True

    async def on_shutdown(self):
        raise RuntimeError("shutdown boom")

    async def run(self, context: Context):
        await context.insert_message("ok")


def _make_app(agents):
    return LLAMPHouse(
        agents=agents,
        authenticator=None,
        worker=AsyncWorker(time_out=5.0),
        event_queue_class=InMemoryEventQueue,
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        compass=False,
    )


async def _run_lifespan(app: LLAMPHouse):
    """Drive the ASGI lifespan manually: startup → yield → shutdown."""
    ctx = app._lifespan(app.fastapi)
    await ctx.__aenter__()
    await ctx.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_on_startup_called():
    agent = LifecycleAgent(id="lc-1")
    app = _make_app([agent])

    assert not agent.started
    await _run_lifespan(app)
    assert agent.started


@pytest.mark.asyncio
async def test_on_shutdown_called():
    agent = LifecycleAgent(id="lc-2")
    app = _make_app([agent])

    assert not agent.stopped
    await _run_lifespan(app)
    assert agent.stopped


@pytest.mark.asyncio
async def test_startup_before_shutdown():
    """on_startup is called before on_shutdown."""
    events = []

    class OrderAgent(Agent):
        async def on_startup(self):
            events.append("startup")

        async def on_shutdown(self):
            events.append("shutdown")

        async def run(self, context: Context):
            pass

    agent = OrderAgent(id="lc-order")
    app = _make_app([agent])
    await _run_lifespan(app)

    assert events == ["startup", "shutdown"]


@pytest.mark.asyncio
async def test_multiple_agents_all_called():
    agents = [LifecycleAgent(id=f"lc-multi-{i}") for i in range(3)]
    app = _make_app(agents)

    await _run_lifespan(app)

    for agent in agents:
        assert agent.started, f"{agent.id} on_startup was not called"
        assert agent.stopped, f"{agent.id} on_shutdown was not called"


@pytest.mark.asyncio
async def test_failing_startup_does_not_block_others():
    good_agent = LifecycleAgent(id="lc-good")
    bad_agent = FailingStartupAgent(id="lc-bad-start")
    app = _make_app([bad_agent, good_agent])

    await _run_lifespan(app)

    # good agent should still have started and stopped despite bad_agent's startup failure
    assert good_agent.started
    assert good_agent.stopped
    # bad agent's shutdown should still be called
    assert bad_agent.stopped


@pytest.mark.asyncio
async def test_failing_shutdown_does_not_block_others():
    good_agent = LifecycleAgent(id="lc-good2")
    bad_agent = FailingShutdownAgent(id="lc-bad-stop")
    app = _make_app([bad_agent, good_agent])

    await _run_lifespan(app)

    # both should have started
    assert bad_agent.started
    assert good_agent.started
    # good agent should still have been shut down despite bad_agent's failure
    assert good_agent.stopped


@pytest.mark.asyncio
async def test_default_hooks_are_noop():
    """An agent that does not override the hooks should not error."""

    class PlainAgent(Agent):
        async def run(self, context: Context):
            pass

    agent = PlainAgent(id="lc-plain")
    app = _make_app([agent])
    # Should not raise
    await _run_lifespan(app)
