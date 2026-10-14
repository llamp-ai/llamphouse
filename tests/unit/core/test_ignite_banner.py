"""Regression tests for the ignite banner's event-queue line.

Background
----------
``event_queue_class`` accepts either a class or a *callable factory instance*.
The factory form is the documented way to run distributed mode::

    LLAMPHouse(event_queue_class=RedisEventQueueFactory(redis_url))

``RedisEventQueueFactory``'s own docstring says to pass it this way, and
``DistributedWorker`` constructs its context with exactly that shape. The
runtime only ever *calls* the value -- ``state.queue_class(assistant_id,
thread_id)`` -- so an instance works everywhere it is consumed.

The banner did not. ``__print_ignite`` read ``.__name__`` off the value
directly, which exists on classes but not on instances, so ``ignite()`` raised
``AttributeError`` *after* the app was fully constructed and every other banner
line had printed:

    AttributeError: 'RedisEventQueueFactory' object has no attribute '__name__'

The default (``InMemoryEventQueue``) is a class, so the bug was invisible until
someone enabled distributed mode -- at which point the server could not start
at all, for the sake of one line of startup logging.

``_callable_name`` already existed for this, and was already applied to the
same value when building the telemetry payload. The banner just missed it.

These tests pin the post-fix behaviour. The first fails on the pre-fix code
with the AttributeError above; the second guards against "fixing" it by
switching to ``type(...).__name__``, which would regress the default to
printing the useless ``type`` instead of ``InMemoryEventQueue``.
"""
import pytest

from llamphouse.core import Agent, Context, LLAMPHouse
from llamphouse.core import llamphouse as llamphouse_module
from llamphouse.core.streaming.event_queue.in_memory_event_queue import (
    InMemoryEventQueue,
)
from llamphouse.core.streaming.event_queue.redis_event_queue import (
    RedisEventQueueFactory,
)

pytestmark = pytest.mark.unit


class BannerAgent(Agent):
    async def run(self, context: Context):
        return None


def _banner_lines(house, monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(llamphouse_module.llamphouse_logger, "info", lines.append)
    # Name-mangled private: the banner is only reachable through ignite(),
    # which would also start uvicorn.
    house._LLAMPHouse__print_ignite("0.0.0.0", 8000)
    return lines


def test_banner_names_a_factory_instance_passed_as_event_queue(monkeypatch):
    """Distributed mode: an instance has no __name__, so read it safely."""
    house = LLAMPHouse(
        agents=[BannerAgent(id="banner")],
        adapters=[],
        tracing_store=None,
        event_queue_class=RedisEventQueueFactory("redis://localhost:6379/0"),
    )

    lines = _banner_lines(house, monkeypatch)

    event_queue_line = next(line for line in lines if "Event queue" in line)
    assert "RedisEventQueueFactory" in event_queue_line


def test_banner_still_names_the_class_in_the_default_case(monkeypatch):
    """In-process mode: the value is a class, and its own name is the useful
    one. ``type(value).__name__`` would print ``type`` here."""
    house = LLAMPHouse(
        agents=[BannerAgent(id="banner")],
        adapters=[],
        tracing_store=None,
    )

    lines = _banner_lines(house, monkeypatch)

    event_queue_line = next(line for line in lines if "Event queue" in line)
    assert InMemoryEventQueue.__name__ in event_queue_line


def test_callable_name_handles_both_shapes():
    assert llamphouse_module._callable_name(InMemoryEventQueue) == "InMemoryEventQueue"
    assert (
        llamphouse_module._callable_name(RedisEventQueueFactory("redis://localhost"))
        == "RedisEventQueueFactory"
    )
