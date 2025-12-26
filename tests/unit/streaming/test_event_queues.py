import asyncio

import pytest

from llamphouse.core.streaming.event import Event
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue

try:
    from llamphouse.core.streaming.event_queue.janus_event_queue import JanusEventQueue
except Exception:
    JanusEventQueue = None

pytestmark = [pytest.mark.asyncio, pytest.mark.unit, pytest.mark.streaming]

def _fallback_event_queue_params():
    params = [pytest.param(InMemoryEventQueue, id="in_memory")]
    if JanusEventQueue is not None:
        params.append(pytest.param(JanusEventQueue, id="janus"))
    return params

try:
    from conftest import event_queue_params # type: ignore
    _QUEUE_PARAMS = event_queue_params()
except Exception:
    _QUEUE_PARAMS = _fallback_event_queue_params()


@pytest.mark.parametrize("queue_cls", _QUEUE_PARAMS)
async def test_add_get_roundtrip(queue_cls):
    """Adds an event and gets it back with matching event/data."""
    q = queue_cls()
    try:
        event = Event("test", "data")
        await q.add(event)
        got = await asyncio.wait_for(q.get(), timeout=0.5)
        assert got.event == "test"
        assert got.data == "data"
    finally:
        await q.close()


@pytest.mark.parametrize("queue_cls", _QUEUE_PARAMS)
async def test_fifo_order(queue_cls):
    """Preserves FIFO order when multiple events are added."""
    q = queue_cls()
    try:
        await q.add(Event("e1", "d1"))
        await q.add(Event("e2", "d2"))
        first = await asyncio.wait_for(q.get(), timeout=0.5)
        second = await asyncio.wait_for(q.get(), timeout=0.5)
        assert (first.event, first.data) == ("e1", "d1")
        assert (second.event, second.data) == ("e2", "d2")
    finally:
        await q.close()


@pytest.mark.parametrize("queue_cls", _QUEUE_PARAMS)
async def test_empty_flag_transitions(queue_cls):
    """empty() reflects state before add, after add, and after get."""
    q = queue_cls()
    try:
        assert q.empty() is True
        await q.add(Event("test", "data"))
        assert q.empty() is False
        _ = await asyncio.wait_for(q.get(), timeout=0.5)
        assert q.empty() is True
    finally:
        await q.close()


@pytest.mark.parametrize("queue_cls", _QUEUE_PARAMS)
async def test_get_nowait_after_add(queue_cls):
    """get_nowait returns the event immediately after add."""
    q = queue_cls()
    try:
        await q.add(Event("test", "data"))
        got = await q.get_nowait()
        assert got is not None
        assert got.event == "test"
        assert got.data == "data"
    finally:
        await q.close()


@pytest.mark.parametrize("queue_cls", _QUEUE_PARAMS)
async def test_close_unblocks_get(queue_cls):
    """Closing the queue causes pending get to return None."""
    q = queue_cls()
    try:
        await q.close()
        got = await asyncio.wait_for(q.get(), timeout=0.5)
        assert got is None
    finally:
        await q.close()

    
@pytest.mark.parametrize("queue_cls", _QUEUE_PARAMS)
async def test_add_after_close_is_noop(queue_cls):
    """Adding after close does not enqueue and get returns None."""
    q = queue_cls()
    try:
        await q.close()
        await q.add(Event("test", "data"))
        got = await asyncio.wait_for(q.get(), timeout=0.5)
        assert got is None
    finally:
        await q.close()


@pytest.mark.parametrize("queue_cls", _QUEUE_PARAMS)
async def test_close_idempotent(queue_cls):
    """Closing the queue twice is safe and does not raise."""
    q = queue_cls()
    try:
        await q.close()
    finally:    
        await q.close()