from datetime import datetime, timezone
from typing import Any, cast

import pytest

from llamphouse.core.types.message import MessageObject, TextPart
from llamphouse.core.wrappers import BaseAgentWrapper, LangGraphAgent


class DummyContext:
    def __init__(self, messages=None):
        self.thread_id = "thread-1"
        self.run_id = "run-1"
        self.messages = messages or []
        self.replies = []
        self.chunks = []
        self.steps_started = []
        self.steps_completed = []

    async def reply(self, content, metadata=None):
        self.replies.append(content)

    def send_chunk(self, text):
        self.chunks.append(text)

    async def start_step(self, name, input=None, metadata=None):
        step_id = f"step-{len(self.steps_started) + 1}"
        self.steps_started.append(
            {
                "id": step_id,
                "name": name,
                "input": input,
                "metadata": metadata,
            }
        )

        class _Step:
            def __init__(self, id):
                self.id = id

        return _Step(step_id)

    async def complete_step(self, step_id, output=None, error=None, status=None):
        self.steps_completed.append(
            {
                "id": step_id,
                "output": output,
                "error": error,
                "status": status,
            }
        )


class WrapperHarness(BaseAgentWrapper):
    def __init__(self, *, framework_output=None, **kwargs):
        super().__init__(**kwargs)
        self.framework_output = framework_output
        self.last_state = None

    async def invoke_framework(self, context, state):
        self.last_state = state
        return self.framework_output


class StreamingGraph:
    async def astream(self, state):
        yield {"text": "hel"}
        yield {"delta": "lo"}

    async def ainvoke(self, state):
        return {"output": "fallback"}


class InvokeOnlyGraph:
    async def ainvoke(self, state):
        return {"output": "invoke-only"}


class StreamNoTextGraph:
    async def astream(self, state):
        yield {"ignored": True}

    async def ainvoke(self, state):
        return {"output": "from-ainvoke"}


class NoInvokeGraph:
    async def astream(self, state):
        yield {"text": "x"}


class EventsGraph:
    async def astream_events(self, state, version="v1"):
        yield {
            "event": "on_chain_start",
            "name": "respond",
            "data": {"input": state},
        }
        yield {
            "event": "on_chain_stream",
            "name": "respond",
            "data": {"chunk": {"output": "hello-from-events"}},
            "output": "hello-from-events",
        }
        yield {
            "event": "on_chain_end",
            "name": "respond",
            "data": {"output": {"output": "hello-from-events"}},
        }

    async def ainvoke(self, state):
        return {"output": "ainvoke-fallback"}


@pytest.mark.asyncio
async def test_base_wrapper_default_state_mapper_includes_messages():
    msg = MessageObject(
        id="msg-1",
        created_at=datetime.now(timezone.utc),
        thread_id="thread-1",
        role="user",
        parts=[TextPart(text="hello")],
        run_id="run-1",
        assistant_id="assistant-1",
        metadata={"k": "v"},
    )
    ctx = DummyContext(messages=[msg])
    wrapper = WrapperHarness(id="w1", framework_output={"output": "ok"})

    await wrapper.run(cast(Any, ctx))

    assert wrapper.last_state is not None
    assert wrapper.last_state["thread_id"] == "thread-1"
    assert wrapper.last_state["run_id"] == "run-1"
    assert wrapper.last_state["messages"][0]["role"] == "user"
    assert wrapper.last_state["messages"][0]["content"][0]["text"] == "hello"
    assert ctx.replies == ["ok"]


@pytest.mark.asyncio
async def test_base_wrapper_output_mapper_handles_common_shapes():
    ctx = DummyContext()
    wrapper = WrapperHarness(
        id="w2",
        framework_output={"messages": [{"content": "from-messages"}]},
    )

    await wrapper.run(cast(Any, ctx))

    assert ctx.replies == ["from-messages"]


@pytest.mark.asyncio
async def test_langgraph_streaming_sends_chunks_and_returns_combined_output():
    ctx = DummyContext()
    agent = LangGraphAgent(id="lg1", graph=StreamingGraph(), stream=True)

    output = await agent.invoke_framework(cast(Any, ctx), {"messages": []})

    assert output == {"output": "hello"}
    assert ctx.chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_langgraph_falls_back_to_ainvoke_when_stream_has_no_text():
    ctx = DummyContext()
    agent = LangGraphAgent(id="lg2", graph=StreamNoTextGraph(), stream=True)

    output = await agent.invoke_framework(cast(Any, ctx), {"messages": []})

    assert output == {"output": "from-ainvoke"}
    assert ctx.chunks == []


@pytest.mark.asyncio
async def test_langgraph_uses_ainvoke_when_streaming_disabled():
    ctx = DummyContext()
    agent = LangGraphAgent(id="lg3", graph=InvokeOnlyGraph(), stream=False)

    output = await agent.invoke_framework(cast(Any, ctx), {"messages": []})

    assert output == {"output": "invoke-only"}


@pytest.mark.asyncio
async def test_langgraph_requires_ainvoke_when_no_stream_output():
    ctx = DummyContext()
    agent = LangGraphAgent(id="lg4", graph=NoInvokeGraph(), stream=False)

    with pytest.raises(RuntimeError, match="ainvoke"):
        await agent.invoke_framework(cast(Any, ctx), {"messages": []})


@pytest.mark.asyncio
async def test_langgraph_events_stream_maps_node_to_steps():
    ctx = DummyContext()
    agent = LangGraphAgent(
        id="lg5",
        graph=EventsGraph(),
        stream=True,
        map_nodes_to_steps=True,
    )

    output = await agent.invoke_framework(cast(Any, ctx), {"messages": []})

    assert output == {"output": "hello-from-events"}
    assert ctx.chunks == ["hello-from-events"]
    assert len(ctx.steps_started) == 1
    assert ctx.steps_started[0]["name"] == "respond"
    assert len(ctx.steps_completed) == 1
    assert ctx.steps_completed[0]["status"] == "completed"
    assert ctx.steps_started[0]["metadata"]["framework"] == "langgraph"
    assert ctx.steps_started[0]["metadata"]["step_type"] == "langgraph_node"
    assert ctx.steps_started[0]["metadata"]["node_name"] == "respond"
    assert "state" in ctx.steps_started[0]["metadata"]
    assert "state" in ctx.steps_completed[0]["output"]
