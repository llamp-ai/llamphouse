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

    async def reply(self, content, metadata=None):
        self.replies.append(content)

    def send_chunk(self, text):
        self.chunks.append(text)


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
