"""The OpenAI Responses and Conversations APIs over the 2.0 engine - driven by the official openai client."""

import httpx
import pytest

openai = pytest.importorskip("openai")

from llamphouse.runtime import Engine  # noqa: E402
from llamphouse.runtime.adapters.assistants import AssistantsAdapter  # noqa: E402
from llamphouse.runtime.adapters.responses import ResponsesAdapter  # noqa: E402
from llamphouse.runtime.api import create_app  # noqa: E402
from llamphouse.sdk import Agent, ToolCallPart  # noqa: E402


class Echo(Agent):
    """Says it back."""

    async def run(self, ctx) -> None:
        said = [m.text for m in await ctx.session.messages()]
        async with ctx.stream() as out:
            await out.write(f"echo: {ctx.input.text} ({len(said)} in session)")


class Weather(Agent):
    async def run(self, ctx) -> None:
        tool = ctx.client.tools[0]
        result = await ctx.client.call_tool(ToolCallPart(call_id="call_1", name=tool.name, arguments={"city": "Oslo"}))
        await ctx.send(f"Oslo: {result.output[0].text}")


class Asker(Agent):
    async def run(self, ctx) -> None:
        city = await ctx.ask("Which city?")
        await ctx.send(f"booked {city.text}")


AGENTS = {"echo": Echo, "weather": Weather, "asker": Asker}


@pytest.fixture
async def client():
    engine = await Engine(AGENTS).open()
    app = create_app(engine, adapters=[ResponsesAdapter(prefix="/v1"), AssistantsAdapter(prefix="/v1")])
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h/v1", timeout=10)
    yield openai.AsyncOpenAI(base_url="http://h/v1", api_key="unused", http_client=http)
    await http.aclose()
    await engine.close()


async def test_a_response_and_a_conversation(client):
    models = await client.models.list()
    assert {m.id for m in models.data} == {"echo", "weather", "asker"}
    first = await client.responses.create(model="echo", input="hello")
    assert first.status == "completed" and first.output_text == "echo: hello (1 in session)"
    second = await client.responses.create(model="echo", input="again", previous_response_id=first.id)
    assert second.output_text == "echo: again (3 in session)"  # the same thread: it remembers
    conversation = await client.conversations.create(metadata={"topic": "test"},
                                                     items=[{"type": "message", "role": "user", "content": "earlier"}])
    third = await client.responses.create(model="echo", input="now", conversation=conversation.id)
    assert third.output_text == "echo: now (2 in session)"
    items = await client.conversations.items.list(conversation.id, order="asc")
    assert [i.role for i in items.data] == ["user", "user", "assistant"]
    got = await client.responses.retrieve(third.id)
    assert got.output_text == third.output_text


async def test_function_calls_and_questions_continue_the_same_run(client):
    tools = [{"type": "function", "name": "get_weather", "description": "the weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    first = await client.responses.create(model="weather", input="weather in Oslo?", tools=tools)
    calls = [o for o in first.output if o.type == "function_call"]
    assert first.status == "completed" and calls[0].name == "get_weather" and calls[0].arguments == '{"city": "Oslo"}'
    second = await client.responses.create(model="weather", previous_response_id=first.id, tools=tools, input=[
        {"type": "function_call_output", "call_id": calls[0].call_id, "output": "snowing"}])
    assert second.id != first.id and second.output_text == "Oslo: snowing"
    assert second.previous_response_id == first.id

    asked = await client.responses.create(model="asker", input="book a trip")
    assert asked.output_text == "Which city?"
    done = await client.responses.create(model="asker", input="Rome", previous_response_id=asked.id)
    assert done.output_text == "booked Rome"


async def test_streaming(client):
    stream = await client.responses.create(model="echo", input="stream me", stream=True)
    kinds, text = [], ""
    async for event in stream:
        kinds.append(event.type)
        if event.type == "response.output_text.delta":
            text += event.delta
        if event.type == "response.completed":
            final = event.response
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert text == "echo: stream me (1 in session)" == final.output_text


async def test_errors(client):
    with pytest.raises(openai.NotFoundError):
        await client.responses.create(model="nobody", input="x")
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve("run_nope")
