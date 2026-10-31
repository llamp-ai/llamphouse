"""Models' streamed answers into a run: OpenAI (Responses), Anthropic and Gemini events, read by their shape."""

from types import SimpleNamespace as O

import pytest

from llamphouse.sdk import Agent, Usage
from llamphouse.helpers.streams import (Finished, StreamFailed, TextDelta, ToolCallDelta, anthropic, gemini, openai,
                                        pipe)
from llamphouse.sdk.testing import FakeRun


def openai_events():  # the Responses API's stream, as `responses.create(..., stream=True)` gives
    yield O(type="response.created", response=O(id="resp_1", model="gpt-x", usage=None))
    yield O(type="response.output_item.added", output_index=0, item=O(type="message", id="msg_1"))
    yield O(type="response.output_text.delta", item_id="msg_1", output_index=0, delta="Hel")
    yield O(type="response.output_text.delta", item_id="msg_1", output_index=0, delta="lo")
    yield O(type="response.output_item.added", output_index=1,
            item=O(type="function_call", id="fc_1", call_id="call_1", name="weather", arguments=""))
    yield O(type="response.function_call_arguments.delta", item_id="fc_1", output_index=1, delta='{"ci')
    yield O(type="response.function_call_arguments.delta", item_id="fc_1", output_index=1, delta='ty": "Lisbon"}')
    yield O(type="response.completed", response=O(id="resp_1", model="gpt-x", usage=O(
        input_tokens=12, output_tokens=7, input_tokens_details=O(cached_tokens=4),
        output_tokens_details=O(reasoning_tokens=0))))


async def anthropic_events():  # an async stream, as `AsyncAnthropic` gives
    for event in (
        O(type="message_start", message=O(id="msg_1", model="claude-x", usage=O(input_tokens=20, output_tokens=1,
                                                                                cache_read_input_tokens=None))),
        O(type="content_block_start", index=0, content_block=O(type="text", text="")),
        O(type="content_block_delta", index=0, delta=O(type="text_delta", text="Checking")),
        O(type="content_block_start", index=1, content_block=O(type="tool_use", id="toolu_1", name="weather")),
        O(type="content_block_delta", index=1, delta=O(type="input_json_delta", partial_json='{"city":')),
        O(type="content_block_delta", index=1, delta=O(type="input_json_delta", partial_json=' "Rome"}')),
        O(type="message_delta", delta=O(stop_reason="tool_use"), usage=O(output_tokens=30)),
        O(type="message_stop"),
    ):
        yield event


def gemini_chunks():
    yield {"candidates": [{"content": {"parts": [{"text": "It is "}]}}]}  # dicts work too
    yield O(candidates=[O(content=O(parts=[O(text="sunny.", function_call=None)]), finish_reason="STOP")],
            usage_metadata=O(prompt_token_count=9, candidates_token_count=3, cached_content_token_count=None,
                             thoughts_token_count=2), model_version="gemini-x", response_id="r1")


async def collect(deltas):
    return [d async for d in deltas]


async def test_openai():
    deltas = await collect(openai(openai_events()))
    assert deltas[:2] == [TextDelta("Hel"), TextDelta("lo")]
    assert deltas[2] == ToolCallDelta(index=0, id="call_1", name="weather", arguments="")
    assert deltas[-1] == Finished(reason="tool_calls", model="gpt-x", response_id="resp_1", input_tokens=12,
                                  output_tokens=7, details={"cached_tokens": 4, "reasoning_tokens": 0})
    said = await pipe(FakeRun("x"), openai(openai_events()))
    assert [(c.call_id, c.name, dict(c.arguments)) for c in said.tool_calls] == [("call_1", "weather", {"city": "Lisbon"})]


async def test_openai_failed_and_cut_off():
    failed = [O(type="response.failed", response=O(error=O(code="server_error", message="try again")))]
    with pytest.raises(StreamFailed, match="server_error: try again"):
        await collect(openai(failed))
    cut = [O(type="response.output_text.delta", delta="Once upon"),
           O(type="response.incomplete", response=O(id="r", model="m", usage=None,
                                                    incomplete_details=O(reason="max_output_tokens")))]
    assert (await collect(openai(cut)))[-1].reason == "length"


async def test_anthropic():
    deltas = await collect(anthropic(anthropic_events()))
    assert TextDelta("Checking") in deltas
    assert deltas[-1] == Finished(reason="tool_calls", model="claude-x", response_id="msg_1", input_tokens=20,
                                  output_tokens=30)


async def test_gemini():
    deltas = await collect(gemini(gemini_chunks()))
    assert deltas[:2] == [TextDelta("It is "), TextDelta("sunny.")]
    assert deltas[-1] == Finished(reason="stop", model="gemini-x", response_id="r1", input_tokens=9, output_tokens=3,
                                  details={"reasoning_tokens": 2})


class Forecaster(Agent):
    async def run(self, ctx) -> list:
        said = await pipe(ctx, anthropic(anthropic_events()))
        return [said.text, [(c.call_id, c.name, dict(c.arguments)) for c in said.tool_calls]]


async def test_pipe_streams_into_the_run_and_gathers_tool_calls():
    run = FakeRun("weather in Rome?")
    assert await run.start(Forecaster()) == ["Checking", [["toolu_1", "weather", {"city": "Rome"}]]]
    assert run.replies[-1].text == "Checking"
    assert Usage(model="claude-x", input_tokens=20, output_tokens=30) in run.emitted


class Plain(Agent):
    async def run(self, ctx) -> str:
        async def words():
            for w in ("one ", "two"):
                yield w
        said = await pipe(ctx, words())
        assert said.tool_calls == [] and said.usage is None
        return said.text


async def test_pipe_takes_plain_text_and_a_tool_call_alone_writes_no_message():
    run = FakeRun("go")
    assert await run.start(Plain()) == "one two"

    class OnlyCalls(Agent):
        async def run(self, ctx) -> int:
            said = await pipe(ctx, [ToolCallDelta(index=0, id="c", name="f", arguments="{}"), Finished(reason="tool_calls")])
            return len(said.tool_calls)

    run = FakeRun("go")
    assert await run.start(OnlyCalls()) == 1 and run.replies == []
