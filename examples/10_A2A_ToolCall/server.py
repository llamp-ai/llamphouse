"""
A2A + Tool Calling + Streaming example.

Exposes an assistant over the A2A protocol that:
  • Receives user messages via A2A JSON-RPC
  • Calls OpenAI with streaming enabled and tool definitions
  • Executes tool calls locally (get_current_time, get_weather)
  • Streams the final answer back through A2A SSE events
"""

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Assistant
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter

# ── OpenAI client ────────────────────────────────────────────────────────────
open_client = AsyncOpenAI()

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools.\n"
    "When the user asks about the current time or date, call get_current_time.\n"
    "When the user asks about the weather, call get_weather with the city name."
)

# ── Tool implementations ────────────────────────────────────────────────────

def get_current_time(_args: dict[str, Any] | None = None) -> str:
    """Returns the current UTC datetime as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def get_weather(args: dict[str, Any] | None = None) -> str:
    """Returns a simulated weather report for a city."""
    city = (args or {}).get("city", "Unknown")
    # Simulated weather data
    report = {
        "city": city,
        "temperature_c": 18,
        "condition": "Partly cloudy",
        "humidity_pct": 62,
        "wind_kph": 15,
    }
    return json.dumps(report)


# ── Tool schemas (OpenAI function-calling format) ────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return current datetime (UTC) as ISO-8601 string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return a weather report for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city to look up.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_REGISTRY: Dict[str, Callable] = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}


# ── Assistant ────────────────────────────────────────────────────────────────

class ToolCallingAssistant(Assistant):
    """
    An assistant that uses OpenAI function calling with streaming.
    Tool calls are resolved in a loop; the final text answer is streamed back.
    """

    async def run(self, context: Context):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in context.messages:
            if m.content and hasattr(m.content[0], "text") and m.content[0].text:
                messages.append({"role": m.role, "content": m.content[0].text})

        adapter = get_adapter("openai")
        max_rounds = 5  # guard against infinite tool-call loops

        for _ in range(max_rounds):
            stream = await open_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True,
            )

            # Stream the response — text deltas and tool-call deltas both
            # flow through the event queue to the A2A client.
            # Tool call steps are automatically persisted by the stream handler.
            full_text = await context.handle_completion_stream_async(stream, adapter)

            if full_text and full_text.strip():
                # Model produced a text answer — we're done.
                await context.insert_message(full_text)
                return

            # No text → model requested tool calls.
            if not context.pending_tool_calls:
                await context.insert_message("I'm sorry, I couldn't process that.")
                return

            # Build the assistant message with tool_calls for the conversation
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in context.pending_tool_calls
                ],
                "content": None,
            })

            # Execute each tool and collect outputs
            from llamphouse.core.types.run import ToolOutput
            tool_outputs: list[ToolOutput] = []

            for tc in context.pending_tool_calls:
                func_name = tc["name"]
                try:
                    func_args = json.loads(tc["arguments"])
                except (json.JSONDecodeError, TypeError):
                    func_args = {}

                if func_name in TOOL_REGISTRY:
                    result = TOOL_REGISTRY[func_name](func_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {func_name}"})

                tool_outputs.append(ToolOutput(tool_call_id=tc["id"], output=result))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            # Submit outputs — completes the run step and resets run to in_progress
            await context.submit_tool_outputs(tool_outputs)

            # Loop back — the model will now see the tool results and produce
            # a text answer on the next round.

        # If we exhaust the loop without a final answer, send a fallback.
        await context.insert_message(
            "I'm sorry, I needed too many steps to process this request."
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    assistant = ToolCallingAssistant(
        id="tool-assistant",
        name="Tool-Calling Assistant",
        description="An assistant that can look up the time and weather using tools.",
        tools=TOOL_SCHEMAS,
    )

    llamphouse = LLAMPHouse(
        assistants=[assistant],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
