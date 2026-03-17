# Tool Calls

LLAMPHouse has native support for **function calling** (tool calls). Your agent defines tool schemas, the LLM decides when to call them, and you execute the functions and feed results back into the conversation.

## Overview

The tool call flow:

1. Define tool schemas and a tool registry
2. Send the schemas to the LLM with the conversation
3. The LLM responds with tool call requests (captured by `process_stream()`)
4. Execute the tools and submit results
5. Call the LLM again with the tool results
6. Repeat until the LLM produces a text response

## Full example

```python
from llamphouse.core import Agent
from llamphouse.core.context import Context
from llamphouse.core.streaming.adapters.registry import get_adapter
from openai import AsyncOpenAI
from datetime import datetime, timezone
from typing import Any, Callable, Dict
import json

openai_client = AsyncOpenAI()

# 1. Define your tools
def get_current_time(_: dict[str, Any] | None = None) -> str:
    return datetime.now(timezone.utc).isoformat()

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
]

TOOL_REGISTRY: Dict[str, Callable] = {
    "get_current_time": get_current_time,
}


# 2. Create an agent with tools
class ToolAgent(Agent):
    def __init__(self):
        super().__init__(id="tool-agent", tools=TOOL_SCHEMAS)

    async def run(self, context: Context):
        messages = [{"role": "system", "content": "You are helpful. Use tools when needed."}]
        for m in context.messages:
            if m.text:
                messages.append({"role": m.role, "content": m.text})

        adapter = get_adapter("openai")

        # 3. Loop: call LLM → handle tool calls → call LLM again
        for _ in range(3):  # max 3 tool call rounds
            stream = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True,
            )

            full_text = await context.process_stream(stream, adapter)

            # If the LLM produced text, we're done
            if full_text and full_text.strip():
                await context.insert_message(full_text)
                return

            # No text → the LLM requested tool calls
            if not context.pending_tool_calls:
                await context.insert_message("I couldn't process that.")
                return

            # 4. Add the assistant's tool_calls message
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in context.pending_tool_calls
                ],
                "content": None,
            })

            # 5. Execute tools and add results
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

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })

            # Clear pending calls before next iteration
            context.pending_tool_calls.clear()
```

## Key concepts

### Tool schemas

Tool schemas follow the [OpenAI function calling format](https://platform.openai.com/docs/guides/function-calling). Pass them to your agent's `tools` parameter:

```python
agent = MyAgent(id="my-agent", tools=TOOL_SCHEMAS)
```

### `context.pending_tool_calls`

After `process_stream()` completes, if the LLM requested tool calls instead of producing text, they're available in `context.pending_tool_calls`. Each entry is a dict:

```python
{
    "id": "call_abc123",      # Tool call ID
    "name": "get_weather",    # Function name
    "arguments": '{"city": "London"}',  # JSON string of arguments
}
```

### Tool execution loop

The standard pattern is a loop that:

1. Calls the LLM with tools
2. Checks if the response is text (done) or tool calls (continue)
3. Executes requested tools
4. Appends results to the message history
5. Repeats

Set a maximum iteration count to prevent infinite loops.

## Next steps

- [Streaming](streaming.md) — streaming works seamlessly with tool calls
- [Context](../concepts/context.md) — full context API reference
- [Examples](../examples.md) — see example 04_ToolCall and 10_A2A_ToolCall
