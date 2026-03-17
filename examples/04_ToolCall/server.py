from llamphouse.core import LLAMPHouse, Agent
from dotenv import load_dotenv
from llamphouse.core.context import Context
from openai import AsyncOpenAI
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.streaming.adapters.registry import get_adapter
from typing import Any, Callable, Dict
from datetime import datetime, timezone
import json

load_dotenv(override=True)

open_client = AsyncOpenAI()
SYSTEM_PROMPT = "You are a helpful assistant. When asked about time/date, always call the get_current_time tool."

def get_current_time(_: dict[str, Any] | None = None) -> str:
    """Returns the current UTC datetime as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return current datetime (UTC) as ISO-8601 string.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

TOOL_REGISTRY: Dict[str, Callable] = {"get_current_time": get_current_time}

# Create a custom agent
class CustomAgent(Agent):

    def __init__(self, agent_id: str):
        super().__init__(id=agent_id, tools=TOOL_SCHEMAS)

    async def run(self, context: Context):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in context.messages:
            if m.text:
                messages.append({"role": m.role, "content": m.text})

        adapter = get_adapter("openai")
        
        for _ in range(3):
            # Stream the OpenAI response — text deltas are forwarded to the
            # A2A client in real time; tool-call deltas are collected.
            stream = await open_client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True,
            )

            full_text = await context.process_stream(stream, adapter)

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

            # Execute each tool and append results
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

            # Submit outputs — resets the run so we can loop back
            await context.submit_tool_outputs(tool_outputs)

        # If loop exceeds limit without final answer
        await context.insert_message(
            role="assistant", 
            content="I'm sorry, I needed too many steps to process this request."
        )

def main():
    # Create an instance of the custom agent
    my_agent = CustomAgent("my-assistant")
    my_agent.name = "Tool-Calling Assistant"
    my_agent.description = "An assistant that can look up the current time using tools."
    my_agent.version = "0.1.0"

    # data store choice
    data_store = InMemoryDataStore()

    # Create a new LLAMPHouse instance with the A2A adapter
    llamphouse = LLAMPHouse(
        agents=[my_agent],
        data_store=data_store,
        adapters=[A2AAdapter()],
    )
    
    # Start the LLAMPHouse server
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
