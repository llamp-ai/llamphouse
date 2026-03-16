"""
Agent Handover — single server (Port 8000).

Both the Receptionist and the Coding Specialist run on the same LLAMPHouse
instance. When the LLM decides a query needs specialist knowledge, the
Receptionist uses `context.handover_to_agent()` to hand control directly
to the Coding Specialist.

`handover_to_agent()` bypasses HTTP entirely — it dispatches the specialist
run through the internal run queue and automatically relays every text
chunk to the client as a MESSAGE_DELTA event.  The Receptionist gives up
control; the specialist streams directly to the client.

The AsyncWorker dispatches each run as a concurrent asyncio task (create_task),
so the dequeue loop is never blocked. While the Receptionist awaits the
Specialist run to complete, the worker immediately picks up the Specialist run.

Start the server:
    python server.py

Then run the client:
    python client.py
"""

import json

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.types.run import ToolOutput
from llamphouse.core.workers.async_worker import AsyncWorker

open_client = AsyncOpenAI()


# ── Coding Specialist ─────────────────────────────────────────────────────────

SPECIALIST_SYSTEM_PROMPT = (
    "You are an expert software engineer and coding assistant. "
    "You specialise in writing clean, efficient code and explaining programming "
    "concepts clearly. Provide well-structured answers with code examples where appropriate."
)


class CodingSpecialistAgent(Agent):
    async def run(self, context: Context):
        messages = [{"role": "system", "content": SPECIALIST_SYSTEM_PROMPT}]
        for msg in context.messages:
            if msg.text:
                messages.append({"role": msg.role, "content": msg.text})

        stream = await open_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )

        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)
        if full_text and full_text.strip():
            await context.insert_message(full_text)


# ── Receptionist ──────────────────────────────────────────────────────────────

RECEPTIONIST_SYSTEM_PROMPT = (
    "You are a helpful receptionist assistant. "
    "You answer general knowledge questions directly. "
    "Whenever the user asks about coding, programming, algorithms, debugging, "
    "or software engineering, call the `handover_to_specialist` tool instead of "
    "answering yourself — the specialist will handle it."
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "handover_to_specialist",
            "description": (
                "Hand the user's query over to the Coding Specialist. "
                "Use this for any question about code, programming languages, "
                "algorithms, debugging, software design, or engineering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact question to forward to the specialist.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]



class ReceptionistAgent(Agent):
    """
    Routes user queries:
      - General questions → answered directly (and streamed) by the LLM
      - Coding questions  → specialist's stream relayed in real-time via A2A
    """

    async def run(self, context: Context):
        messages = [{"role": "system", "content": RECEPTIONIST_SYSTEM_PROMPT}]
        for m in context.messages:
            if m.text:
                messages.append({"role": m.role, "content": m.text})

        adapter = get_adapter("openai")
        max_rounds = 3

        for _ in range(max_rounds):
            stream = await open_client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True,
            )

            full_text = await context.process_stream(stream, adapter)

            if full_text and full_text.strip():
                # Receptionist answered directly — no handover.
                await context.insert_message(full_text)
                return

            if not context.pending_tool_calls:
                await context.insert_message("I'm sorry, I couldn't process that.")
                return

            # Check for handover — hand control to the specialist.
            handover = next(
                (tc for tc in context.pending_tool_calls if tc["name"] == "handover_to_specialist"),
                None,
            )
            if handover:
                try:
                    args = json.loads(handover["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}

                query = args.get("query", "")
                if not query and context.messages:
                    last = context.messages[-1]
                    if last.text:
                        query = last.text

                # handover_to_agent streams the specialist's response
                # directly to the client and returns the full text.
                specialist_text = await context.handover_to_agent(
                    "coding-specialist", query,
                )

                await context.submit_tool_outputs([
                    ToolOutput(tool_call_id=handover["id"], output=specialist_text)
                ])
                return

            # Normal (non-handover) tool calls — loop continues for LLM relay.
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

            tool_outputs: list[ToolOutput] = []
            for tc in context.pending_tool_calls:
                result = json.dumps({"error": f"Unknown tool: {tc['name']}"})
                tool_outputs.append(ToolOutput(tool_call_id=tc["id"], output=result))
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

            await context.submit_tool_outputs(tool_outputs)

        await context.insert_message(
            "I'm sorry, I needed too many steps to process this request."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    receptionist = ReceptionistAgent(
        id="receptionist",
        name="Receptionist",
        description=(
            "Routes questions to the right agent — answers general queries directly "
            "and hands coding questions over to the Coding Specialist."
        ),
        tools=TOOL_SCHEMAS,
    )

    specialist = CodingSpecialistAgent(
        id="coding-specialist",
        name="Coding Specialist",
        description="An expert coding assistant that answers programming questions.",
    )

    llamphouse = LLAMPHouse(
        agents=[receptionist, specialist],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
        worker=AsyncWorker(time_out=120.0),
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
