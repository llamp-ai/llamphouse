import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Assistant
from llamphouse.core.context import Context
from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.streaming.event_queue.janus_event_queue import JanusEventQueue
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.streaming.stream_events import TextDelta, ToolCallDelta, StreamFinished, StreamError

import logging
logging.basicConfig(level=logging.INFO)

open_client = OpenAI()

class CustomAssistant(Assistant):
    async def run(self, context: Context):
        messages = [{"role": message.role, "content": message.content[0].text} for message in context.messages]

        stream = open_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        def on_event(evt):
            if isinstance(evt, TextDelta):
                print(evt.text, end="", flush=True)
            elif isinstance(evt, ToolCallDelta):
                print(f"\n[tool_delta] name={evt.name} args+= {evt.arguments_delta!r}", flush=True)
            elif isinstance(evt, StreamError):
                print(f"\n[stream_error] {evt.code}: {evt.message}", flush=True)
            elif isinstance(evt, StreamFinished):
                print(f"\n[finished] reason={evt.reason}", flush=True)

        adapter = get_adapter("openai")

        full_text = await asyncio.to_thread(context.handle_completion_stream, stream, adapter, on_event)
        if full_text and full_text.strip():
            await context.insert_message(full_text)


def main():
    # Create an instance of the custom assistant
    my_assistant = CustomAssistant("my-assistant")

    # data store choice
    data_store = InMemoryDataStore() # PostgresDataStore() or InMemoryDataStore() for in-memory testing

    # event queue choice
    event_queue_class = InMemoryEventQueue # InMemoryEventQueue or JanusEventQueue for async support

    # Create a new LLAMPHouse instance
    llamphouse = LLAMPHouse(assistants=[my_assistant], data_store=data_store, event_queue_class=event_queue_class)
    
    # Start the LLAMPHouse server
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()