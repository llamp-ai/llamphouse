import socket
import threading
import time
import pytest

from openai import OpenAI

from llamphouse.core import LLAMPHouse, Agent, Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.workers.async_worker import AsyncWorker

pytestmark = [pytest.mark.integration, pytest.mark.streaming]

ASSISTANT_ID = "chunking-assistant"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Server {host}:{port} did not start within {timeout}s")


class ChunkingAgent(Agent):
    """Agent that streams two chunks via send_chunk."""

    async def run(self, context: Context):
        context.send_chunk("hello ")
        context.send_chunk("world")


@pytest.fixture(scope="module")
def openai_client():
    port = _free_port()
    app = LLAMPHouse(
        agents=[ChunkingAgent(ASSISTANT_ID)],
        authenticator=None,
        worker=AsyncWorker(time_out=5.0),
        event_queue_class=InMemoryEventQueue,
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
    )

    thread = threading.Thread(target=lambda: app.ignite(host="127.0.0.1", port=port), daemon=True)
    thread.start()
    _wait_for_server("127.0.0.1", port)

    return OpenAI(base_url=f"http://127.0.0.1:{port}", api_key="test")


def test_openai_sdk_can_stream_and_assemble_message(openai_client):
    """The OpenAI SDK must be able to stream a run and get the final assembled message."""
    thread = openai_client.beta.threads.create()

    with openai_client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    ) as stream:
        messages = stream.get_final_messages()

    assert messages, "No final messages returned by SDK stream"
    text = messages[0].content[0].text.value
    assert text == "hello world", f"Unexpected text: {text!r}"


def test_openai_sdk_streams_text_deltas(openai_client):
    """The OpenAI SDK must receive individual text delta events."""
    thread = openai_client.beta.threads.create()
    chunks = []

    with openai_client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    ) as stream:
        for delta in stream.text_deltas:
            chunks.append(delta)

    assert "".join(chunks) == "hello world", f"Unexpected chunks: {chunks}"
