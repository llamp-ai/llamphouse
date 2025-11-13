from llamphouse.core import LLAMPHouse, Assistant
from llamphouse.core.context import Context

from openai import OpenAI, NotFoundError, BadRequestError
import threading

class CustomAssistant(Assistant):
    def run(self, _: Context):
        pass

my_assistant = CustomAssistant("my-assistant")
llamphouse = LLAMPHouse(assistants=[my_assistant])

# Start the server in a separate thread
def start_server():
    llamphouse.ignite(host="127.0.0.1", port=8085)

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

client = OpenAI(base_url="http://127.0.0.1:8085", api_key="your_api_key")

def test_create_new_thread():
    thread = client.beta.threads.create()
    assert thread.id is not None
    assert thread.created_at is not None

def test_create_full_new_thread():
    thread = client.beta.threads.create(
        tool_resources={"code_interpreter": {"enabled": True}},
        metadata={"topic": "Test Thread", "priority": "high"}
    )
    assert thread.id is not None
    assert thread.created_at is not None
    assert thread.tool_resources.code_interpreter.enabled is True
    assert thread.metadata["topic"] == "Test Thread"
    assert thread.metadata["priority"] == "high"

def test_create_thread_with_id():
    thread = client.beta.threads.create(metadata={"thread_id": "custom-thread-id"})
    assert thread.id == "custom-thread-id"

def test_create_thread_with_same_id():
    thread1 = client.beta.threads.create(metadata={"thread_id": "duplicate-thread-id"})
    assert thread1.id == "duplicate-thread-id"
    try:
        client.beta.threads.create(metadata={"thread_id": "duplicate-thread-id"})
    except BadRequestError as e:
        assert e.status_code == 400
        assert "Thread with the same ID already exists." in str(e)

def test_create_thread_with_messages():
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]
    thread = client.beta.threads.create(messages=messages)
    assert thread.id is not None
    retrieved_messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    assert len(retrieved_messages.data) == 2
    assert retrieved_messages.data[0].content[0].text == "Hello!"
    assert retrieved_messages.data[0].role == "user"
    assert retrieved_messages.data[1].content[0].text == "Hi there! How can I assist you today?"
    assert retrieved_messages.data[1].role == "assistant"

def test_modify_thread_metadata():
    thread = client.beta.threads.create(
        metadata={"topic": "Initial Topic"}
    )
    assert thread.metadata["topic"] == "Initial Topic"
    updated_thread = client.beta.threads.update(
        thread_id=thread.id,
        metadata={"topic": "New Topic", "priority": "high"}
    )
    assert updated_thread.metadata["topic"] == "New Topic"
    assert updated_thread.metadata["priority"] == "high"

def test_modify_thread_tool_resources():
    thread = client.beta.threads.create()
    updated_thread = client.beta.threads.update(
        thread_id=thread.id,
        tool_resources={"code_interpreter": {"enabled": True}}
    )
    assert updated_thread.tool_resources.code_interpreter.enabled is True