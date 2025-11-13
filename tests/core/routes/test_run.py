from llamphouse.core import LLAMPHouse, Assistant
from llamphouse.core.context import Context

from openai import OpenAI, NotFoundError
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

def test_create_full_new_run():
    thread = client.beta.threads.create()
    assert thread.id is not None
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="my-assistant",
        additional_instructions="Please assist me with this task.",
        additional_messages=[
            {"role": "user", "content": "This is an additional message."}
        ],
        instructions="You are a helpful assistant.",
        max_completion_tokens=150,
        max_prompt_tokens=1000,
        metadata={"topic": "Test Run", "priority": "high"},
        model="gpt-4",
        parallel_tool_calls=True,
        reasoning_effort="high",
        response_format={ "type": "json_object" },
        stream=False,
        temperature=1.7,
        tool_choice="required",
        tools=[{"code_interpreter": {"enabled": True}}],
        top_p=0.9,
        truncation_strategy={ "type": "auto" },
    )
    assert run.id is not None
    assert run.created_at is not None
    assert run.model == "gpt-4"
    assert run.instructions.startswith("You are a helpful assistant.")
    assert run.metadata["topic"] == "Test Run"
    assert run.metadata["priority"] == "high"
    assert run.temperature == 1.7
    assert run.top_p == 0.9
    assert run.truncation_strategy.type == "auto"    
    assert run.tool_choice == "required"
    assert run.parallel_tool_calls is True
    assert run.response_format.type == "json_object"
    assert run.tools[0].code_interpreter["enabled"] is True
    assert run.max_completion_tokens == 150
    assert run.max_prompt_tokens == 1000
    assert run.reasoning_effort == "high"
    assert run.status == "queued"
    assert run.instructions.endswith("Please assist me with this task.")
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    assert len(messages.data) == 1
    assert messages.data[0].content[0].text == "This is an additional message."
    assert messages.data[0].role == "user"

def test_create_run_invalid_thread():
    try:
        client.beta.threads.retrieve(thread_id="non-existent-thread")
    except NotFoundError as e:
        assert "Thread not found." in str(e)

def test_create_thread_and_run_invalid_assistant():
    try:
        client.beta.threads.create_and_run(
            assistant_id="non-existent-assistant"
        )
    except NotFoundError as e:
        assert "Assistant not found." in str(e)

def test_modify_run_metadata():
    thread = client.beta.threads.create(
        metadata={"topic": "Initial Topic"}
    )
    assert thread.metadata["topic"] == "Initial Topic"
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="my-assistant",
        metadata={"task": "Initial Task"}
    )
    assert run.metadata["task"] == "Initial Task"
    updated_run = client.beta.threads.runs.update(
        thread_id=thread.id,
        run_id=run.id,
        metadata={"task": "Updated Task", "status": "in-progress"}
    )
    assert updated_run.metadata["task"] == "Updated Task"
    assert updated_run.metadata["status"] == "in-progress"