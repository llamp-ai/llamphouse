from openai import NotFoundError, BadRequestError

import pytest

def test_create_new_thread(client):
    """Creates an empty thread and verifies id/created_at are returned."""
    thread = client.beta.threads.create()
    assert thread.id is not None
    assert thread.created_at is not None

def test_create_full_new_thread(client):
    """Creates a thread with tool_resources and metadata and verifies both persist."""
    thread = client.beta.threads.create(
        tool_resources={"code_interpreter": {"enabled": True}},
        metadata={"topic": "Test Thread", "priority": "high"}
    )
    assert thread.id is not None
    assert thread.created_at is not None
    assert thread.tool_resources.code_interpreter.enabled is True
    assert thread.metadata["topic"] == "Test Thread"
    assert thread.metadata["priority"] == "high"

def test_create_thread_with_id(client):
    """Creates a thread with a custom thread_id in metadata and asserts it is used."""
    thread = client.beta.threads.create(metadata={"thread_id": "custom-thread-id"})
    assert thread.id == "custom-thread-id"

def test_create_thread_with_same_id(client):
    """Rejects a duplicate thread_id with a 400 error."""
    thread1 = client.beta.threads.create(metadata={"thread_id": "duplicate-thread-id"})
    assert thread1.id == "duplicate-thread-id"
    with pytest.raises(BadRequestError) as exc:
        client.beta.threads.create(metadata={"thread_id": "duplicate-thread-id"})
    assert exc.value.status_code == 400
    assert "Thread with the same ID already exists." in str(exc.value)

def test_create_thread_with_messages(client):
    """Creates a thread with initial messages and verifies ordering/content/roles."""
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

def test_modify_thread_metadata(client):
    """Updates thread metadata and confirms changes are applied."""
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

def test_modify_thread_tool_resources(client):
    """Updates tool_resources and verifies code_interpreter is enabled."""
    thread = client.beta.threads.create()
    updated_thread = client.beta.threads.update(
        thread_id=thread.id,
        tool_resources={"code_interpreter": {"enabled": True}}
    )
    assert updated_thread.tool_resources.code_interpreter.enabled is True

def test_get_thread_by_id(client):
    """Retrieves a thread by id and confirms metadata matches."""
    thread = client.beta.threads.create(metadata={"topic": "Fetch"})
    fetched = client.beta.threads.retrieve(thread_id=thread.id)
    assert fetched.id == thread.id
    assert fetched.metadata["topic"] == "Fetch"

def test_get_thread_not_found(client):
    """Returns 404 when retrieving a missing thread id."""
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.retrieve(thread_id="thread_does_not_exist")
    assert exc.value.status_code == 404

def test_delete_thread(client):
    """Deletes a thread and verifies it can no longer be retrieved."""
    thread = client.beta.threads.create()
    deleted = client.beta.threads.delete(thread_id=thread.id)
    assert deleted.id == thread.id
    assert deleted.deleted is True
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.retrieve(thread_id=thread.id)
    assert exc.value.status_code == 404