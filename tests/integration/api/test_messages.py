import pytest
from openai import NotFoundError

def _create_thread(client):
    return client.beta.threads.create()

def test_create_message(client):
    """Creates a new message in a thread and validates basic fields."""
    thread = _create_thread(client)
    msg = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Hello"
    )
    assert msg.id is not None
    assert msg.role == "user"
    assert msg.content[0].text == "Hello"
    assert msg.thread_id == thread.id

def test_list_messages_order(client):
    """Lists messages in ascending order and verifies ordering matches insertion."""
    thread = _create_thread(client)
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content="one")
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content="two")
    page = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    assert len(page.data) >= 2
    assert page.data[0].content[0].text == "one"
    assert page.data[1].content[0].text == "two"

def test_retrieve_message(client):
    """Retrieves a message by id within its thread."""
    thread = _create_thread(client)
    msg = client.beta.threads.messages.create(thread_id=thread.id, role="user", content="ping")
    fetched = client.beta.threads.messages.retrieve(thread_id=thread.id, message_id=msg.id)
    assert fetched.id == msg.id
    assert fetched.content[0].text == "ping"

def test_update_message_metadata(client):
    """Updates message metadata and verifies persisted changes."""
    thread = _create_thread(client)
    msg = client.beta.threads.messages.create(thread_id=thread.id, role="user", content="meta")
    updated = client.beta.threads.messages.update(
        thread_id=thread.id,
        message_id=msg.id,
        metadata={"k": "v"}
    )
    assert updated.metadata["k"] == "v"

def test_delete_message(client):
    """Deletes a message and confirms it is no longer retrievable."""
    thread = _create_thread(client)
    msg = client.beta.threads.messages.create(thread_id=thread.id, role="user", content="bye")
    deleted = client.beta.threads.messages.delete(thread_id=thread.id, message_id=msg.id)
    assert deleted.id == msg.id
    assert deleted.deleted is True
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.messages.retrieve(thread_id=thread.id, message_id=msg.id)
    assert exc.value.status_code == 404

def test_create_message_thread_not_found(client):
    """Returns 404 when creating a message for a missing thread."""
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.messages.create(
            thread_id="thread_does_not_exist",
            role="user",
            content="x",
        )
    assert exc.value.status_code == 404