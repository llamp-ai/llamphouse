import pytest

from openai import NotFoundError

def test_list_assistants_default(client, assistant_ids):
    """Lists assistants in ascending order and verifies the default ordering."""
    page = client.beta.assistants.list(order="asc")
    ids = [a.id for a in page.data]
    assert ids[: len(assistant_ids)] == assistant_ids

def test_list_assistant_limit(client, assistant_ids):
    """Respects the limit parameter when listing assistants."""
    page = client.beta.assistants.list(order="asc", limit=2)
    assert len(page.data) == 2
    assert [a.id for a in page.data] == assistant_ids[:2]

def test_list_assistants_after(client, assistant_ids):
    """Supports pagination by returning assistants after a given id."""
    page = client.beta.assistants.list(order="asc", after=assistant_ids[0])
    ids = [a.id for a in page.data]
    assert ids == assistant_ids[1:]

def test_list_assistants_before(client, assistant_ids):
    """Supports pagination by returning assistants before a given id."""
    page = client.beta.assistants.list(order="asc", before=assistant_ids[-1])
    ids = [a.id for a in page.data]
    assert ids == assistant_ids[:-1]

def test_retrieve_assistant(client, assistant_ids):
    """Retrieves a single assistant by id."""
    fetched = client.beta.assistants.retrieve(assistant_id=assistant_ids[1])
    assert fetched.id == assistant_ids[1]

def test_retrieve_assistant_not_found(client):
    """Returns 404 when retrieving a missing assistant id."""
    with pytest.raises(NotFoundError) as exc:
        client.beta.assistants.retrieve(assistant_id="assistant_missing")
    assert exc.value.status_code == 404