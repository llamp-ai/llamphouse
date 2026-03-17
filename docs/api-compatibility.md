# API Compatibility

LLAMPHouse implements the [OpenAI Assistants API v2](https://platform.openai.com/docs/api-reference/assistants). This means you can use the standard `openai` Python SDK (or any HTTP client) to interact with your agents.

## Supported endpoints

### Assistants

| Endpoint | Method | Status |
|---|---|---|
| List assistants | `GET /assistants` | ✅ Supported |
| Retrieve assistant | `GET /assistants/{id}` | ✅ Supported |
| Create assistant | `POST /assistants` | _By design: agents are defined in code_ |
| Modify assistant | `POST /assistants/{id}` | _By design: agents are defined in code_ |
| Delete assistant | `DELETE /assistants/{id}` | _By design: agents are defined in code_ |

/// details | Why no Create/Modify/Delete?
    type: note

In LLAMPHouse, agents are defined as Python classes in your server code — not created dynamically via API. This is by design: your agent logic, tools, and configuration live in version-controlled code. The List and Retrieve endpoints return information about your registered agents.
///

### Threads

| Endpoint | Method | Status |
|---|---|---|
| Create thread | `POST /threads` | ✅ Supported |
| Retrieve thread | `GET /threads/{id}` | ✅ Supported |
| Modify thread | `POST /threads/{id}` | ✅ Supported |
| Delete thread | `DELETE /threads/{id}` | ✅ Supported |

### Messages

| Endpoint | Method | Status |
|---|---|---|
| Create message | `POST /threads/{id}/messages` | ✅ Supported |
| List messages | `GET /threads/{id}/messages` | ✅ Supported |
| Retrieve message | `GET /threads/{id}/messages/{id}` | ✅ Supported |
| Modify message | `POST /threads/{id}/messages/{id}` | ✅ Supported |
| Delete message | `DELETE /threads/{id}/messages/{id}` | ✅ Supported |

### Runs

| Endpoint | Method | Status |
|---|---|---|
| Create run | `POST /threads/{id}/runs` | ✅ Supported |
| Create thread and run | `POST /threads/runs` | ✅ Supported |
| List runs | `GET /threads/{id}/runs` | ✅ Supported |
| Retrieve run | `GET /threads/{id}/runs/{id}` | ✅ Supported |
| Modify run | `POST /threads/{id}/runs/{id}` | ✅ Supported |
| Cancel run | `POST /threads/{id}/runs/{id}/cancel` | ✅ Supported |
| Submit tool outputs | `POST /threads/{id}/runs/{id}/submit_tool_outputs` | ✅ Supported |

### Run Steps

| Endpoint | Method | Status |
|---|---|---|
| List run steps | `GET /threads/{id}/runs/{id}/steps` | ✅ Supported |
| Retrieve run step | `GET /threads/{id}/runs/{id}/steps/{id}` | ✅ Supported |

### Streaming

| Feature | Status |
|---|---|
| Message delta events | ✅ Supported |
| Run step events | ✅ Supported |
| Assistant stream events | ✅ Supported |

### Not yet implemented

| Feature | Status |
|---|---|
| Vector Stores | Not yet implemented |
| File Search | Not yet implemented |
| Code Interpreter | Not yet implemented |

## A2A Protocol

In addition to the OpenAI Assistants API, LLAMPHouse supports the [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/):

| Endpoint | Description |
|---|---|
| `GET /.well-known/agent.json` | Agent card discovery |
| `POST /` | A2A JSON-RPC endpoint |

/// details | A2A version note
    type: note

A2A protocol support requires LLAMPHouse **v1.2.0** or later.
///

## Using the OpenAI SDK

```python
from openai import OpenAI

# Point the SDK at your LLAMPHouse server
client = OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="any",  # or your actual key if auth is enabled
)

# Standard Assistants API usage
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Hello!",
)
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id="my-agent",
)
```

## Next steps

- [Adapters](concepts/adapters.md) — configure protocol adapters
- [Configuration](configuration.md) — authentication and other settings
- [Examples](examples.md) — see the API in action
