# Adapters

**Adapters** control how clients communicate with your LLAMPHouse agents. Each adapter exposes a different protocol or interface, allowing the same agents to be accessible from multiple client types simultaneously.

## Available adapters

| Adapter | Protocol | Import | Use case |
|---|---|---|---|
| `AssistantAPIAdapter` | OpenAI Assistants API | `from llamphouse.core.adapters.assistant_api import AssistantAPIAdapter` | OpenAI SDK compatibility |
| `A2AAdapter` | A2A (Agent-to-Agent) | `from llamphouse.core.adapters.a2a import A2AAdapter` | Interoperable agent communication |
| `CompassAdapter` | Compass Dashboard | _auto-mounted_ | Built-in dev UI |

## Configuring adapters

Pass adapters to the `LLAMPHouse` constructor:

```python
from llamphouse.core import LLAMPHouse
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.adapters.assistant_api import AssistantAPIAdapter

app = LLAMPHouse(
    agents=[...],
    adapters=[A2AAdapter(), AssistantAPIAdapter()],
)
```

### Default behavior

- If `adapters` is **not specified** (or `None`): `AssistantAPIAdapter` is used by default
- If `adapters` is an **empty list** (`[]`): no protocol adapters are mounted
- The **Compass dashboard** adapter is always auto-mounted unless `compass=False`

## AssistantAPIAdapter

Exposes the [OpenAI Assistants API v2](https://platform.openai.com/docs/api-reference/assistants), allowing any OpenAI SDK client to interact with your agents.

**Endpoints:**

- `POST /threads` — create a thread
- `GET /threads/{id}` — retrieve a thread
- `POST /threads/{id}/messages` — add a message
- `GET /threads/{id}/messages` — list messages
- `POST /threads/{id}/runs` — create a run
- `GET /threads/{id}/runs` — list runs
- `GET /threads/{id}/runs/{id}` — retrieve a run
- `GET /threads/{id}/runs/{id}/steps` — list run steps
- `GET /assistants` — list agents
- `GET /assistants/{id}` — retrieve an agent

**Client example:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000", api_key="any")

thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content="Hello!"
)
run = client.beta.threads.runs.create(
    thread_id=thread.id, assistant_id="my-agent"
)
```

## A2AAdapter

Exposes the [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/) — Google's standard for interoperable agent communication.

**Endpoints:**

- `GET /.well-known/agent.json` — agent card discovery (per-agent cards)
- `POST /` — A2A JSON-RPC endpoint

Each agent registered in LLAMPHouse gets its own agent card, enabling discovery and routing in multi-agent ecosystems.

/// details | A2A version note
    type: note

A2A protocol support requires LLAMPHouse **v1.2.0** or later. Earlier versions only support the OpenAI Assistants API adapter.
///

## Using both adapters

You can mount both adapters simultaneously — the same agents are accessible via both protocols:

```python
app = LLAMPHouse(
    agents=[my_agent],
    adapters=[A2AAdapter(), AssistantAPIAdapter()],
    data_store=InMemoryDataStore(),
)
```

This means:
- OpenAI SDK clients connect via the Assistants API
- A2A-compatible agents discover and call your agents via A2A
- The Compass dashboard works with either protocol

## Next steps

- [Agents](agents.md) — defining your agent logic
- [Configuration](../configuration.md) — full constructor reference
- [API Compatibility](../api-compatibility.md) — supported OpenAI endpoints
