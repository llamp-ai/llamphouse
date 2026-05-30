# Adapters

**Adapters** control how clients communicate with your LLAMPHouse agents. Each adapter exposes a different protocol or interface, allowing the same agents to be accessible from multiple client types simultaneously.

## Available adapters

| Adapter | Default prefix | Protocol / purpose | YAML key |
|---|---|---|---|
| `AssistantAPIAdapter` | _(root)_ | OpenAI Assistants API — drop-in for the `openai` Python SDK | `assistant_api` |
| `A2AAdapter` | _(root)_ | A2A JSON-RPC — interoperable agent-to-agent communication | `a2a` |
| `CompassAdapter` | `/compass` | Full observability UI: threads, runs, traces, charts & dashboards | `compass` |

## Feature comparison

| Feature | `AssistantAPIAdapter` | `A2AAdapter` | `CompassAdapter` |
|---|:---:|:---:|:---:|
| Streaming (SSE) | ✅ | ✅ | — |
| Agent list / get | ✅ | ✅ | ✅ |
| Threads (CRUD) | ✅ | — | read |
| Messages (CRUD) | ✅ | — | read |
| Runs (create / list / get / cancel) | ✅ | ✅ | read |
| Run steps | ✅ | — | read |
| Tool call submission | ✅ | — | — |
| Config store read | — | — | ✅ |
| Config store write | — | — | ✅ |
| Per-run config snapshot | — | — | ✅ |
| Run comparison | — | — | ✅ |
| OTel traces | — | — | ✅ |
| Run flow / timeline | — | — | ✅ |
| Charts (CRUD) | — | — | ✅ |
| Dashboards (CRUD) | — | — | ✅ |

## Configuring adapters

Pass adapters to the `LLAMPHouse` constructor:

```python
from llamphouse import LLAMPHouse, A2AAdapter, AssistantAPIAdapter
from llamphouse.core.adapters import CompassAdapter

app = LLAMPHouse(
    agents=[...],
    adapters=[A2AAdapter(), AssistantAPIAdapter(), CompassAdapter()],
)
```

Or declare them in `llamphouse.yaml`:

```yaml
adapters:
  - assistant_api:
  - a2a:
  - compass:
      prefix: /compass      # optional, this is the default
```

### Default behavior

- If `adapters` is **not specified** (or `None`): `AssistantAPIAdapter` is used by default
- If `adapters` is an **empty list** (`[]`): no protocol adapters are mounted
- The **Compass dashboard** adapter is always auto-mounted unless `compass=False`

## AssistantAPIAdapter

Exposes the [OpenAI Assistants API v2](https://platform.openai.com/docs/api-reference/assistants), allowing any OpenAI SDK client to interact with your agents.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/assistants` | List agents |
| `GET` | `/assistants/{id}` | Retrieve an agent |
| `POST` | `/threads` | Create a thread |
| `GET` | `/threads/{id}` | Retrieve a thread |
| `POST` | `/threads/{id}` | Modify a thread |
| `DELETE` | `/threads/{id}` | Delete a thread |
| `POST` | `/threads/{id}/messages` | Add a message |
| `GET` | `/threads/{id}/messages` | List messages |
| `GET` | `/threads/{id}/messages/{msg_id}` | Get a message |
| `POST` | `/threads/{id}/messages/{msg_id}` | Modify a message |
| `DELETE` | `/threads/{id}/messages/{msg_id}` | Delete a message |
| `POST` | `/threads/{id}/runs` | Create a run |
| `POST` | `/threads/runs` | Create thread + run atomically |
| `GET` | `/threads/{id}/runs` | List runs |
| `GET` | `/threads/{id}/runs/{run_id}` | Retrieve a run |
| `POST` | `/threads/{id}/runs/{run_id}` | Modify a run |
| `POST` | `/threads/{id}/runs/{run_id}/submit_tool_outputs` | Submit tool call outputs |
| `POST` | `/threads/{id}/runs/{run_id}/cancel` | Cancel a run |
| `GET` | `/threads/{id}/runs/{run_id}/steps` | List run steps |
| `GET` | `/threads/{id}/runs/{run_id}/steps/{step_id}` | Get a run step |

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

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/.well-known/agent-card.json` | Primary agent card |
| `GET` | `/agents` | List agents with card URLs |
| `GET` | `/agents/{agent_id}/.well-known/agent-card.json` | Per-agent card |
| `POST` | `/` | JSON-RPC — `message/send`, `message/stream`, `tasks/get`, `tasks/cancel` |

Each agent registered in LLAMPHouse gets its own agent card, enabling discovery and routing in multi-agent ecosystems.

/// details | A2A version note
    type: note

A2A protocol support requires LLAMPHouse **v1.2.0** or later. Earlier versions only support the OpenAI Assistants API adapter.
///

## CompassAdapter

A full observability UI mounted at `/compass` (default). Includes a React SPA and a backing REST API.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Compass SPA |
| `GET` | `/api/overview` | Server-wide overview stats |
| `GET` | `/api/assistants` | List agents |
| `GET/POST` | `/api/assistants/{id}/config` | Read / write agent config |
| `GET` | `/api/threads` | List threads |
| `GET` | `/api/threads/{id}` | Get thread |
| `GET` | `/api/threads/{id}/messages` | List messages |
| `GET` | `/api/threads/{id}/runs` | List runs for thread |
| `GET` | `/api/runs` | List all runs |
| `GET` | `/api/threads/{id}/runs/{run_id}` | Get run |
| `GET` | `/api/threads/{id}/runs/{run_id}/steps` | List run steps |
| `GET` | `/api/threads/{id}/runs/{run_id}/config` | Per-run config snapshot |
| `GET` | `/api/compare` | Compare runs side-by-side |
| `GET` | `/api/traces` | List OTel traces |
| `GET` | `/api/traces/{run_id}` | Traces for a run |
| `GET` | `/api/runs/{run_id}/flow` | Run flow / timeline |
| `GET/POST` | `/api/charts` | List / create charts |
| `GET/PUT/DELETE` | `/api/charts/{id}` | Get / update / delete chart |
| `GET/POST` | `/api/dashboards` | List / create dashboards |
| `GET/PUT/DELETE` | `/api/dashboards/{id}` | Get / update / delete dashboard |
| `POST` | `/api/dashboards/query` | Execute dashboard query |

## Using multiple adapters

You can mount all adapters simultaneously — the same agents are accessible via every enabled protocol:

```python
app = LLAMPHouse(
    agents=[my_agent],
    adapters=[A2AAdapter(), AssistantAPIAdapter(), CompassAdapter()],
    data_store=InMemoryDataStore(),
)
```

This means:
- OpenAI SDK clients connect via the Assistants API
- A2A-compatible agents discover and call your agents via A2A
- The Compass dashboard provides full observability

## Next steps

- [Agents](agents.md) — defining your agent logic
- [Configuration](../configuration.md) — full constructor reference
- [API Compatibility](../api-compatibility.md) — supported OpenAI endpoints
