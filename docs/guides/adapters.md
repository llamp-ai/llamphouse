# Adapters

Adapters put other protocols in front of the same agents, next to the native API (`/v2`): the OpenAI Responses
and Conversations APIs, the OpenAI Assistants API, A2A v1.0 and AG-UI. Each one maps its protocol onto threads,
runs and messages. Agent code doesn't change: an agent doesn't know which protocol a run came from, and the same
thread can be read in Compass and over `/v2` whichever protocol made it.

Adapters are listed in the runtime file:

```yaml
adapters:
  - responses: {prefix: /v1, agent: desk}      # OpenAI Responses + Conversations
  - assistants: {prefix: /v1, agent: desk}     # OpenAI Assistants
  - a2a: {prefix: /a2a}                        # A2A v1.0, an agent card per agent
  - ag_ui: {prefix: /ag-ui}                    # AG-UI (pip install 'llamphouse[ag-ui]')
  - my_pkg.adapters:MyAdapter: {prefix: /x}    # your own
```

Every adapter takes:

| Option | |
|---|---|
| `prefix` | where its routes are mounted |
| `agents` | `"*"` (default) or a list of agent names it exposes |

The same kind can be mounted several times with different prefixes and agents (a public `/v1` for one agent, an
internal one for all). Every adapter is behind the deployment's [auth and access](auth.md): the same callers, the
same tenants, `404` for what a caller may not see.

What every protocol has in common, and how each maps it:

| llamphouse | Responses | Assistants | A2A | AG-UI |
|---|---|---|---|---|
| agent | model | assistant | agent (card) | agent |
| thread | conversation | thread | context | thread (`threadId`) |
| run | one or more responses | run | task | one or more AG-UI runs |
| `ctx.ask` | the question as the output, answered by the next response | `requires_action` / `wait_for_input`, answered on `/signals` | `TASK_STATE_INPUT_REQUIRED`, answered with the task id | an `interrupt` outcome, answered by `resume` |
| client tools (`ctx.client`) | function `tools`, `function_call` / `function_call_output` | function `tools`, `submit_tool_outputs` | (none) | frontend `tools`, tool messages |

## OpenAI Responses and Conversations

`responses` (options: `prefix` `/v1`, `agents`, `agent`: the agent a conversation is created for; default the
first exposed one).

| | |
|---|---|
| `GET {prefix}/models` | the exposed agents, as models |
| `POST {prefix}/responses` | a response; `stream: true` for SSE, `background: true` to return at once and poll |
| `GET {prefix}/responses/{id}` | a response |
| `POST {prefix}/responses/{id}/cancel` | |
| `POST {prefix}/conversations` | a conversation (with `items`, `metadata`) |
| `GET`, `POST`, `DELETE {prefix}/conversations/{id}` | read, change its metadata, close it |
| `GET`, `POST {prefix}/conversations/{id}/items` | its items; add some (refused while a response runs) |
| `GET {prefix}/conversations/{id}/items/{item_id}` | |

`model` is the agent's name. A response continues the thread of its `conversation` or of its
`previous_response_id`, or starts a new thread without either. Its `input` goes on the thread; the last user
message is the run's input.

A run is not always one response. When the run calls a client tool or asks a question, the response completes
with `function_call` items or the question as an assistant message. The next response (with
`previous_response_id`, or the same `conversation`) answers it with `function_call_output` items or a user
message, and **the same run carries on**, as a new response with its own id.

`instructions` reach the agent as `ctx.run.request.metadata["responses.instructions"]`, and the request's
`metadata` as `ctx.run.request.metadata["responses.metadata"]`: the agent decides what to do with them.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<your API key, or anything without auth>")

first = client.responses.create(model="desk", input="Compare Bangkok and Lisbon")
print(first.output_text)                                   # "For which month?"
done = client.responses.create(model="desk", input="March", previous_response_id=first.id)

stream = client.responses.create(model="desk", input="And Rome?", previous_response_id=done.id, stream=True)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

## OpenAI Assistants

`assistants` (options: `prefix` `/v1`, `agents`, `agent`), for clients built on the Assistants API, including
1.x llamphouse clients.

| | |
|---|---|
| `GET {prefix}/assistants[/{id}]` | the exposed agents |
| `POST {prefix}/threads` | a thread (with `messages`) |
| `GET`, `POST`, `DELETE {prefix}/threads/{id}` | read, change metadata, close |
| `POST`, `GET {prefix}/threads/{id}/messages[/{message_id}]` | append (refused while a run is active), list |
| `POST {prefix}/threads/{id}/runs` | a run (`stream: true` for SSE) |
| `POST {prefix}/threads/runs` | a thread and a run |
| `GET {prefix}/threads/{id}/runs[/{run_id}]` | runs |
| `POST {prefix}/threads/{id}/runs/{run_id}/submit_tool_outputs` | the client's tool results |
| `POST {prefix}/threads/{id}/runs/{run_id}/signals` | `{"name": "input", "payload": ...}` answers the run's question; any other name is a signal |
| `POST {prefix}/threads/{id}/runs/{run_id}/cancel` | |
| `GET {prefix}/threads/{id}/runs/{run_id}/steps` | always empty: a 2.0 run records steps differently |

An assistant is an agent (`assistant_id` is its name). A thread is created for the adapter's `agent` and goes to
the agent of its first run. A run takes the thread's last message as its input (or the last of
`additional_messages`); function `tools` are the run's client tools. A run that waits shows as
`requires_action`: `submit_tool_outputs` for a client tool call, `wait_for_input` for a question,
`wait_for_signal` / `wait_until` while it waits on a signal or the clock.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
thread = client.beta.threads.create(messages=[{"role": "user", "content": "hi there"}])
run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id="desk")
print(client.beta.threads.messages.list(thread_id=thread.id).data[0].content[0].text.value)
```

## A2A v1.0

`a2a` (options: `prefix` `/a2a`, `agents`, `provider` (`{organization, url}` for the cards),
`documentation_url`, `icon_url`). Every exposed agent is an A2A agent, over the JSON-RPC binding.

| | |
|---|---|
| `GET {prefix}/{agent}/.well-known/agent-card.json` | its agent card |
| `POST {prefix}/{agent}` | JSON-RPC: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, and `Create` / `Get` / `List` / `DeleteTaskPushNotificationConfig` |
| `GET /.well-known/agent-card.json` | the card, when the adapter exposes exactly one agent |

A context is a thread; a task is a run.

- A message without a `contextId` starts a new thread; with one, a run on that thread (an error while a task is
  active there: send to it with its `taskId`).
- A message with a `taskId` goes to that run: it answers the run's question (input-required, auth-required), or
  reaches its inbox while it works. A task that ended takes no messages.
- `SendMessage` waits until the task ends or needs input, unless `configuration.returnImmediately` is set.
- The task's artifacts are the `ArtifactPart`s its run wrote; its history, the messages to and from the run;
  `status.message`, what the agent asked or said last.
- Push notification configs are [subscriptions](subscriptions.md) to the task: the workers POST a
  `statusUpdate` on every status change, with the config's token in `X-A2A-Notification-Token` and its
  authentication in `Authorization`.
- The card lists the deployment's authenticators as security schemes. Its skills and modes come from the agent
  file's `publish.a2a` (`skills`, `input_modes`, `output_modes`, `extensions`), else one skill from the agent's
  description. An extension listed there with `required: true` must be activated in the request's
  `A2A-Extensions` header, else the message is refused (`-32008`); the activated ones reach the agent as
  `ctx.run.request.extensions`.
- Requests should send `A2A-Version: 1.0` (an empty version is taken as 1.0; another is refused with `-32009`).
- Parts A2A doesn't have travel as data with the media type `application/vnd.llamphouse.<kind>+json`; a
  llamphouse client on the other side turns them back into the part. The extended card is not supported.

```bash
curl -s localhost:8000/a2a/desk -H 'content-type: application/json' -H 'A2A-Version: 1.0' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
  "params": {"message": {"messageId": "m-1", "role": "ROLE_USER", "parts": [{"text": "Compare Bangkok and Lisbon"}]}}}'
```

## AG-UI

`ag_ui` (options: `prefix` `/ag-ui`, `agents`, `agent`; needs `pip install 'llamphouse[ag-ui]'`). A
`RunAgentInput` goes in, the run's events come out as SSE.

| | |
|---|---|
| `POST {prefix}` | the adapter's `agent` (the first exposed one by default) |
| `POST {prefix}/agents/{agent}` | an agent |

- The AG-UI `threadId` names a llamphouse thread (kept as its `ag_ui.thread_id` label), made on first use, for
  that agent. Each request sends the whole conversation: the user messages the thread doesn't have yet (by their
  AG-UI id) are added, and the last one is the run's input.
- `tools` are the run's client tools: a call ends the AG-UI run with its `TOOL_CALL_*` events; the next request's
  tool message (its `toolCallId`) answers it, and the same run carries on.
- A question (`ctx.ask`) or a wait for a signal ends the AG-UI run with an `interrupt` outcome (the question as
  `message`, its schema as `responseSchema`); the next request's `resume` entries
  (`{"interruptId", "status": "resolved", "payload"}`) answer it, or cancel the run. A sleep keeps the stream open.
- Text streams as `TEXT_MESSAGE_*` events; steps as `STEP_STARTED` / `STEP_FINISHED`; what the agent emits as
  `StateSnapshot` / `StateDelta` as `STATE_*` events; `Custom` and `Progress` as `CUSTOM` events.

The input's `state`, `context`, `tools` and `forwardedProps` reach the agent through `ag_ui_input(ctx)`:

```python
from llamphouse.runtime.adapters.ag_ui import ag_ui_input

shared = ag_ui_input(ctx).get("state") or {}   # also "context", "tools", "forwarded_props", "run_id"
```

## Your own adapter

An adapter is a class with `mount(app, gateway)`, which adds routes to the FastAPI app. Subclass
`llamphouse.runtime.adapters.Adapter` (its constructor takes `prefix` and `agents`) and use the gateway for what
every protocol needs: `gateway.authenticate` (a FastAPI dependency), `gateway.known(id, "thread" | "run",
"read" | "write")` (exists and this caller may touch it, else 404), `gateway.message(...)` / `gateway.sender()`
(messages written as the caller), `gateway.wait(run_id)` and `gateway.engine` for the primitives. Name it in the
runtime file as `{my_pkg.adapters:MyAdapter: {...}}`.

An adapter translates; it decides nothing the engine doesn't offer as a primitive. What a message means while a
run is busy is the protocol's convention, written in its adapter.

## See also

- [Example 16_Adapters](../../examples/16_Adapters): one agent behind every protocol
- [Example 17_AGUI](../../examples/17_AGUI): a web frontend over AG-UI
- [Tool calls](tool-calls.md): client tools
