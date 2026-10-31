# Adapters

One agent served over several protocols at once: the native API (`/v2`), the OpenAI **Responses** and
**Conversations** APIs and the **Assistants** API (any OpenAI client, pointed at the server), and **A2A v1.0**
(JSON-RPC). The agent knows nothing about them. It asks a question, keeps thread state, calls a tool the client
declared and streams its reply; each adapter maps that onto its protocol: a conversation, an Assistants thread
or an A2A context is a thread, and a response, a run or a task is a run. No model needed.

## Run it

```bash
pip install -r requirements.txt                              # openai, for the client
llamphouse up --agents agents.yaml --config llamphouse.yaml  # http://127.0.0.1:8000/compass
python client.py                                             # or: python client.py responses | assistants | a2a
```

The runtime file `llamphouse.yaml` names only the adapters, so everything else is in memory. (`--memory`
would change nothing here: it replaces a runtime file's stores and queues, and keeps its adapters.)

```
== Responses API: a conversation is a thread, a response a stretch of a run
models: ['concierge']
> Hi!
Hello! What's your name?
> Ada
Ada, you said: 'Hi!'. Turn 1 of this conversation.
...
> What time is it?
[the run calls the client's tool: local_time({})]
Grace, you said: 'What time is it?'. Turn 2 of this conversation. It's 18:01 where you are.

== Assistants API: an assistant is an agent; a question is requires_action
> Good morning
[requires_action: wait_for_input]
> Alan
Alan, you said: 'Good morning'. Turn 1 of this conversation.

== A2A v1.0: a context is a thread, a task a run
card: Concierge - Asks your name once per conversation, then answers - with the local time when your client offers it.
> Hi there
Hello! What's your name?
[task run_97a1…: TASK_STATE_INPUT_REQUIRED]
> Lin
Lin, you said: 'Hi there'. Turn 1 of this conversation.
[task run_97a1…: TASK_STATE_COMPLETED]
```

## How each protocol sees the agent

| the agent does | Responses | Assistants | A2A |
|---|---|---|---|
| `ctx.ask(...)` | the response completes with the question; the next response (`previous_response_id` or the same `conversation`) is the answer, and the same run carries on | `requires_action` / `wait_for_input`; answered with `POST .../runs/{id}/signals {"name": "input", "payload": ...}` | `TASK_STATE_INPUT_REQUIRED`; answered by a message with the task's `taskId` |
| `ctx.client.call_tool(...)` | `function_call` items, for the function `tools` the request declared; answered with `function_call_output` | `requires_action` / `submit_tool_outputs` | - |
| `ctx.stream()` | `stream=True`: `response.output_text.delta` | `stream=True`: `thread.message.delta` | `SendStreamingMessage` |
| `ctx.state` | kept on the thread: the next response in the conversation knows your name | the same | the same, across tasks of one `contextId` |

The agent's name is the `model` (Responses), the `assistant_id` (Assistants), and the path
`/a2a/<agent>` with its card at `/a2a/<agent>/.well-known/agent-card.json` (A2A).

## In Compass

Every conversation from the client is a thread of `concierge`: open one to see the question asked and answered
in the run's story, the `name` and `turns` keys in its state, and - for the Responses conversation with the
tool - the run waiting on the client's tool, then carrying on.

## The runtime file

```yaml
adapters:
  - responses: {prefix: /v1, agent: concierge}    # OpenAI Responses + Conversations (+ /v1/models)
  - assistants: {prefix: /v1, agent: concierge}   # OpenAI Assistants API
  - a2a: {prefix: /a2a}                           # A2A v1.0
```

Each adapter takes a `prefix` and `agents` (which agents it exposes, all by default); `agent` is the one a new
conversation or thread starts with. They sit behind the deployment's auth and access, like `/v2`: with an API
key configured, set `LLAMPHOUSE_API_KEY` for the client (the OpenAI client sends it as the bearer token).
AG-UI is one more: `- ag_ui: {prefix: /ag-ui}` (`pip install 'llamphouse[ag-ui]'`, see `17_AGUI`).
