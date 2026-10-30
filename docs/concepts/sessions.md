# Sessions

A thread keeps every message. An agent rarely wants them all: past a few
dozen, older messages cost tokens and attention for little. A **session**
is an agent's view of a thread for one run — the messages it gets, and a
summary of older ones if it keeps one — and a **session manager** decides
that view.

```python
from llamphouse.core import Agent, Compacting, Context, openai_summarizer

class SupportAgent(Agent):
    session_manager = Compacting(openai_summarizer("gpt-4o-mini"), max_tokens=100_000)

    async def run(self, context: Context):
        reply = await my_llm(messages=context.session.to_messages())
        await context.reply(reply)
```

## `context.session`

| | |
|---|---|
| `session.messages` | the messages in view, oldest first (`MessageObject`) |
| `session.summary` | what older messages said, or `None` |
| `session.folded` | how many of the thread's messages the summary stands for |
| `session.to_messages(summary_role="system")` | `[{"role", "content"}]`: the summary first, then the messages |
| `session.text()` | the session as one text |

`context.messages` is still there too: the thread's latest 100 messages.

## Which manager an agent gets

Every agent has a session. Without anything configured, it is a
**sliding window of the latest 20 messages**. In order, the first that is
set wins:

1. the agent's own entry in `llamphouse.yaml` (`agents[].session_manager`);
2. the agent's own `session_manager` (class attribute, or
   `LangGraphAgent(session_manager=...)`);
3. the server's default: `LLAMPHouse(session_manager=...)`
   (`DistributedWorker(session_manager=...)`), or `session_manager:` at the
   top of `llamphouse.yaml`;
4. `SlidingWindow(max_messages=20)` (`sessions.DEFAULT_SESSION_MANAGER`).

```yaml
session_manager:              # the default of every agent that sets none
  summarizing:
    summarizer: openai        # or {openai: {model: gpt-4o-mini}}, or file.py:callable
    max_messages: 40
    keep_messages: 20

agents:
  - name: router
    definition: router
    session_manager:          # this agent's own
      sliding_window:
        max_messages: 4
  - name: reviewer
    definition: reviewer
    session_manager:
      full_history:
```

The YAML names are `full_history`, `sliding_window`, `summarizing` and `compacting`,
with the constructor arguments below; callables (`summarizer`,
`count_tokens`) are given as `file.py:name` or `module:name`.

## Built-in managers

| Manager | View | Summarizes |
|---|---|---|
| `FullHistory()` | the whole thread | — |
| `SlidingWindow(max_messages=20, max_tokens=None, keep_first=0)` | the latest messages, within a message count and/or token budget; `keep_first` keeps the thread's first n in view too | — |
| `Summarizing(summarizer, max_messages=50, keep_messages=None, summary_tokens=400)` | the latest messages, and a running summary of the older ones | past `max_messages`, the oldest — down to `keep_messages` (default: half) |
| `Compacting(summarizer, max_tokens, compact_at=0.9, compact_to=0.25, summary_tokens=None, reserve_tokens=0)` | a token budget that fills up | at `compact_at` of the budget, everything but the latest messages — down to `compact_to` |

Every manager also takes:

| Argument | |
|---|---|
| `count_tokens` | `(text) -> int`; about 4 characters a token by default — pass your model's tokenizer for exact counts |
| `max_message_tokens` | shorten any single message in view past this many tokens (a pasted document, a long tool output) — in the view only; the thread keeps it whole |

### `Summarizing` — a count of recent messages, and a summary

Once the session holds more than `max_messages` messages, the oldest are
folded into the summary until `keep_messages` remain — so the summarizer
runs once in a while, not on every run. Each compaction folds only what is
new, on top of the summary so far.

### `Compacting` — a token budget that fills up and is compacted

```python
Compacting(summarizer, max_tokens=100_000, reserve_tokens=8_000)
```

```
tokens
 90% ┤      ╱│          ╱│          ╱│
     │    ╱  │        ╱  │        ╱  │
     │  ╱    │      ╱    │      ╱    │
 25% ┤╱      │    ╱      │    ╱      │
     │       │  ╱        │  ╱        │
     │       ╰╱          ╰╱          ╰ …
     └──────────────────────────────────▶ runs
```

- **The budget** is `max_tokens` less `reserve_tokens` (what the agent adds
  that is not in the session: instructions, tools, room for the answer).
  The session's size is its summary *and* its messages.
- **While under `compact_at`** (default 90%) of the budget, nothing
  happens: every message since the last compaction is there, as it is.
- **At `compact_at`**, it is compacted to `compact_to` (default 25%):
  the latest messages that fit stay, next to a new summary of at most
  `summary_tokens` (default: a tenth of the budget) of everything else —
  the previous summary included. Then it fills up again.

Size `max_tokens` from the model's context window and what else goes in
the prompt; pass the model's tokenizer as `count_tokens` for exact counts.

### Summaries

A summary is kept per thread and agent (in the data store's key-value
records, deleted with the thread), and so is up to which message it goes.
The summarizer is any `async (previous_summary, messages) -> str`; one that
also takes a `max_tokens` keyword is told the summary's target length
(`summary_tokens`). Two helpers make one from an LLM call, and aim for
that length:

```python
from llamphouse.core import llm_summarizer, openai_summarizer

# Any LLM: a function from a prompt to text (sync or async).
summarizer = llm_summarizer(lambda prompt: my_llm.complete(prompt))

# An OpenAI-compatible chat-completions API (client: an AsyncOpenAI; default from the environment).
summarizer = openai_summarizer("gpt-4o-mini", client=AsyncOpenAI(base_url=...), temperature=0)
```

Both build the prompt from instructions (`instructions=` to change them —
the default keeps goals, facts, preferences, decisions and open questions),
the summary so far, and the messages to fold in.

If the summarizer fails (a time out, a rate limit), the run goes on with
the session as it is — not compacted — a warning is logged, and a later
run compacts.

## When compaction happens

Compaction happens **inside a run** — never as a run of its own, and not
between runs:

- **At the start of a run**, when the worker loads the run's session: if
  the session is past its manager's limit (`Summarizing`: `max_messages`;
  `Compacting`: `compact_at`), it is compacted before `Agent.run` starts. It
  runs under the run's heartbeat and time out, and is recorded as a
  `session.compact` step of that run. The thread keeps every message; the
  summary is kept per thread and agent.
- **During a run**, when the agent asks for it: `context.compact(messages)`
  compacts the agent's *own* working transcript — see below.

A run that runs again replays its compaction (no second summarizer call).
A thread nobody runs is never compacted.

### Events

A compaction tells the run's client, on its event stream:

| Event | Data |
|---|---|
| `session.compaction.started` | `message`, `scope` (`"thread"`: the session; `"run"`: `context.compact`), `messages` (to fold), `tokens` |
| `session.compaction.completed` | `message`, `scope`, `folded`, `tokens_before`, `tokens_after` |
| `session.compaction.failed` | `message`, `scope`, `error` — the run goes on, uncompacted |

Over A2A a streaming client sees each `message` as a status update; over
AG-UI they are `CUSTOM` events (`name`: the event, `value`: the data).

## Coding and long-running agents: `context.compact`

An agent that works in a loop — call the model, run a tool, read the
result, call the model again — grows its context *within* one run, in its
own list of messages, not in the thread. `context.compact` keeps that list
within budget, by the same `Compacting` policy:

```python
class CodingAgent(Agent):
    session_manager = Compacting(openai_summarizer("gpt-4o-mini"), max_tokens=150_000, reserve_tokens=20_000)

    async def run(self, context: Context):
        messages = [{"role": "system", "content": SYSTEM}, *context.session.to_messages()]
        while True:
            messages = await context.compact(messages)          # as is, until it reaches 90%
            response = await llm(messages=messages, tools=TOOLS)
            messages.append(response.message)
            if not response.tool_calls:
                break
            for call in response.tool_calls:
                messages.append(tool_result(call, await run_tool(call)))
        await context.reply(response.message.content)
```

- Under `compact_at` of the budget the list comes back as it is; at it, it
  is compacted to `compact_to`: the leading system messages stay, the latest
  messages that fit stay as they are, and the rest becomes one summary
  message (an earlier summary message folded in).
- **Tool calls stay whole:** the kept messages never start with a tool
  result whose call was folded (OpenAI `tool` messages, Anthropic
  `tool_result` blocks); the call is kept with it.
- Messages are dicts or objects with `role` / `content` — the OpenAI,
  Anthropic and LangChain shapes, tool calls included. The summary message
  has `summary_role` (`"user"` by default, which every chat API accepts).
- `policy=Compacting(...)` uses another policy than the agent's session
  manager (one is needed: `context.compact` raises `TypeError` without).
- Recorded as a `session.compact_messages` step, replayed by a run that
  runs again; told to the client as above, with `scope: "run"`.

## A manager of your own

Subclass `SessionManager` and implement `build`:

```python
import re

from llamphouse.core import Session, SessionManager
from llamphouse.core.sessions import thread_messages

class KeyFacts(SessionManager):
    """The latest 2 messages, and every number the user mentioned."""

    async def messages(self, context, memory):
        # Optional: what build() gets. Default: the messages after memory["folded_until"].
        return await thread_messages(context, after=memory.get("folded_until"))

    async def build(self, messages, memory):
        facts = memory.get("facts", []) + [n for m in messages for n in re.findall(r"\d+", m.text)]
        return Session(
            summary=f"Numbers mentioned: {', '.join(facts)}" if facts else None,
            messages=messages[-2:],
            memory={"facts": facts, "folded_until": messages[-1].id} if messages else None,
        )
```

- **`build(messages, memory) -> Session`** — the view, from the thread's
  messages (oldest first) and the manager's memory (`{}` at first).
- **The memory** is a JSON dict kept per thread and agent. Return it in
  `Session.memory` to change it (it replaces the old one); leave it `None`
  to keep it as it is. A `"folded_until"` message id in it means "the
  session no longer needs this message and those before it": the default
  `messages()` reads from after it.
- **`messages(context, memory)`** — what of the thread `build` gets;
  override it to read less. `thread_messages(context, after=..., latest=...,
  first=...)` reads the thread, less what a resumed run already said.
- **`uses_memory = False`** — for a manager without a memory (skips reading
  and writing it).
- `self.tokens(messages)` counts tokens with the manager's `count_tokens`.

## Durable runs

A session whose memory changed (a compaction) is recorded as a
`session.compact` step of the run, with the view it gave. A run that runs
again — resumed after a wait, retried, taken over — gets that same session,
and the summarizer is not called twice. Any other `build` must give the
same view for the same messages and memory. See [Durable Runs](durable-runs.md).

## LangGraph

The default state mapper passes the session's `messages` and its `summary`
to the graph: by default the latest 20 messages. Give the agent
`session_manager=FullHistory()` to pass the whole thread.

See the [Sessions example](https://github.com/llamp-ai/llamphouse/tree/main/examples/17_Sessions).
