# Sessions

A thread keeps every message. An agent rarely wants them all: past a few
dozen, older messages cost tokens and attention for little. A **session**
is an agent's view of a thread for one run — the messages it gets, and a
summary of older ones if it keeps one — and a **session manager** decides
that view.

```python
from llamphouse.core import Agent, Context, Summarizing, openai_summarizer

class SupportAgent(Agent):
    session_manager = Summarizing(openai_summarizer("gpt-4o-mini"), max_messages=40, keep_messages=20)

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

`context.session` is `None` for an agent without a session manager;
`context.messages` is still there: the thread's latest 100 messages.

## Built-in managers

| Manager | View | Summarizes |
|---|---|---|
| `FullHistory()` | the whole thread | — |
| `SlidingWindow(max_messages=20, max_tokens=None, keep_first=0)` | the latest messages, within a message count and/or token budget; `keep_first` keeps the thread's first n in view too | — |
| `Summarizing(summarizer, max_messages=50, keep_messages=None, max_tokens=None, keep_tokens=None)` | every message since the last compaction | past a limit, the oldest into a running summary, down to `keep_*` |

Every manager also takes:

| Argument | |
|---|---|
| `count_tokens` | `(text) -> int`; about 4 characters a token by default — pass your model's tokenizer for exact counts |
| `max_message_tokens` | shorten any single message in view past this many tokens (a pasted document, a long tool output) — in the view only; the thread keeps it whole |

### `Summarizing`

Compaction runs only past a limit, and folds down to the `keep_*` limits —
so the summarizer runs once in a while, not on every run; the latest
message always stays. The summary, and up to which message it goes, are
kept per thread and agent (in the data store's key-value records, deleted
with the thread): each compaction folds only what is new, on top of the
summary so far.

The summarizer is any `async (previous_summary, messages) -> str`. Two
helpers make one from an LLM call:

```python
from llamphouse.core import llm_summarizer, openai_summarizer

# Any LLM: a function from a prompt to text (sync or async).
summarizer = llm_summarizer(lambda prompt: my_llm.complete(prompt), max_words=150)

# An OpenAI-compatible chat-completions API (client: an AsyncOpenAI; default from the environment).
summarizer = openai_summarizer("gpt-4o-mini", client=AsyncOpenAI(base_url=...), temperature=0)
```

Both build the prompt from instructions (`instructions=` to change them —
the default keeps goals, facts, preferences, decisions and open questions),
the summary so far, and the messages to fold in.

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

`LangGraphAgent(session_manager=...)`: the default state mapper passes the
session's `messages` and its `summary` to the graph.

See the [Sessions example](https://github.com/llamp-ai/llamphouse/tree/main/examples/17_Sessions).
