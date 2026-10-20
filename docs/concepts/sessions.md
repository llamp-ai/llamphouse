# Sessions

A thread keeps every message. An agent rarely wants them all: past a few
dozen, older messages cost tokens and attention for little. A **session**
is an agent's view of a thread for one run — a **summary** of the older
messages, and the **recent** ones as they are. A **session manager** keeps
that view within limits, folding the oldest messages into the summary when
the thread grows past them (compaction).

```python
from llamphouse.core import Agent, Context, SessionManager

async def summarize(previous: str | None, messages) -> str:
    transcript = "\n".join(f"{m.role}: {m.text}" for m in messages)
    return await my_llm(f"Summary so far: {previous}\n\nAdd:\n{transcript}")

class SupportAgent(Agent):
    session_manager = SessionManager(
        summarizer=summarize,
        max_messages=40,     # compact past 40 messages ...
        keep_messages=20,    # ... down to the latest 20
    )

    async def run(self, context: Context):
        reply = await my_llm(messages=context.session.to_messages())
        await context.reply(reply)
```

## `context.session`

| | |
|---|---|
| `session.summary` | what the older messages said, or `None` |
| `session.messages` | the recent messages, oldest first (`MessageObject`) |
| `session.folded` | how many of the thread's messages the summary stands for |
| `session.to_messages(summary_role="system")` | `[{"role", "content"}]`: the summary first, then the recent messages |
| `session.text()` | the session as one text |

`context.session` is `None` for an agent without a session manager.
`context.messages` is still there: the thread's latest 100 messages.

## `SessionManager`

| Argument | Default | |
|---|---|---|
| `summarizer` | `None` | `async (previous_summary, messages) -> str`; `None`: folded messages are dropped (a sliding window) |
| `max_messages` | `50` | compact past this many messages (`None`: no limit) |
| `keep_messages` | half of `max_messages` | messages left after compacting |
| `max_tokens` | `None` | compact past this many tokens |
| `keep_tokens` | half of `max_tokens` | tokens left after compacting |
| `count_tokens` | ~4 characters a token | `(text) -> int`; pass your model's tokenizer for exact counts |

Compaction runs only past a limit, and folds down to the `keep_*` limits —
so the summarizer runs once in a while, not on every run. The latest
message always stays.

- **Kept per thread and agent.** The summary, and up to which message it
  goes, are stored in the data store's key-value records (and deleted with
  the thread). Each run folds only what is new, on top of the summary so
  far. Two agents on one thread each have their own session.
- **Recorded.** A compaction is a `session.compact` step of the run that
  did it, with the new summary as its output.
- **Durable.** A run that runs again (resumed, retried, taken over) gets
  the same session, without calling the summarizer twice. See
  [Durable Runs](durable-runs.md).
- **LangGraph.** `LangGraphAgent(session_manager=...)`: the default state
  mapper passes the session's recent `messages` and its `summary` to the
  graph.

The summarizer is yours: any LLM, any prompt, any provider — LLAMPHouse
calls it with the summary so far and the messages to fold into it.

See the [Sessions example](https://github.com/llamp-ai/llamphouse/tree/main/examples/17_Sessions).
