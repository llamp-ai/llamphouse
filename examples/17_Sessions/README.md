# 🧵 Sessions

An agent that works with a **session** — its view of the thread: a summary
of the older messages and the recent ones as they are. The session manager
folds the oldest messages into the summary when the thread grows past its
limits, so long conversations stay within an LLM's context.

## What you'll learn

- How `SessionManager` keeps an agent's view of a thread within limits
- How `context.session` gives a run its summary and recent messages
- How `session.to_messages()` feeds a chat-completion API
- How to bring your own summarizer (any LLM, any prompt)

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |
| `OPENAI_API_KEY` (optional) | For an LLM summary; without it a stub keeps the first words of each message |

## Quick start

```sh
pip install -r requirements.txt
llamphouse up
```

In a second terminal:

```sh
python client.py
```

```
> User: I'll travel with my partner, who loves museums.
> Agent: I see 3 recent messages (4 folded into the summary).
Summary: Hi, I'm planning a; I like food markets
...
```

The session grows to 6 messages, then the oldest are folded into the
summary, down to the latest 3. The thread itself keeps every message —
look at it in Compass (<http://127.0.0.1:8000/compass>), where each
compaction is a `session.compact` step of the run that did it.

## How it works

```python
async def summarize(previous: str | None, messages) -> str:
    ...  # your LLM call: fold `messages` into `previous`

class SessionAgent(Agent):
    session_manager = SessionManager(summarizer=summarize, max_messages=6, keep_messages=3)

    async def run(self, context):
        session = context.session
        reply = await llm(messages=session.to_messages())   # summary first, then recent messages
        await context.reply(reply)
```

- The summary is kept per thread and agent, so each run folds only what is
  new since the last compaction, on top of the summary so far.
- Limits by tokens work too: `SessionManager(max_tokens=8000, count_tokens=...)`.
- Without a summarizer, the session is a sliding window.
- A run that runs again (durable runs) gets the same session, without
  summarizing twice.

See [Sessions](../../docs/concepts/sessions.md).
