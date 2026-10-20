# 🧵 Sessions

Three agents, three **session managers** — each agent's view of the same
kind of conversation: a running summary, a sliding window, and a manager of
your own that remembers the numbers the user mentioned.

## What you'll learn

- How `context.session` gives a run its messages and summary
- The built-in managers: `Summarizing`, `SlidingWindow` (and `FullHistory`)
- How a summarizer is made from an LLM call (`openai_summarizer`, `llm_summarizer`)
- How to write a session manager of your own (`SessionManager.build`)

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
python client.py                          # all three agents
python client.py --agent facts-agent      # one of them
```

Set `LLAMPHOUSE_URL` if the server runs elsewhere.

```
======== summarizing-agent ========
> Agent: I see 3 messages (4 folded into the summary).
Summary: Hi, I'm planning a; I like food markets
======== window-agent ========
> Agent: I see 5 messages (0 folded into the summary).
======== facts-agent ========
> Agent: I see 2 messages (0 folded into the summary).
Summary: Numbers the user mentioned: 1500
```

The threads keep every message — look at them in Compass
(<http://127.0.0.1:8000/compass>), where each compaction is a
`session.compact` step of the run that did it.

## How it works

```python
summarizing = Summarizing(openai_summarizer("gpt-4o-mini"), max_messages=6, keep_messages=3)
window = SlidingWindow(max_messages=4, keep_first=1)


class KeyFacts(SessionManager):
    async def build(self, messages, memory):
        facts = memory.get("facts", []) + numbers_in(messages)
        return Session(
            summary=f"Numbers the user mentioned: {', '.join(facts)}",
            messages=messages[-2:],
            memory={"facts": facts, "folded_until": messages[-1].id},   # kept for the next run
        )
```

- **`Summarizing`** folds the oldest messages into a running summary once
  the session passes 6 messages, down to the latest 3. The summary is kept
  per thread and agent, so each compaction folds only what is new.
- **`SlidingWindow`** shows the latest 4 messages, and the conversation's
  first one (the request it started with).
- **`KeyFacts`** implements `build(messages, memory)`: the view from the
  new messages and what it remembered. Its memory says to read only what
  came after the last message it saw (`folded_until`).

An agent hands the session to its LLM with `context.session.to_messages()`
(the summary first, then the messages).

See [Sessions](../../docs/concepts/sessions.md).
