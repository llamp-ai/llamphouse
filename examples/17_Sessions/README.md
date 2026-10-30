# 🧵 Sessions

Four agents, four **session managers** — each agent's view of the same
kind of conversation: a running summary, a sliding window, a token budget
that is compacted when it fills up, and a manager of your own that
remembers the numbers the user mentioned.

## What you'll learn

- How `context.session` gives a run its messages and summary
- The built-in managers: `Summarizing`, `SlidingWindow`, `Compacting` (and `FullHistory`)
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
python client.py                          # all four agents
python client.py --agent facts-agent      # one of them
```

Set `LLAMPHOUSE_URL` if the server runs elsewhere.

```
======== summarizing-agent ========
> Agent: I see 5 messages, ~55 tokens (0 folded into the summary).
> Agent: I see 3 messages, ~44 tokens (4 folded into the summary).
Summary: Hi, I'm planning a; I like food markets
======== window-agent ========
> Agent: I see 5 messages, ~58 tokens (0 folded into the summary).
======== compacting-agent ========
> Agent: I see 7 messages, ~81 tokens (0 folded into the summary).
> Agent: I see 1 messages, ~21 tokens (8 folded into the summary).
Summary: My budget is about; I'll travel with my
> Agent: I see 5 messages, ~94 tokens (8 folded into the summary).
> Agent: I see 1 messages, ~22 tokens (14 folded into the summary).
======== facts-agent ========
> Agent: I see 2 messages, ~45 tokens (0 folded into the summary).
Summary: Numbers the user mentioned: 1500
```

Without an LLM, the stand-in summarizer keeps the first words of each
message, cut to the length the manager asks for; with `OPENAI_API_KEY` set,
`openai_summarizer` condenses instead.

The threads keep every message — look at them in Compass
(<http://127.0.0.1:8000/compass>), where each compaction is a
`session.compact` step of the run that did it.

## How it works

```python
summarizing = Summarizing(openai_summarizer("gpt-4o-mini"), max_messages=6, keep_messages=3)
window = SlidingWindow(max_messages=4, keep_first=1)
compacting = Compacting(openai_summarizer("gpt-4o-mini"), max_tokens=120)   # at 90%, compact to 25%


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
- **`Compacting`** lets the session fill up to 90% of its 120-token budget
  (summary included), then folds everything but the latest messages into a
  new summary — down to 25% — and lets it fill up again.
- **`KeyFacts`** implements `build(messages, memory)`: the view from the
  new messages and what it remembered. Its memory says to read only what
  came after the last message it saw (`folded_until`).

An agent hands the session to its LLM with `context.session.to_messages()`
(the summary first, then the messages).

See [Sessions](../../docs/concepts/sessions.md).
