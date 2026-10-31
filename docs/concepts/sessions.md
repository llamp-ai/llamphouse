# Sessions

A session is an agent's working memory on a thread: what it hands its model. It holds the thread's
messages plus the agent's own entries that aren't on the thread (tool calls, tool results, retrieved
documents, notes), shaped by a session provider.

| | the thread | the session |
|---|---|---|
| what | the shared record: what was said between the parties, runs, state changes | one agent's working memory on that thread |
| who sees it | everyone with access: people, adapters, Compass, audit | that agent's runs (and Compass) |
| grows | forever, append-only | as the provider decides: a window, a summary, compaction |

Keeping them apart means the transcript people see stays the conversation, while the model gets what
it needs, kept short by a policy you can change without touching agent code.

## In agent code

```python
view = await ctx.session.messages()     # the session as it is now: a sequence of Messages, newest last
ctx.session.is_own(m)                   # did this agent say it? (the model's assistant role)
await ctx.session.add(ToolCallPart(call_id="c1", name="search", arguments={"q": "Lisbon"}))
await ctx.session.add(ToolResultPart(call_id="c1", output=[TextPart(text="3 hotels")]))
await ctx.session.revise(reason="before the big step")   # ask for a revision now; returns at once
view = await ctx.session.settled()      # wait until no revision is in flight, then read
```

- **`messages()`** is cheap: it never computes anything. It returns the current version plus what was
  added since, `ctx.input` and messages the run read from its inbox included. It's recorded, so a run
  that executes again sees the same view. A `SessionView` is a sequence of messages, with `id`,
  `version` and `revising` (a revision was in flight when it was read).
- **`add(*content)`** adds one entry: the agent's own (`is_own` is true), durable, placed after the
  thread position it was added at. Entries aren't on the thread, so other parties don't see them.
- **`is_own(m)`** is decided by which run wrote the message, not by the agent's name: this run, earlier
  runs of the agent on the thread, and (in a fork) the parent before the fork. Another agent, a person,
  or a parent steering its fork are not own.
- **`revise(**options)`** passes its options to the provider (a reason, a target, a hint). The provider
  may decline. One revision at a time: a second request joins the first.
- **`settled()`** waits (holding no worker) until no revision is in flight. Most agents never need it;
  call it before forking or a large model call if you want the revised version.

Turning the session into a model's input is your code, because every API has its own format. See
[Adding an LLM](../getting-started/adding-llm.md) for the OpenAI Responses API, tool calls included.

## One session per agent per thread

The runs of an agent on a thread share its session there: the agent remembers across turns, and
`ctx.restart()` keeps it. A fork, another agent's thread and a handover each start a session of their
own, from what `session=` says:

| `session=` | the new session starts with | the default for |
|---|---|---|
| `ctx.session` | a copy of this session, as it is now. In a fork, what this agent said stays its own | `ctx.fork` |
| a `SessionView` or any list of `Message`s | exactly those: a slice, a summary the agent wrote | |
| `None` | nothing: built from the new thread's own messages | `ctx.call`, `ctx.handover` |

```python
recent = (await ctx.session.messages())[-6:]
await ctx.call("reviewer", "Review the plan above.", session=recent)
```

A thread created over the API can start its first session from given messages
(`POST /v2/threads` with `"session": [...]`): for example the summary of an earlier thread this one
continues.

There is no shared session: a parent and its fork never write the same one. The parent hears from its
fork by listening, and copies over what it wants.

## Session providers

The provider decides what the session shows and when to revise it. Two come with llamphouse:

| provider | what it does |
|---|---|
| `rolling_window` (`RollingWindow(size=50)`) | the default: the most recent `size` messages; never revises |
| `summarizing` (`Summarizing(summarize, max_messages=50, keep=10)`) | when the session holds more than `max_messages`, everything but the last `keep` becomes one summary, written by `summarize`: your function, usually a model call |

The agent file chooses one per agent; the runtime file sets the default for all:

```yaml
# agents.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: coder}
spec:
  entrypoint: agents.py:Coder
  session:
    summarizing:
      summarize: {import: agents:summarize}   # a Python function, by import path
      max_messages: 40
      keep: 8
```

```yaml
# llamphouse.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Runtime
sessions: {rolling_window: {size: 30}}
```

```python
# agents.py
from openai import OpenAI


def summarize(older) -> str:
    """The older messages, as one summary: a model call of your own (async functions work too)."""
    transcript = "\n".join(f"{m.sender.id}: {m.text}" for m in older)
    return OpenAI().responses.create(model="gpt-5-mini",
                                     input=f"Summarize this conversation:\n{transcript}").output_text
```

The summary goes into the session as a system message (sender kind `system`); your conversion to model
input decides its role.

## Revisions run in the background

A revision is never done on a read. The provider's policy hooks watch what is added and ask for a
revision when they decide to; the revision itself runs as a job on a worker, on a snapshot of the
session. Runs carry on meanwhile with the current version. When the revision is done, the new version
is swapped in; what was added during the revision survives, after the revised part. A failed
revision leaves the session as it was.

Each session keeps its own log: entries added and revisions started, completed, declined or failed,
with their options. Compass shows the session a run saw, with its revisions, lined up with the thread.
The API has it at `GET /v2/runs/{id}/session`. With tracing, each revision is a
`llamphouse.session.revise` span.

## A provider of your own

Subclass `SessionProvider` and override what you need. The defaults show everything, never revise and
decline requests.

```python
from llamphouse.runtime.sessions import Revise, SessionInfo, SessionProvider
from llamphouse.sdk import Message, Sender, ToolResultPart


class DropOldToolResults(SessionProvider):
    """Keeps the last few tool results; older ones are compacted away on request or past a size."""

    def __init__(self, keep_results: int = 3, max_messages: int = 200):
        self.keep_results, self.max_messages = keep_results, max_messages

    def view(self, session: SessionInfo):
        return session.messages  # what ctx.session.messages() shows: cheap, no model calls

    def on_entry(self, session: SessionInfo, entry: Message):
        if len(session.messages) > self.max_messages:
            return Revise({"reason": "size"})  # policy hooks only decide
        return None

    async def revise(self, session: SessionInfo, options):
        results = [m for m in session.messages if any(isinstance(p, ToolResultPart) for p in m.parts)]
        old = {m.id for m in results[:-self.keep_results]}
        if not old:
            return None  # nothing to do: declined
        return [m for m in session.messages if m.id not in old]
```

| hook | when | returns |
|---|---|---|
| `admit(session, event)` | every event on the thread its runs may see | the messages it adds to the session |
| `view(session)` | every `ctx.session.messages()` | the messages to show |
| `on_event(session, event)` | after `admit`, for every such event | `Revise(options)` or `None` |
| `on_entry(session, entry)` | the agent added an entry | `Revise(options)` or `None` |
| `after_run(session, run)` | a run of the agent on the thread ended | `Revise(options)` or `None` |
| `revise(session, options)` | the revision, as a job | the new messages, or `None` to decline |

`admit` decides what goes in. The session hears everything on its thread that its runs may see:
messages, the run's questions (`ctx.ask`) and their answers, state changes, runs starting and ending,
events other systems post. The default admits messages, a question (as the agent's own message: the
prompt it asked) and its answer, and nothing else - so a model given the session sees "Which month?" and
"March". Override it to keep more, or less:

```python
class WithStateNotes(RollingWindow):
    def admit(self, session, event):
        if str(event.kind) == "state.changed":
            keys = ", ".join(c["key"] for c in event.data["changes"])
            return [Message(id=f"note_{event.position}", sender=Sender.system("state"),
                            parts=[TextPart(text=f"(state changed: {keys})")])]
        return super().admit(session, event)
```

A message made in `admit` needs a stable id (from the event's position, say): one already in the session
isn't added again. A message sent to a run's inbox enters its session when the run reads it, not before.

A request from agent code (`ctx.session.revise(**options)`) goes straight to `revise`, with
`reason="requested"` added. `SessionInfo` is read-only: `id`, `thread_id`, `agent`, `messages`,
`version`, `revisions`, `revising`.

Name it in the agent file by import path, with its constructor's options (checked before anything
starts):

```yaml
spec:
  session: {my_sessions:DropOldToolResults: {keep_results: 5}}
```

See the [Sessions example](../../examples/14_Sessions).
