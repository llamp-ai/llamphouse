# Quickstart

In five minutes: an agent, a file that describes it, a server in memory, and a conversation over HTTP.
No model and no database are needed.

## 1. Write the agent

```python
# agents.py
from llamphouse.sdk import Agent, Config


class GreeterConfig(Config):
    greeting: str = "Hello"


class Greeter(Agent[GreeterConfig]):
    """Greets you, and asks your name if it doesn't know it yet."""

    async def run(self, ctx) -> None:
        name = await ctx.state.get("name")
        if name is None:
            name = (await ctx.ask("What's your name?")).text
            await ctx.state.set("name", name)
        await ctx.send(f"{ctx.config.greeting}, {name}! You said: {ctx.input.text}")
```

An agent is a class with an async `run(self, ctx)`. Each time someone starts a run, llamphouse calls
`run` with a [run context](../concepts/context.md):

- `ctx.input` is the message that started the run.
- `ctx.state` is the thread's state: facts that stay across turns.
- `ctx.ask(...)` asks a question and waits for the answer. The run pauses without holding a worker,
  and carries on from this line when the answer arrives.
- `ctx.send(...)` puts a message on the thread.
- `ctx.config` is the agent's typed config.

## 2. Describe it

```yaml
# agents.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: greeter}
spec:
  entrypoint: agents.py:Greeter   # a file next to this one, or a module path (my_pkg.agents:Greeter)
  config: {greeting: Hi}          # checked against GreeterConfig
```

The agent file names the agent (`greeter`: what callers use) and says which class runs it. The class
must be importable by module path (defined at module level, not inside a function).

Check it without starting anything:

```bash
llamphouse check --agents agents.yaml
```

```
agent    greeter                  agents:Greeter
runtime  in memory (no runtime file, or no components in it)
ok
```

## 3. Run it

```bash
llamphouse up --memory --agents agents.yaml
```

`--memory` runs everything in one process, in memory: the API, a worker and Compass. Nothing survives
a restart. The server listens on `http://127.0.0.1:8000` (`--port` to change it).

## 4. Talk to it

Create a thread for the agent, then start a run on it with your message:

```bash
curl -s -X POST localhost:8000/v2/threads -H 'content-type: application/json' \
  -d '{"agent": "greeter"}'
# {"id": "thread_5d73f54fbcca49b4", "agent": "greeter", "active_run": null, ...}

curl -s -X POST localhost:8000/v2/threads/thread_5d73f54fbcca49b4/runs \
  -H 'content-type: application/json' -d '{"text": "hello there"}'
# {"id": "run_377e317f58d04794", "status": "queued", ...}
```

The run asks for your name, so it stops at `input_required` and shows its question:

```bash
curl -s localhost:8000/v2/runs/run_377e317f58d04794
# {"status": "input_required", "pending": {"reason": "What's your name?", ...}, ...}
```

Answer it. The run continues from where it asked:

```bash
curl -s -X POST localhost:8000/v2/runs/run_377e317f58d04794/answer \
  -H 'content-type: application/json' -d '{"text": "Ada"}'
```

Read the thread:

```bash
curl -s localhost:8000/v2/threads/thread_5d73f54fbcca49b4/events
```

```
1  message             hello there
2  run.created
3  run.input_required  What's your name?
4  run.input_answered
5  state.changed       name = Ada
6  message             Hi, Ada! You said: hello there
7  run.completed
```

Start another run on the same thread (`{"text": "again"}`): this time the agent knows your name, from
the thread's state, and doesn't ask.

The thread records everything in order: messages, runs starting and ending, questions, state changes.
To follow it live, read `GET /v2/threads/{id}/events/stream` (server-sent events); a run's own stream,
with text as it is written, is `GET /v2/runs/{id}/events/stream`.

## 5. Open Compass

Go to [http://127.0.0.1:8000/compass](http://127.0.0.1:8000/compass). Open the thread: you see the
conversation, the timeline, the state at any position, and each run. Open the run: its input, what it
asked and was answered, the state it changed, and why it is where it is. The composer at the bottom
offers the actions the thread allows right now: start a run when it is idle; send, answer or cancel
while a run is active.

## Next

- [Adding an LLM](adding-llm.md): replace the fixed reply with a model.
- [Threads and runs](../concepts/threads-and-runs.md): what append, start, send and answer mean.
- [Agents](../concepts/agents.md): the agent class, workflows, config and the agent file.
- [Hello world](../../examples/01_HelloWorld) and [Chat](../../examples/02_Chat) examples.
