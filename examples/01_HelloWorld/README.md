# Hello world

The simplest llamphouse agent: no model, no keys. `Hello` reads the message that started its run and answers
it on the thread. The agent file (`agents.yaml`) names it; `llamphouse up` serves it over the native HTTP API
(`/v2`) with Compass next to it. `client.py` creates a thread, starts a run with a message, follows the run's
live stream to its end and prints the reply.

## Run it

```bash
pip install llamphouse
llamphouse up --memory --agents agents.yaml     # http://127.0.0.1:8000, Compass at /compass
python client.py                                # in a second terminal
```

```
thread thread_291f…, run run_777e…
you:   Hi there
hello: Hello Ada! You said: Hi there
[run completed]
```

`--memory` keeps everything in memory: nothing survives a restart. `LLAMPHOUSE_URL` points the client at
another server.

## Try

- `python client.py "what's new?"` - every call is a new thread with one run.
- `llamphouse check --agents agents.yaml` - loads the agent file and says what it found, without serving.
- The same with curl:

  ```bash
  curl -s localhost:8000/v2/threads -d '{"agent": "hello"}' -H 'content-type: application/json'
  curl -s localhost:8000/v2/threads/<thread id>/runs -d '{"text": "hi"}' -H 'content-type: application/json'
  curl -s localhost:8000/v2/threads/<thread id>/events
  ```

## In Compass

Open http://127.0.0.1:8000/compass: the `hello` agent, its threads, and on each thread your message, the run it
started and the agent's reply.

## The code

```python
class Hello(Agent):
    async def run(self, ctx) -> None:
        name = ctx.input.sender.name or ctx.input.sender.id
        await ctx.send(f"Hello {name}! You said: {ctx.input.text}")
```

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: hello}        # what callers address it by
spec:
  entrypoint: agents.py:Hello
```

`ctx.input` is the message that started the run; `ctx.send` writes a message on the run's thread. Returning
ends the run (`completed`); raising `AgentError` fails it.
