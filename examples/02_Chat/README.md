# Chat

A conversation with a model. Each message you send starts a run of `chat` on the same thread; the agent reads
its session - the conversation so far, as it sees it - turns it into input for OpenAI's Responses API, and
sends the model's answer back on the thread. It keeps no history of its own: next turn, its earlier answers
are in the session, marked as its own (`ctx.session.is_own`). The model call is the example's own code;
llamphouse doesn't know which model or provider it talks to.

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
llamphouse up --memory --agents agents.yaml     # http://127.0.0.1:8000, Compass at /compass
python client.py                                # in a second terminal: chat; an empty line stops
```

```
thread thread_2b27… - Compass: http://127.0.0.1:8000/compass - empty line or Ctrl-D to stop
you: My name is Ada.
chat: Nice to meet you, Ada!
you: What's my name?
chat: Your name is Ada.
```

`python client.py <thread id>` carries on with an earlier conversation (while the server runs: `--memory`
keeps nothing across a restart).

## Try

- Another model: `config: {model: gpt-5}` in `agents.yaml`, or a different `instructions`. The config is the
  agent's `Config` class (`ChatConfig`), validated when the server starts.
- Two clients on two threads: each conversation has its own session.

## In Compass

Open the thread: your messages and the agent's answers, one run per message. Open a run: its **Session**
tab is what the agent read (its own answers marked `own`), its **Live** tab the `usage` event it reported
(the model and its tokens).

## The code

```python
class ChatConfig(Config):
    model: str = "gpt-5-mini"
    instructions: str = "You are a friendly assistant. Answer briefly."


class Chat(Agent[ChatConfig]):
    async def run(self, ctx) -> None:
        session = await ctx.session.messages()          # the conversation so far, this message included
        response = await openai().responses.create(
            model=ctx.config.model, instructions=ctx.config.instructions,
            input=[{"role": "assistant" if ctx.session.is_own(m) else "user", "content": m.text}
                   for m in session if m.text])
        await ctx.send(response.output_text)            # on the thread: next turn it is in the session
```

`is_own` is decided by which run wrote a message, not by a role field: the agent's earlier runs on the thread
are its own; a person, another agent or a parent steering a fork is not.

Next: [03_Streaming](../03_Streaming) streams the answer token by token.
