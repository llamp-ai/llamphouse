# Streaming

The chat from [02_Chat](../02_Chat), streamed token by token. The agent calls a model with streaming on -
OpenAI's Responses API by default, Anthropic or Gemini if you choose - and hands the stream to `pipe`
(`llamphouse.helpers.streams`), which writes the text into the run's message as it arrives and reports the
tokens it cost. The client follows the run's live stream and prints each piece of text as it comes; the
thread gets the whole message when it is done. The SDK calls stay in the example: the helpers only read the
stream they are given.

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
llamphouse up --memory --agents agents.yaml     # http://127.0.0.1:8000, Compass at /compass
python client.py                                # in a second terminal: chat; an empty line stops
```

```
you: Tell me a short story about a lighthouse.
  [Asking gpt-5-mini]
streaming: Once upon a time, on a rocky … (printed as it is written)
  [gpt-5-mini: 25 tokens in, 180 out]
```

Another provider - its SDK and key, then the agent file picks it up from the environment:

```bash
pip install anthropic && export ANTHROPIC_API_KEY=...
MODEL_PROVIDER=anthropic llamphouse up --memory --agents agents.yaml

pip install google-genai && export GEMINI_API_KEY=...
MODEL_PROVIDER=gemini MODEL=gemini-2.5-pro llamphouse up --memory --agents agents.yaml
```

## Try

- Watch the raw stream: `curl -N localhost:8000/v2/runs/<run id>/events/stream` while a run is going
  (`progress`, `message.started`, `message.delta`…, `message.completed`, `usage`, then `end`).
- `MODEL_PROVIDER=mistral llamphouse check --agents agents.yaml`: the config is validated before anything
  runs.

## In Compass

Open a run while it streams: its **Live** tab shows the message growing, then the `progress` and `usage`
events (the model, tokens in and out). Its **Session** tab is what the agent read; the thread holds only
the finished messages.

## The code

```python
async def stream_openai(ctx, model, conversation) -> Streamed:
    stream = await client("openai").responses.create(model=model, instructions=ctx.config.instructions,
                                                     input=[...], stream=True)
    return await pipe(ctx, streams.openai(stream))       # text into the run's message, live

async def stream_anthropic(ctx, model, conversation) -> Streamed:
    async with client("anthropic").messages.stream(model=model, max_tokens=16000, ...) as stream:
        return await pipe(ctx, streams.anthropic(stream))
```

`streams.openai`, `streams.anthropic` and `streams.gemini` turn each provider's stream events into the same
deltas; `pipe` returns what was said (`.text`, `.tool_calls`, `.finished`, `.usage`). Text deltas are live
only: a client that starts listening late gets the whole message from `message.completed`.

Which provider and model is the agent's `Config` (`StreamingConfig`), filled in `agents.yaml` from
`MODEL_PROVIDER` and `MODEL`.
