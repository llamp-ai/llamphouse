# Adding an LLM

The [quickstart](quickstart.md) example returned a hardcoded string. In practice, you'll want to connect to an LLM provider. LLAMPHouse is **LLM-agnostic** — use any provider you like inside your agent's `run()` method.

## OpenAI

```python
from dotenv import load_dotenv
from openai import AsyncOpenAI
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter

load_dotenv()
openai_client = AsyncOpenAI()


class ChatAgent(Agent):
    async def run(self, context: Context):
        messages = [
            {"role": m.role, "content": m.text}
            for m in context.messages
        ]
        result = await openai_client.chat.completions.create(
            messages=messages, model="gpt-4o-mini",
        )
        await context.insert_message(result.choices[0].message.content)


app = LLAMPHouse(
    agents=[ChatAgent(
        id="chat", name="Chat Agent",
        description="Chat with GPT", version="0.1.0",
    )],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
)
app.ignite(host="127.0.0.1", port=8000)
```

Set your API key in a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Google Gemini

```python
from google import genai
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter

gemini_client = genai.Client()


class GeminiAgent(Agent):
    async def run(self, context: Context):
        messages = [
            {"role": m.role, "parts": [{"text": m.text}]}
            for m in context.messages
        ]
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages,
        )
        await context.insert_message(response.text)


app = LLAMPHouse(
    agents=[GeminiAgent(
        id="gemini", name="Gemini Agent",
        description="Chat with Gemini", version="0.1.0",
    )],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
)
app.ignite(host="127.0.0.1", port=8000)
```

## Anthropic

```python
from anthropic import AsyncAnthropic
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter

anthropic_client = AsyncAnthropic()


class ClaudeAgent(Agent):
    async def run(self, context: Context):
        messages = [
            {"role": m.role, "content": m.text}
            for m in context.messages
        ]
        result = await anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages,
        )
        await context.insert_message(result.content[0].text)


app = LLAMPHouse(
    agents=[ClaudeAgent(
        id="claude", name="Claude Agent",
        description="Chat with Claude", version="0.1.0",
    )],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
)
app.ignite(host="127.0.0.1", port=8000)
```

## LangChain / LangGraph

LLAMPHouse works with any framework. Here's a LangChain example:

```python
from langchain_openai import ChatOpenAI
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore

llm = ChatOpenAI(model="gpt-4o-mini")


class LangChainAgent(Agent):
    async def run(self, context: Context):
        last_message = context.messages[-1].text
        result = await llm.ainvoke(last_message)
        await context.insert_message(result.content)
```

See the [LangGraph example](https://github.com/llamp-ai/llamphouse/tree/main/examples/LangGraph) for a full integration.

## Key takeaway

LLAMPHouse doesn't care what happens inside `run()`. Call any API, use any SDK, run any framework — LLAMPHouse handles the server infrastructure, protocol compliance, storage, and streaming around your logic.

## Next steps

- [Streaming](../guides/streaming.md) — stream tokens in real time instead of waiting for the full response
- [Tool Calls](../guides/tool-calls.md) — add function calling to your agents
- [Core Concepts](../concepts/agents.md) — deep dive into agents, context, and adapters
