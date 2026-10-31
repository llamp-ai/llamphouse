# Examples

The [examples/](https://github.com/llamp-ai/llamphouse/tree/main/examples) directory contains runnable samples for every major feature.
Each example includes an `agents.py` (or `server.py`), `client.py`, and `README.md` with instructions.

## Example index

| Example | Description | Key features |
|---|---|---|
| [01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld) | The simplest llamphouse agent: no model, no keys. Hello reads the message that started its run and answers |  |
| [02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat) | A conversation with a model. Each message you send starts a run of chat on the same thread; the agent reads |  |
| [03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming) | The chat from [02_Chat](../02_Chat), streamed token by token. The agent calls a model with streaming on - |  |
| [04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall) | A model tool loop. The agent calls OpenAI's Responses API with two function tools (current_time, |  |
| [05_ResearchDesk](https://github.com/llamp-ai/llamphouse/tree/main/examples/05_ResearchDesk) | Several agents working together. A desk that plans team offsites compares cities on cost and weather: it |  |
| [06_Handover](https://github.com/llamp-ai/llamphouse/tree/main/examples/06_Handover) | One conversation, passed between agents. A front desk (triage) answers general questions; when you ask |  |
| [07_Tracing](https://github.com/llamp-ai/llamphouse/tree/main/examples/07_Tracing) | An agent that answers questions from a small handbook, traced with OpenTelemetry. llamphouse makes a |  |
| [08_Config](https://github.com/llamp-ai/llamphouse/tree/main/examples/08_Config) | One agent class, set up three ways. A support agent declares what can be tuned as a Config (tone, |  |
| [09_Auth](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_Auth) | Who may call, and what they may touch. The runtime file names the authenticators - API keys, each with its |  |
| [10_Server](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_Server) | The research desk from 05_ResearchDesk, served for real: API processes and worker processes on one Postgres |  |
| [11_Triggers](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_Triggers) | An order desk that other systems and the clock start, not a person. All of it is in the agent file's |  |
| [12_Workflow](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_Workflow) | A workflow: an agent that is a process rather than a conversation. TripPlanner takes typed input (a |  |
| [13_DurableRuns](https://github.com/llamp-ai/llamphouse/tree/main/examples/13_DurableRuns) | A refund agent that asks a human before it pays. The run drafts the refund, asks for approval and waits |  |
| [14_Sessions](https://github.com/llamp-ai/llamphouse/tree/main/examples/14_Sessions) | A session is an agent's working memory on a thread: what it hands its model. The thread keeps every message; |  |
| [15_LangGraph](https://github.com/llamp-ai/llamphouse/tree/main/examples/15_LangGraph) | A LangGraph graph served by llamphouse, with llamphouse.helpers.langgraph. The graph is a small support desk: |  |
| [16_Adapters](https://github.com/llamp-ai/llamphouse/tree/main/examples/16_Adapters) | One agent served over several protocols at once: the native API (/v2), the OpenAI Responses and |  |
| [17_AGUI](https://github.com/llamp-ai/llamphouse/tree/main/examples/17_AGUI) | A trip planner served over [AG-UI](https://docs.ag-ui.com), the protocol agent front-ends (CopilotKit and |  |
| [18_Subscriptions](https://github.com/llamp-ai/llamphouse/tree/main/examples/18_Subscriptions) | Another system hears about runs without holding a stream open. client.py subscribes a URL to the |  |
| [19_CodingAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/19_CodingAgent) | A long-running coding agent that splits its work into subtasks running in parallel, each on a fork of the |  |

## Running an example

Most examples follow the same pattern:

```bash
# Navigate to the example
cd examples/01_HelloWorld

# Install dependencies
pip install -r requirements.txt

# Start the server
llamphouse up   # or: python server.py

# In another terminal, run the client
python client.py
```

Some examples require environment variables (e.g., `OPENAI_API_KEY`). Check each example's `README.md` for specific instructions.

## Progression guide

If you're new to LLAMPHouse, we recommend working through the examples in this order:

1. **[01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld)** — Hello world
2. **[02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat)** — Chat
3. **[03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming)** — Streaming
4. **[04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall)** — Tool calls
5. **[05_ResearchDesk](https://github.com/llamp-ai/llamphouse/tree/main/examples/05_ResearchDesk)** — 05 Research desk
6. **[06_Handover](https://github.com/llamp-ai/llamphouse/tree/main/examples/06_Handover)** — 06 Handover
7. **[07_Tracing](https://github.com/llamp-ai/llamphouse/tree/main/examples/07_Tracing)** — Tracing
8. **[08_Config](https://github.com/llamp-ai/llamphouse/tree/main/examples/08_Config)** — Config
9. **[09_Auth](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_Auth)** — Auth
10. **[10_Server](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_Server)** — Server
11. **[11_Triggers](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_Triggers)** — Triggers
12. **[12_Workflow](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_Workflow)** — 12 Workflow
13. **[13_DurableRuns](https://github.com/llamp-ai/llamphouse/tree/main/examples/13_DurableRuns)** — 13 Durable runs
14. **[14_Sessions](https://github.com/llamp-ai/llamphouse/tree/main/examples/14_Sessions)** — Sessions
15. **[15_LangGraph](https://github.com/llamp-ai/llamphouse/tree/main/examples/15_LangGraph)** — LangGraph
16. **[16_Adapters](https://github.com/llamp-ai/llamphouse/tree/main/examples/16_Adapters)** — Adapters
17. **[17_AGUI](https://github.com/llamp-ai/llamphouse/tree/main/examples/17_AGUI)** — 17 AG-UI
18. **[18_Subscriptions](https://github.com/llamp-ai/llamphouse/tree/main/examples/18_Subscriptions)** — Subscriptions
19. **[19_CodingAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/19_CodingAgent)** — Coding agent

## Next steps

- [Quickstart](getting-started/quickstart.md) — build your first agent from scratch
- [Core Concepts](concepts/agents.md) — understand the fundamentals
- [Guides](guides/streaming.md) — deep dives into specific features
