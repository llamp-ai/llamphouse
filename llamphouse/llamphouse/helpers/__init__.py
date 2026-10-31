"""Helpers for writing agents: optional, on top of the agent contract (`llamphouse.sdk`) - each its own
module, imported by name, so one never needs another's dependencies.

    llamphouse.helpers.streams     a model's streamed answer into a run: OpenAI (Responses), Anthropic and
                                   Gemini stream events as text and tool-call deltas, and `pipe(ctx, ...)`
    llamphouse.helpers.langgraph   LangGraph: a checkpointer in the run's state, `run_graph`, `LangGraphAgent`
                                   (pip install llamphouse[langgraph])

They use only what any agent can (the RunContext): llamphouse works the same without them.
"""
