"""LangGraph wrapper example for LLAMPHouse.

Run:
  1) pip install -r requirements.txt
  2) python server.py
  3) python client.py
"""

from dotenv import load_dotenv
from typing import Any, TypedDict

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.wrappers import LangGraphAgent


try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    raise RuntimeError(
        "This example requires langgraph. Install dependencies with: pip install -r requirements.txt"
    ) from exc


def _build_graph():
    class GraphState(TypedDict, total=False):
        messages: list[dict[str, Any]]
        output: str

    def respond(state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            text = "Hello from LangGraph inside LLAMPHouse."
        else:
            last = messages[-1]
            text = "Echo from LangGraph: " + str(last.get("content", ""))
        return {"output": text}

    graph = StateGraph(GraphState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def main():
    graph = _build_graph()
    agent = LangGraphAgent(
        id="langgraph-agent",
        name="LangGraph Agent",
        description="A minimal LangGraph-backed agent running in LLAMPHouse.",
        graph=graph,
        stream=False,
    )

    app = LLAMPHouse(
        agents=[agent],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
    )
    app.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
