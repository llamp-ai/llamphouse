"""LangGraph branching workflow example for LLAMPHouse.

Run:
    1) pip install -r requirements.txt
    2) python server.py
    3) python client.py
"""

from dotenv import load_dotenv
from typing import Any, TypedDict
import os
import logging

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.adapters.compass import CompassAdapter
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.tracing.stores.in_memory_tracing_store import InMemoryTracingStore
from llamphouse.core.wrappers import LangGraphAgent


logger = logging.getLogger(__name__)


try:
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor  # type: ignore[import-not-found]
except ImportError:
    LangchainInstrumentor = None


if LangchainInstrumentor is not None:
    # Instrument LangChain/LangGraph internals so node/model/tool calls emit spans.
    LangchainInstrumentor().instrument()
else:
    logger.warning(
        "LangChain instrumentor not available. Install opentelemetry-instrumentation-langchain for node-level spans."
    )


try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    raise RuntimeError(
        "This example requires langgraph. Install dependencies with: pip install -r requirements.txt"
    ) from exc


try:
    from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
except ImportError:
    ChatOpenAI = None


def _build_graph():
    llm = None
    if ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    elif ChatOpenAI is None:
        logger.warning("langchain-openai not installed. Using deterministic fallback responses.")
    else:
        logger.warning("OPENAI_API_KEY is not set. Using deterministic fallback responses.")

    async def _call_llm(system_prompt: str, user_prompt: str, fallback: str) -> str:
        if llm is None:
            return fallback
        try:
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = getattr(response, "content", "")
            if isinstance(content, str):
                return content.strip() or fallback
            if isinstance(content, list):
                chunks: list[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
                joined = "\n".join(chunks).strip()
                return joined or fallback
        except Exception as exc:
            logger.warning("LLM call failed; using fallback response: %s", exc)
        return fallback

    class GraphState(TypedDict, total=False):
        messages: list[dict[str, Any]]
        user_text: str
        intent: str
        plan: str
        math_answer: str
        facts: str
        risks: str
        general_answer: str
        output: str

    def _extract_text(parts_or_text: Any) -> str:
        if isinstance(parts_or_text, str):
            return parts_or_text
        if isinstance(parts_or_text, list):
            chunks: list[str] = []
            for part in parts_or_text:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            return "\n".join(chunks)
        return str(parts_or_text)

    def ingest_user_input(state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return {"user_text": ""}
        last = messages[-1]
        return {"user_text": _extract_text(last.get("content", ""))}

    def route_intent(state: GraphState) -> GraphState:
        text = state.get("user_text", "").lower()
        if any(tok in text for tok in ["solve", "equation", "calculate", "math"]):
            intent = "math"
        elif any(tok in text for tok in ["compare", "plan", "research", "analyze"]):
            intent = "research"
        else:
            intent = "general"
        return {"intent": intent}

    def choose_branch(state: GraphState) -> str:
        return state.get("intent", "general")

    async def solve_math(state: GraphState) -> GraphState:
        text = state.get("user_text", "")
        compact = text.replace(" ", "")
        # Deterministic toy solver for the demo path.
        if "x+1=4" in compact:
            answer = "x = 3"
        else:
            answer = "I detected a math request. I would solve it step-by-step here."
        answer = await _call_llm(
            system_prompt="You are a precise math tutor. Show concise reasoning and a final answer.",
            user_prompt=f"Solve this math request: {text}",
            fallback=answer,
        )
        return {"math_answer": answer}

    def format_math_output(state: GraphState) -> GraphState:
        return {
            "output": (
                "[math branch]\n"
                f"Input: {state.get('user_text', '')}\n"
                f"Answer: {state.get('math_answer', '')}"
            )
        }

    def make_research_plan(state: GraphState) -> GraphState:
        return {
            "plan": (
                "1) Extract requirements\n"
                "2) Gather supporting facts\n"
                "3) Surface risks/tradeoffs\n"
                "4) Synthesize recommendation"
            )
        }

    async def collect_facts(state: GraphState) -> GraphState:
        text = state.get("user_text", "")
        fallback = (
            f"Key facts for '{text}': scope clarified, constraints identified, "
            "and candidate approaches listed."
        )
        facts = await _call_llm(
            system_prompt="Extract practical facts and constraints. Keep it to 3 bullet-style sentences.",
            user_prompt=f"Create actionable facts for: {text}",
            fallback=fallback,
        )
        return {"facts": facts}

    def collect_risks(state: GraphState) -> GraphState:
        return {
            "risks": (
                "Risks: unclear success metrics, hidden integration effort, "
                "and timeline pressure."
            )
        }

    def synthesize_research(state: GraphState) -> GraphState:
        return {
            "output": (
                "[research branch]\n"
                f"Plan:\n{state.get('plan', '')}\n\n"
                f"Facts: {state.get('facts', '')}\n"
                f"Risks: {state.get('risks', '')}\n\n"
                "Recommendation: start with a thin vertical slice, then iterate."
            )
        }

    async def general_reply(state: GraphState) -> GraphState:
        text = state.get("user_text", "")
        fallback = f"You said: {text}"
        llm_reply = await _call_llm(
            system_prompt="You are a helpful assistant. Reply in 1-2 concise sentences.",
            user_prompt=text,
            fallback=fallback,
        )
        return {
            "general_answer": llm_reply,
            "output": "[general branch]\n" + f"Response: {llm_reply}",
        }

    graph = StateGraph(GraphState)
    graph.add_node("ingest_user_input", ingest_user_input)
    graph.add_node("route_intent", route_intent)
    graph.add_node("solve_math", solve_math)
    graph.add_node("format_math_output", format_math_output)
    graph.add_node("make_research_plan", make_research_plan)
    graph.add_node("collect_facts", collect_facts)
    graph.add_node("collect_risks", collect_risks)
    graph.add_node("synthesize_research", synthesize_research)
    graph.add_node("general_reply", general_reply)

    graph.add_edge(START, "ingest_user_input")
    graph.add_edge("ingest_user_input", "route_intent")

    graph.add_conditional_edges(
        "route_intent",
        choose_branch,
        {
            "math": "solve_math",
            "research": "make_research_plan",
            "general": "general_reply",
        },
    )

    graph.add_edge("solve_math", "format_math_output")
    graph.add_edge("format_math_output", END)

    graph.add_edge("make_research_plan", "collect_facts")
    graph.add_edge("make_research_plan", "collect_risks")
    graph.add_edge("collect_facts", "synthesize_research")
    graph.add_edge("collect_risks", "synthesize_research")
    graph.add_edge("synthesize_research", END)

    graph.add_edge("general_reply", END)
    return graph.compile()


def main():
    # Example safety net: ensure tracing is enabled even when local .env disables it.
    os.environ["LLAMPHOUSE_TRACING_ENABLED"] = "true"

    # Langfuse-style span metadata: these become resource attributes on every span
    # (visible in Compass as chips + queryable in ClickHouse/Postgres stores).
    os.environ.setdefault("OTEL_SERVICE_NAME", "langgraph-demo")
    os.environ.setdefault("OTEL_SERVICE_VERSION", "1.0.0")
    os.environ.setdefault(
        "OTEL_DEPLOYMENT_ENVIRONMENT",
        os.getenv("APP_ENV", "development"),
    )
    # Anything in OTEL_RESOURCE_ATTRIBUTES is merged in as well (comma-separated key=value).
    os.environ.setdefault(
        "OTEL_RESOURCE_ATTRIBUTES",
        "llamphouse.example=15_LangGraph,team=platform",
    )

    graph = _build_graph()
    agent = LangGraphAgent(
        id="langgraph-agent",
        name="LangGraph Agent",
        description="A branched LangGraph workflow running inside LLAMPHouse.",
        graph=graph,
        stream=True,
        map_nodes_to_steps=True,
    )

    app = LLAMPHouse(
        agents=[agent],
        data_store=InMemoryDataStore(),
        tracing_store=InMemoryTracingStore(),
        adapters=[A2AAdapter(), CompassAdapter()],
    )
    app.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
