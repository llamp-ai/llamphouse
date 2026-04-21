"""
Example 12 — PlannerAgent demo server.

The agent is wired with a small set of mock tools that simulate a
data-analysis workflow:
  • search_web(query)          — "search the internet" for a topic
  • get_stock_price(ticker)    — look up a stock price
  • calculate(expression)      — evaluate a maths expression
  • summarise_text(text)       — compress a long text to bullet points

Everything is mocked so no API keys are needed beyond an OpenAI key for
the LLM itself.

Start:   python server.py
Client:  python client.py
"""

import math
from typing import Annotated
from dotenv import load_dotenv

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore

from agents.planner.agent import PlannerAgent
from agents.tools import tool, collect_tools


# ── Mock data ─────────────────────────────────────────────────────────────────

_MOCK_WEB: dict[str, str] = {
    "apple":     "Apple Inc. (AAPL) is a technology company. Revenue FY2024: $391B.",
    "microsoft": "Microsoft (MSFT) is a technology company. Revenue FY2024: $245B.",
    "tesla":     "Tesla (TSLA) makes electric vehicles. Revenue FY2024: $97B.",
}

_MOCK_PRICES: dict[str, float] = {
    "AAPL": 213.49,
    "MSFT": 415.30,
    "TSLA": 178.02,
    "GOOG": 174.51,
    "AMZN": 192.45,
}


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def search_web(query: Annotated[str, "Search query"]) -> str:
    """Search the web for information about a company or topic."""
    for key, result in _MOCK_WEB.items():
        if key in query.lower():
            return result
    return f"No results found for '{query}'."


@tool
def get_stock_price(ticker: Annotated[str, "Stock ticker symbol, e.g. AAPL"]) -> dict:
    """Look up the latest stock price for a ticker symbol."""
    price = _MOCK_PRICES.get(ticker.upper())
    if price is None:
        return {"error": f"Unknown ticker: {ticker}"}
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}


@tool
def calculate(expression: Annotated[str, "Python math expression, e.g. '213.49 * 1.1'"]) -> dict:
    """Evaluate a math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}


@tool
def summarise_text(text: Annotated[str, "Text to summarise"]) -> str:
    """Compress a long text to a short summary."""
    return text[:100].strip() + ("…" if len(text) > 100 else "")


# Mock human responses keyed by topic keyword
_MOCK_HUMAN_RESPONSES: dict[str, str] = {
    "company":  "Apple (AAPL)",
    "ticker":   "AAPL",
    "stock":    "Apple (AAPL)",
    "budget":   "$5,000",
    "how many": "10 shares",
}


@tool
def request_clarification(question: Annotated[str, "Clarifying question to ask the user when the task is ambiguous"]) -> str:
    """
    Ask the user a clarifying question when the task is ambiguous.
    Returns the user's response. Use this only when critical information
    is missing and you cannot proceed without it.
    """
    for keyword, answer in _MOCK_HUMAN_RESPONSES.items():
        if keyword in question.lower():
            return f"[User]: {answer}"
    return "[User]: Please use Apple (AAPL) as the default."


TOOL_SCHEMAS, TOOL_REGISTRY = collect_tools(
    search_web,
    get_stock_price,
    calculate,
    summarise_text,
    request_clarification,
)


# ── Agent ─────────────────────────────────────────────────────────────────────

planner = PlannerAgent(
    id="planner-agent",
    name="Planner Agent",
    description="A planner agent that coordinates multiple tools to answer complex research and analysis questions.",
    version="0.1.0",
    tools=TOOL_SCHEMAS,
    tool_registry=TOOL_REGISTRY,
    instructions=(
        "You are a research assistant. Use the available tools to answer the user's question. "
        "Plan your approach, execute tool calls, and synthesise the results into a final answer. "
        "If the user asks you to call another agent, use the `call_agent` tool with the agent's ID and the query to send."
    ),
)

def main():
    llamphouse = LLAMPHouse(
        agents=[planner],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
    )
    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
