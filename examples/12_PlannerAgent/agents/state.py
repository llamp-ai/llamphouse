from typing import TypedDict, Optional
from agents.planner.custom_types import PlannerResponse
from llamphouse import Context

class GraphState(TypedDict):
    messages: list[dict]
    context: Context
    total_calls: int
    answer: Optional[str]
    iteration: int
    last_response: Optional[PlannerResponse]
    tool_results: list