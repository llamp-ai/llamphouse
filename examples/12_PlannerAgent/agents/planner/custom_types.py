from typing import Literal
from pydantic import BaseModel

class ToolCallSchema(BaseModel):
    name: str
    arguments: str  # JSON-encoded dict


class PlanStep(BaseModel):
    type: Literal["single", "parallel"]
    call: ToolCallSchema | None = None
    parallel: list[ToolCallSchema] | None = None


class PlannerResponse(BaseModel):
    type: Literal["plan", "final_answer"]
    steps: list[PlanStep] | None = None
    answer: str | None = None