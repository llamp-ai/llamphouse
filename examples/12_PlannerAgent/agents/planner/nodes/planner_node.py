from agents.base_node import BaseNode, GraphState
from agents.planner.custom_types import PlannerResponse

class PlannerNode(BaseNode):

    def __init__(self, llm, model: str, max_plan_steps: int, **kwargs):
        self.llm = llm
        self.model = model
        self.max_plan_steps = max_plan_steps
        super().__init__(**kwargs)

    async def run(self, state: GraphState) -> dict:
        context = state["context"]
        messages = state["messages"]

        resp = await self.llm.responses.parse(
            model=self.model,
            input=messages,
            text_format=PlannerResponse,
        )
        response = resp.output_parsed
        
        new_messages = messages + [
            {"role": "assistant", "content": response.model_dump_json()}
        ]
        update: dict = {
            "messages": new_messages,
            "last_response": response,
            "iteration": state["iteration"] + 1,
            "tool_results": [],
        }
        if response.type == "plan":
            steps = (response.steps or [])[:self.max_plan_steps]
            context.send_chunk(
                f"**Iteration {state['iteration'] + 1} — {len(steps)} step(s)**\n\n"
            )
        elif response.type == "final_answer":
            answer = response.answer or ""
            update["answer"] = answer
            context.send_chunk(f"\n---\n\n{answer}")
        return update