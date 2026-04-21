from agents.state import GraphState
from agents.base_node import BaseNode
from agents.planner.custom_types import PlannerResponse

class SynthesizerNode(BaseNode):

    def __init__(self, chat_fn, **kwargs):
        self._chat = chat_fn
        super().__init__(**kwargs)

    async def run(self, state: GraphState) -> dict:
        context = state["context"]
        
        context.send_chunk("\n**Synthesising final answer…**\n\n")
        messages = state["messages"] + [{
            "role": "user",
            "content": (
                "You have used all your iterations. "
                "Write the best final answer you can with the information gathered."
            ),
        }]

        resp = await self.llm.responses.parse(
            model=self.model,
            input=messages,
            text_format=PlannerResponse,
        )
        response = resp.output_parsed

        return {"answer": response.answer or ""}