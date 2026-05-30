from agents.state import GraphState
from agents.planner.prompts.planner_reflect_prompt import PLANNER_REFLECT_PREFIX_PROMPT
import json
import asyncio
import logging
from agents.base_node import BaseNode
from agents.planner.custom_types import ToolCallSchema

logger = logging.getLogger("llamphouse.planner")

class ExecutorNode(BaseNode):

    def __init__(self, max_plan_steps: int, max_tool_calls: int, tool_registry: dict, **kwargs):
        self.max_plan_steps = max_plan_steps
        self.max_tool_calls = max_tool_calls
        self.tool_registry = tool_registry
        super().__init__(**kwargs)

    async def _execute_call(self, call: ToolCallSchema) -> dict:
        """Execute one ToolCallSchema; return result dict."""
        name = call.name
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError:
            args = {}
        fn = self.tool_registry.get(name)
        if fn is None:
            result = {"error": f"Unknown tool: {name}"}
        else:
            try:
                result = await fn.acall(**args)
            except Exception as exc:
                result = {"error": str(exc)}

        return {"tool": name, "arguments": args, "result": result}

    async def run(self, state: GraphState) -> dict:
        response = state["last_response"]
        steps = (response.steps or [])[:self.max_plan_steps]
        total_calls = state["total_calls"]
        tool_results: list[dict] = []
        messages = list(state["messages"])
        context = state["context"]

        for step in steps:
            if total_calls >= self.max_tool_calls:
                logger.warning("max_tool_calls (%d) reached", self.max_tool_calls)
                context.send_chunk(
                    f"\n⚠️ Tool call limit ({self.max_tool_calls}) reached.\n"
                )
                break

            if step.type == "single" and step.call:
                result = await self._execute_call(step.call)
                total_calls += 1
                tool_results.append(result)
                context.send_chunk(
                    f"- `{result['tool']}` → {json.dumps(result['result'], ensure_ascii=False)}\n"
                )

            elif step.type == "parallel" and step.parallel:
                remaining = self.max_tool_calls - total_calls
                calls = step.parallel[:remaining]
                if not calls:
                    break
                context.send_chunk(
                    f"- Parallel ({len(calls)} calls): "
                    + ", ".join(f"`{c.name}`" for c in calls)
                    + "\n"
                )
                results = await asyncio.gather(
                    *[self._execute_call(c) for c in calls]
                )
                total_calls += len(results)
                for r in results:
                    tool_results.append(r)
                    context.send_chunk(
                        f"  - `{r['tool']}` → {json.dumps(r['result'], ensure_ascii=False)}\n"
                    )

        results_summary = json.dumps(tool_results, indent=2, ensure_ascii=False)
        messages.append({
            "role": "user",
            "content": f"{PLANNER_REFLECT_PREFIX_PROMPT}\n\n```json\n{results_summary}\n```",
        })
        context.send_chunk("\n")

        return {
            "messages": messages,
            "total_calls": total_calls,
            "tool_results": tool_results,
        }