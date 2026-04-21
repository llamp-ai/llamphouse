"""
PlannerAgent — a generic ReAct-style planner you can drop into any project.

Loop
────
  1. PLAN   — LLM produces a JSON plan: a list of steps, each step being
              either a single tool call or a parallel group of tool calls.
  2. ACT    — Execute the next step (or all calls in a parallel group)
              concurrently with asyncio.gather.
  3. OBSERVE— Append every tool result to the conversation.
  4. REFLECT— LLM decides: done → write final answer, or revise the plan.

Configurable knobs (all have sensible defaults)
────────────────────────────────────────────────
  max_iterations   Max plan-act-observe-reflect cycles       (default 6)
  max_plan_steps   Max steps the planner may emit per cycle  (default 10)
  max_tool_calls   Hard cap on total tool calls across all   (default 20)
                   iterations (guards against runaway agents)
  model            OpenAI model used for planning/answering  (default gpt-4.1)

Usage
─────
  from planner_agent import PlannerAgent

  agent = PlannerAgent(
      id="my-planner",
      name="My Planner",
      description="...",
      tools=[...],            # list of OpenAI-style tool schemas
      tool_registry={...},    # {"tool_name": callable}
      max_iterations=5,
      max_tool_calls=15,
  )
"""

from llamphouse import Agent, Context
from agents.planner.custom_types import PlannerResponse, ToolCallSchema
from agents.planner.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from agents.state import GraphState
import json
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI
from typing import Dict, Callable, Any

from agents.planner.nodes.planner_node import PlannerNode
from agents.planner.nodes.executor_node import ExecutorNode
from agents.planner.nodes.synthesizer_node import SynthesizerNode


llm = AsyncOpenAI()

class PlannerAgent(Agent):
    """
    Generic ReAct / Planner-Executor agent.

    Drop-in for any use-case — just supply ``tools`` and ``tool_registry``.
    """

    def __init__(
        self,
        id: str,
        *,
        tool_registry: Dict[str, Callable[..., Any]],
        max_iterations: int = 6,
        max_plan_steps: int = 10,
        max_tool_calls: int = 20,
        model: str = "gpt-4.1",
        config: list | None = None,
        **kwargs,
    ):
        # Pass tools (schemas) up to the Agent base class so they appear in
        # the agent card and on context.
        super().__init__(id=id, **kwargs)
        if config is not None:
            self.config = config
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.max_plan_steps = max_plan_steps
        self.max_tool_calls = max_tool_calls
        self.model = model

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self, context: Context) -> None:
        import traceback as _tb
        try:
            await self._run(context)
        except Exception:
            _tb.print_exc()
            raise

    async def _run(self, context: Context) -> None:
        # ── Build initial messages ─────────────────────────────────────────────
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            max_plan_steps=self.max_plan_steps,
            max_tool_calls=self.max_tool_calls,
        )
        if self.tools:
            system_prompt += f"\n\nAvailable tools:\n```json\n{json.dumps(self.tools, indent=2)}\n```"

        init_messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for m in context.messages:
            if m.text:
                init_messages.append({"role": m.role, "content": m.text})

        context.send_chunk("**Planning…**\n\n")

        def route_after_planner(state: GraphState) -> str:
            response = state["last_response"]
            if response is None or response.type == "final_answer":
                return END
            steps = (response.steps or [])[:self.max_plan_steps]
            if not steps:
                return END
            if state["iteration"] >= self.max_iterations:
                return "synthesizer"
            return "executor"
        
        planner_node = PlannerNode(llm, model=self.model, max_plan_steps=self.max_plan_steps, name="planner")
        executor_node = ExecutorNode(max_plan_steps=self.max_plan_steps, max_tool_calls=self.max_tool_calls, tool_registry=self.tool_registry, name="executor")
        synthesizer_node = SynthesizerNode(llm, name="synthesizer")

        # ── Build and compile graph ───────────────────────────────────────────
        builder = StateGraph(GraphState)
        builder.add_node("planner", planner_node.run)
        builder.add_node("executor", executor_node.run)
        builder.add_node("synthesizer", synthesizer_node.run)
        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", route_after_planner)
        builder.add_edge("executor", "planner")
        builder.add_edge("synthesizer", END)
        graph = builder.compile()

        # ── Invoke ────────────────────────────────────────────────────────────
        final_state: GraphState = await graph.ainvoke(
            {
                "messages": init_messages,
                "total_calls": 0,
                "answer": None,
                "iteration": 0,
                "last_response": None,
                "tool_results": [],
                "context": context,
            },
            config={"recursion_limit": self.max_iterations * 2 + 5},
        )

        answer = final_state.get("answer") or ""
        await context.insert_message(answer)
