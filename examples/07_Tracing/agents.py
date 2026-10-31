"""An agent that answers questions from a small handbook, traced.

llamphouse makes a `llamphouse.run` span around each execution of a run and a `llamphouse.step` span for
each step it executes. Spans the agent makes itself, with any OpenTelemetry tracer, nest under them: here
`handbook.search`, `rerank` and `chat stand-in` (where a model call would be).

No model: `generate` stands in for one, so the example runs offline.
"""

from __future__ import annotations

import asyncio
import json

from opentelemetry import trace

from llamphouse.sdk import Agent, step

tracer = trace.get_tracer("handbook")

HANDBOOK = {
    "leave": "You get 25 days of paid leave a year; ask your manager two weeks ahead.",
    "expenses": "Expenses go through the finance portal within 30 days, with a receipt.",
    "laptop": "IT replaces laptops every three years; broken ones the same week.",
    "remote": "You may work remotely three days a week; tell your team which.",
}


async def generate(question: str, passages: list[str]) -> str:
    await asyncio.sleep(0.05)  # a model, in a real agent
    return passages[0] if passages else "I couldn't find that in the handbook."


class HandbookAgent(Agent):
    """Answers questions about the company handbook."""

    @step
    async def search(self, ctx, question: str) -> list[str]:
        # a step: done once and recorded; it gets a `llamphouse.step` span, and this span nests under it
        with tracer.start_as_current_span("handbook.search") as span:
            words = {w.strip("?.,!").lower() for w in question.split()}
            found = [text for topic, text in HANDBOOK.items() if topic in words]
            span.set_attribute("search.query", question)
            span.set_attribute("search.hits", len(found))
            await asyncio.sleep(0.02)
            return found

    async def run(self, ctx) -> str:
        question = ctx.input.text
        passages = await self.search(ctx, question)

        with tracer.start_as_current_span("rerank") as span:  # under `llamphouse.run`
            passages = sorted(passages, key=len)
            span.set_attribute("rerank.kept", len(passages))

        # where a model call would be: the GenAI attributes are what Compass shows readable and foldable
        with tracer.start_as_current_span("chat stand-in") as span:
            span.set_attribute("gen_ai.request.model", "stand-in")
            span.set_attribute("gen_ai.input.messages", json.dumps(
                [{"role": "system", "content": "Answer from these passages:\n" + "\n".join(passages)},
                 {"role": "user", "content": question}]))
            answer = await generate(question, passages)
            span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "content": answer}]))

        await ctx.send(answer)
        return answer
