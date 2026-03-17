"""
Central Orchestrator — single server (Port 8000).

Three agents share one LLAMPHouse instance:
  • Orchestrator (default) — reviews sub-agent output, requests corrections
  • Researcher             — gathers factual information
  • Writer                 — crafts polished articles

The orchestrator demonstrates a review-and-correct loop:

  1. Ask the Researcher for raw facts.
  2. Review the research — identify errors and gaps.
  3. Send feedback to the Researcher — they revise.
  4. Pass revised research to the Writer for an article draft.
  5. Review the article — check structure, engagement, accuracy.
  6. Send feedback to the Writer — they produce the final version.

Both sub-agents receive explicit feedback and correct their work,
showing how an orchestrator can enforce quality across a pipeline.

Start the server:
    python server.py

Then run the client:
    python client.py
"""

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.workers.async_worker import AsyncWorker

openai_client = AsyncOpenAI()


# ── Researcher ───────────────────────────────────────────────────────────────────

RESEARCHER_SYSTEM_PROMPT = (
    "You are a thorough research analyst. "
    "When given a topic or question, provide accurate, well-structured factual "
    "information: key facts, background context, and relevant data. "
    "Be concise but comprehensive. Do not include writing tips or style advice."
)


class ResearcherAgent(Agent):
    async def run(self, context: Context):
        messages = [{"role": "system", "content": RESEARCHER_SYSTEM_PROMPT}]
        for msg in context.messages:
            if msg.text:
                messages.append({"role": msg.role, "content": msg.text})

        stream = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )

        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)
        if full_text and full_text.strip():
            await context.insert_message(full_text)


# ── Writer ───────────────────────────────────────────────────────────────────────

WRITER_SYSTEM_PROMPT = (
    "You are a professional writer and editor. "
    "When given research material and a writing goal, craft clear, engaging content. "
    "Focus on structure, clarity, and readability. "
    "Transform raw information into polished prose."
)


class WriterAgent(Agent):
    async def run(self, context: Context):
        messages = [{"role": "system", "content": WRITER_SYSTEM_PROMPT}]
        for msg in context.messages:
            if msg.text:
                messages.append({"role": msg.role, "content": msg.text})

        stream = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )

        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)
        if full_text and full_text.strip():
            await context.insert_message(full_text)


# ── Review prompts ────────────────────────────────────────────────────────────────

REVIEW_RESEARCH_PROMPT = (
    "You are a senior fact-checker reviewing research material.\n"
    "Evaluate the research and provide specific, actionable feedback:\n"
    "  1. Flag any factual errors with corrections.\n"
    "  2. Identify important gaps that must be filled.\n"
    "  3. Note any unsupported claims.\n\n"
    "Format your feedback as a numbered list of specific issues.\n"
    "Start with a one-line verdict: how many issues you found.\n"
    "Be direct and concise."
)

REVIEW_ARTICLE_PROMPT = (
    "You are a senior editor reviewing a draft article.\n"
    "Evaluate the article and provide specific, actionable feedback:\n"
    "  1. Flag structural issues (poor flow, missing intro/conclusion).\n"
    "  2. Identify unclear or confusing passages.\n"
    "  3. Note any factual claims that don't match the research provided.\n"
    "  4. Suggest improvements for engagement and readability.\n\n"
    "Format your feedback as a numbered list of specific issues.\n"
    "Start with a one-line verdict: how many issues you found.\n"
    "Be direct and concise."
)


# ── Orchestrator ─────────────────────────────────────────────────────────────────


class OrchestratorAgent(Agent):
    """
    Review-and-correct orchestrator:
      1. Researcher drafts  →  2. Review research  →  3. Researcher revises
      4. Writer drafts      →  5. Review article   →  6. Writer revises
    """

    async def _review(self, context, system_prompt, content, original_request):
        """Run an LLM review and stream the feedback to the client."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f'Original request: "{original_request}"\n\n'
                f"Content to review:\n---\n{content}\n---"
            )},
        ]

        stream = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )

        feedback = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                feedback += delta.content
                context.send_chunk(delta.content)
        return feedback

    async def run(self, context: Context):
        user_message = ""
        for m in context.messages:
            if m.role == "user" and m.text:
                user_message = m.text

        if not user_message:
            await context.insert_message("I didn't receive a message to work with.")
            return

        # ══ Phase 1: Researcher drafts ═══════════════════════════════════════════
        context.emit("progress", {"message": "📚 Phase 1: Researcher gathering facts..."})
        context.send_chunk("## 📚 Draft Research\n\n")

        research = ""
        async for chunk in context.call_agent("researcher", user_message):
            research += chunk
            context.send_chunk(chunk)

        # Capture the researcher's thread so we can send feedback on it
        researcher_thread = context.last_call_thread_id

        # ══ Phase 2: Review research ════════════════════════════════════════════
        context.emit("progress", {"message": "🔍 Phase 2: Reviewing research..."})
        context.send_chunk("\n\n## 🔍 Research Review\n\n")

        research_feedback = await self._review(
            context, REVIEW_RESEARCH_PROMPT, research, user_message
        )

        # ══ Phase 3: Researcher revises (same thread — sees its draft) ═══════
        context.emit("progress", {"message": "🔄 Phase 3: Researcher revising..."})
        context.send_chunk("\n\n## 🔄 Revised Research\n\n")

        revision_prompt = (
            f"An editor reviewed your research and found these issues:\n\n"
            f"---\n{research_feedback}\n---\n\n"
            f"Please revise your research to address every issue listed above. "
            f"Output the complete revised research."
        )

        revised_research = ""
        async for chunk in context.call_agent(
            "researcher", revision_prompt, thread_id=researcher_thread,
        ):
            revised_research += chunk
            context.send_chunk(chunk)

        # ══ Phase 4: Writer drafts ═══════════════════════════════════════════════
        context.emit("progress", {"message": "✍️ Phase 4: Writer creating article..."})
        context.send_chunk("\n\n## ✍️ Draft Article\n\n")

        writer_prompt = (
            f"Write a polished, engaging article based on this research.\n\n"
            f"Research:\n---\n{revised_research}\n---\n\n"
            f'Original request: "{user_message}"'
        )

        draft_article = ""
        async for chunk in context.call_agent("writer", writer_prompt):
            draft_article += chunk
            context.send_chunk(chunk)

        # Capture the writer's thread so we can send feedback on it
        writer_thread = context.last_call_thread_id

        # ══ Phase 5: Review article ═════════════════════════════════════════════
        context.emit("progress", {"message": "🔍 Phase 5: Reviewing article..."})
        context.send_chunk("\n\n## 🔍 Article Review\n\n")

        article_feedback = await self._review(
            context, REVIEW_ARTICLE_PROMPT, draft_article, user_message
        )

        # ══ Phase 6: Writer revises (same thread — sees its draft) ═══════════
        context.emit("progress", {"message": "✏️ Phase 6: Writer revising..."})
        context.send_chunk("\n\n## ✏️ Final Article\n\n")

        revision_prompt = (
            f"An editor reviewed your article and found these issues:\n\n"
            f"---\n{article_feedback}\n---\n\n"
            f"Please revise the article to address every issue listed above. "
            f"Output the complete revised article."
        )

        final_article = ""
        async for chunk in context.call_agent(
            "writer", revision_prompt, thread_id=writer_thread,
        ):
            final_article += chunk
            context.send_chunk(chunk)

        # Save complete output
        full_output = (
            f"## 📚 Draft Research\n\n{research}"
            f"\n\n## 🔍 Research Review\n\n{research_feedback}"
            f"\n\n## 🔄 Revised Research\n\n{revised_research}"
            f"\n\n## ✍️ Draft Article\n\n{draft_article}"
            f"\n\n## 🔍 Article Review\n\n{article_feedback}"
            f"\n\n## ✏️ Final Article\n\n{final_article}"
        )
        await context.insert_message(full_output)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    orchestrator = OrchestratorAgent(
        id="orchestrator",
        name="Orchestrator",
        description=(
            "A central orchestrator that reviews sub-agent output and requests "
            "corrections. Coordinates a Researcher and Writer through a "
            "draft → review → revise pipeline."
        ),
    )

    researcher = ResearcherAgent(
        id="researcher",
        name="Researcher",
        description="Gathers facts and background information on any topic.",
    )

    writer = WriterAgent(
        id="writer",
        name="Writer",
        description="Crafts polished, engaging content from research material.",
    )

    llamphouse = LLAMPHouse(
        agents=[orchestrator, researcher, writer],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
        worker=AsyncWorker(time_out=180.0),
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
