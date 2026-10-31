"""Sessions hear every event on their thread; the provider decides what goes in. By default: messages, the
run's question and the answer to it."""

from llamphouse.runtime.sessions import RollingWindow
from llamphouse.sdk import Agent, AuthRequest, Sender
from llamphouse.sdk.testing import Scenario

AN = Sender.person("an")


class Planner(Agent):
    async def run(self, ctx) -> list[str]:
        await ctx.ask("Which month?")
        await ctx.state.set("month", "set")
        await ctx.run.request_auth(AuthRequest(reason="Sign in to the calendar"))
        return [f"{'own' if ctx.session.is_own(m) else 'other'}:{m.text}" for m in await ctx.session.messages()]


class WithNotes(RollingWindow):
    """Keeps state changes too, as notes."""

    def admit(self, session, event):
        if str(event.kind) == "state.changed":
            from llamphouse.sdk import Message, TextPart

            keys = ", ".join(c["key"] for c in event.data["changes"])
            return [Message(id=f"note_{event.position}", sender=Sender.system("state"),
                            parts=[TextPart(text=f"(state changed: {keys})")])]
        return super().admit(session, event)


async def run_it(sessions=None):
    async with Scenario({"planner": Planner}, sessions=sessions) as sc:
        thread = await sc.thread("planner")
        await sc.say(thread, "plan a trip", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        await sc.answer(run.ref, "March", by=AN)
        await sc.settle()
        await sc.answer(run.ref, "secret-token", by=AN)  # credentials
        await sc.settle()
        (run,) = sc.runs(thread)
        answered = [e.data for e in sc.timeline(thread) if str(e.kind) == "run.input_answered"]
        return run.output, answered


async def test_the_question_and_its_answer_are_in_the_session_and_on_the_record():
    output, answered = await run_it()
    assert output == ["other:plan a trip", "own:Which month?", "other:March"]  # never the credentials
    assert answered[0]["answer"].text == "March" and "answer" not in answered[1]  # an auth answer isn't kept


async def test_a_provider_decides_what_goes_in():
    output, _ = await run_it(sessions=WithNotes(50))
    assert output == ["other:plan a trip", "own:Which month?", "other:March", "other:(state changed: month)"]
