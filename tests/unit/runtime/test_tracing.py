"""Tracing: a span per execution of a run, under which agent code's spans nest; a span per step executed (none
for a replayed one) and per session revision; a run's spans over the API; runs untouched without tracing."""

import asyncio
import subprocess
import sys

import httpx
import pytest
from opentelemetry import trace

from llamphouse.core.tracing import InMemoryTracingStore
from llamphouse.runtime import Engine
from llamphouse.runtime.api import create_app
from llamphouse.sdk import Agent, RunRef

app_tracer = trace.get_tracer("the.app")


def made_in_a_thread() -> None:
    with app_tracer.start_as_current_span("app.in_thread"):
        pass


class Traced(Agent):
    """Makes spans of its own: in the run, and in a thread."""

    async def run(self, ctx) -> str:
        with app_tracer.start_as_current_span("app.work") as span:
            span.set_attribute("input.value", '{"question": "hi"}')
            await asyncio.sleep(0)
            await asyncio.to_thread(made_in_a_thread)
        return "done"


class Stepper(Agent):
    """A step, a question (the run waits, and is executed again: the step replayed), another step."""

    async def run(self, ctx) -> str:
        a = await ctx.step("first", self.one)
        answer = await ctx.ask("More?")
        b = await ctx.step("second", self.one)
        return f"{a}{b}{answer.text}"

    async def one(self) -> int:
        with app_tracer.start_as_current_span("app.in_step"):
            return 1


class Boom(Agent):
    async def run(self, ctx) -> None:
        raise RuntimeError("boom")


class Reviser(Agent):
    async def run(self, ctx) -> None:
        await ctx.session.revise(why="test")


AGENTS = {"traced": Traced, "stepper": Stepper, "boom": Boom, "reviser": Reviser}


@pytest.fixture
async def traced():
    store = InMemoryTracingStore()
    engine = Engine(AGENTS, tracing_store=store)
    await engine.open()
    yield engine, store
    await engine.close()


async def spans_of(engine, store, run_id):
    engine.tracing.flush()
    return await store.get_trace(run_id)


def named(spans, name):
    return [s for s in spans if s["SpanName"] == name]


async def test_a_run_is_a_span_and_the_spans_agent_code_makes_nest_under_it(traced):
    engine, store = traced
    ref = await engine.start("traced", "hi")
    await engine.settle()
    spans = await spans_of(engine, store, ref.id)
    [run] = named(spans, "llamphouse.run")
    attrs = run["SpanAttributes"]
    assert attrs["run.id"] == attrs["llamphouse.run_id"] == ref.id
    assert attrs["thread.id"] == engine.run(ref).thread_id and attrs["llamphouse.agent"] == "traced"
    assert attrs["llamphouse.run.status"] == "completed" and run["StatusCode"] == "STATUS_CODE_OK"
    assert run["ParentSpanId"] == ""  # an execution is a trace of its own
    [work] = named(spans, "app.work")
    [in_thread] = named(spans, "app.in_thread")
    assert work["TraceId"] == in_thread["TraceId"] == run["TraceId"]
    assert work["ParentSpanId"] == run["SpanId"] and in_thread["ParentSpanId"] == work["SpanId"]
    assert work["SpanAttributes"]["input.value"] == '{"question": "hi"}'


async def test_a_step_is_a_span_and_a_replayed_step_is_none(traced):
    engine, store = traced
    ref = await engine.start("stepper", "go")
    await engine.settle()
    await engine.answer(ref, "yes")
    await engine.settle()
    assert engine.run(ref).output == "11yes"
    spans = await spans_of(engine, store, ref.id)
    runs = named(spans, "llamphouse.run")
    assert len(runs) == 2 and len({r["TraceId"] for r in runs}) == 2  # executed twice: two traces, one run
    assert [r["SpanAttributes"]["llamphouse.run.status"] for r in runs] == ["input_required", "completed"]
    assert [r["SpanAttributes"]["llamphouse.run.replayed_steps"] for r in runs] == [0, 1]
    steps = named(spans, "llamphouse.step")
    assert [s["SpanAttributes"]["llamphouse.step.name"] for s in steps] == ["first", "second"]  # "first" once
    for s, r in zip(steps, runs):
        assert s["ParentSpanId"] == r["SpanId"] and s["SpanAttributes"]["run.id"] == ref.id
    assert [s["ParentSpanId"] for s in named(spans, "app.in_step")] == [s["SpanId"] for s in steps]


async def test_a_failed_run_is_an_error_span(traced):
    engine, store = traced
    ref = await engine.start("boom", "go")
    await engine.settle()
    [run] = named(await spans_of(engine, store, ref.id), "llamphouse.run")
    assert run["StatusCode"] == "STATUS_CODE_ERROR" and "boom" in run["SpanAttributes"]["llamphouse.run.error"]
    assert "exception" in run["Events.Name"]


async def test_a_session_revision_is_a_span(traced):
    engine, store = traced
    ref = await engine.start("reviser", "go")
    await engine.settle()
    engine.tracing.flush()
    session_id = engine.run(ref).session_id
    [revise] = [s for s in store._all_spans if s["SpanName"] == "llamphouse.session.revise"
                and s["SpanAttributes"]["llamphouse.session.id"] == session_id]
    attrs = revise["SpanAttributes"]
    assert attrs["thread.id"] == engine.run(ref).thread_id and attrs["llamphouse.session.reason"] == "requested"
    assert attrs["llamphouse.session.revision"] == "declined"  # the default provider declines


def client(engine):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://test")


async def test_a_runs_trace_over_the_api(traced):
    engine, _ = traced
    async with client(engine) as c:
        t = (await c.post("/v2/threads", json={"agent": "traced"})).json()
        run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "hi"})).json()
        await engine.settle()
        got = (await c.get(f"/v2/runs/{run['id']}/trace")).json()
        assert got["run"] == run["id"] and got["tracing"] is True
        assert {"llamphouse.run", "app.work", "app.in_thread"} <= {s["SpanName"] for s in got["spans"]}
        assert all({"TraceId", "SpanId", "ParentSpanId", "Duration", "StatusCode", "SpanAttributes"} <= set(s)
                   for s in got["spans"])
        assert (await c.get("/v2/runs/run_nope/trace")).status_code == 404


async def test_without_a_tracing_store_runs_run_and_the_api_says_tracing_is_off():
    engine = Engine(AGENTS)
    async with engine:
        async with client(engine) as c:
            t = (await c.post("/v2/threads", json={"agent": "stepper"})).json()
            run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "go"})).json()
            await engine.settle()
            await engine.answer(RunRef(run["id"]), "yes")
            await engine.settle()
            assert (await c.get(f"/v2/runs/{run['id']}")).json()["output"] == "11yes"
            assert (await c.get(f"/v2/runs/{run['id']}/trace")).json() == {"run": run["id"], "tracing": False,
                                                                           "spans": []}


def test_without_tracing_set_up_the_engine_does_not_load_opentelemetry():
    code = (
        "import asyncio, sys\n"
        "from llamphouse.runtime import Engine\n"
        "from llamphouse.sdk import Agent\n"
        "class A(Agent):\n"
        "    async def run(self, ctx):\n"
        "        return await ctx.step('s', lambda: 1)\n"
        "async def main():\n"
        "    async with Engine({'a': A}) as e:\n"
        "        ref = await e.start('a', 'x')\n"
        "        await e.settle()\n"
        "        print(e.run(ref).status, 'opentelemetry' in sys.modules)\n"
        "asyncio.run(main())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["completed", "False"]
