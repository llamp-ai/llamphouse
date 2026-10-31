"""`llamphouse migrate`: a 1.x database's conversations as 2.0 threads - re-runnable, and the 2.0 agent
carries on with what was said in 1.x."""

import asyncio
import json
import os

import pytest

from llamphouse.runtime import Engine, migrations
from llamphouse.runtime.cli import main
from llamphouse.runtime.migrations.from_v1 import RUN, THREAD, migrate, parts
from llamphouse.sdk import Agent, TextPart, ThreadRef
from llamphouse.sdk.content import UnknownPart

POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")
postgres = pytest.mark.skipif(not POSTGRES, reason="no Postgres")


class Concierge(Agent):
    async def run(self, ctx) -> list[str]:
        return [f"{m.sender.kind}:{m.text}" for m in await ctx.session.messages()]


def test_1x_content_becomes_parts():
    text, image = parts(json.dumps([
        {"type": "text", "text": {"value": "see this", "annotations": [{"type": "file_citation"}]}},
        {"type": "image_file", "image_file": {"file_id": "file-1"}},
    ]))
    assert text == TextPart(text="see this", metadata={"annotations": [{"type": "file_citation"}]})
    assert isinstance(image, UnknownPart) and image.kind == "image_file" and image.raw["image_file"]["file_id"] == "file-1"
    assert parts(json.dumps("plain")) == [TextPart(text="plain")]  # 1.x kept some content as a JSON string


async def _v1_database():
    """An empty database as a 1.x deployment left it, with two conversations and one without runs."""
    import asyncpg

    con = await asyncpg.connect(POSTGRES)
    await con.execute("drop schema public cascade; create schema public")
    await con.close()
    await asyncio.to_thread(migrations.upgrade, POSTGRES, "v1@head")
    con = await asyncpg.connect(POSTGRES)
    await con.execute("""
        insert into threads (id, name, meta, created_at) values
            ('thread_a', 'Trip', '{"org": "acme"}', '2026-01-01 10:00+00'),
            ('thread_b', '', null, '2026-01-02 10:00+00'),
            ('thread_c', '', null, '2026-01-03 10:00+00');
        insert into messages (id, thread_id, role, content, status, created_at) values
            ('msg_1', 'thread_a', 'user', '[{"type": "text", "text": {"value": "Lisbon in May?", "annotations": []}}]',
             'completed', '2026-01-01 10:00:01+00'),
            ('msg_2', 'thread_a', 'assistant', '[{"type": "text", "text": {"value": "Mild, 22C.", "annotations": []}}]',
             'completed', '2026-01-01 10:00:05+00'),
            ('msg_3', 'thread_b', 'user', '[{"type": "text", "text": {"value": "hi", "annotations": []}}]',
             'completed', '2026-01-02 10:00:01+00'),
            ('msg_4', 'thread_c', 'user', '[{"type": "text", "text": {"value": "anyone?", "annotations": []}}]',
             'completed', '2026-01-03 10:00:01+00');
        update messages set assistant_id = 'trip-bot', run_id = 'run_1' where id = 'msg_2';
        insert into runs (id, thread_id, assistant_id, status, model, instructions, created_at, completed_at, usage) values
            ('run_1', 'thread_a', 'trip-bot', 'completed', 'gpt-x', 'Be brief.', '2026-01-01 10:00:02+00',
             extract(epoch from timestamptz '2026-01-01 10:00:06+00'), '{"total_tokens": 42}'),
            ('run_2', 'thread_b', 'helpdesk', 'in_progress', 'gpt-x', '', '2026-01-02 10:00:02+00', null, null);
        insert into run_steps (id, object, assistant_id, thread_id, run_id, type, status, step_details, created_at) values
            ('step_1', 'thread.run.step', 'trip-bot', 'thread_a', 'run_1', 'tool_calls', 'completed',
             '{"type": "tool_calls", "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": "{}", "output": "22C"}}]}',
             '2026-01-01 10:00:03+00');
    """)
    return con


@postgres
async def test_a_1x_database_migrates_and_the_agent_carries_on():
    from llamphouse.runtime.data_store.postgres import PostgresStore

    source = await _v1_database()
    store = PostgresStore(POSTGRES, min_size=1, max_size=2)
    engine = Engine({}, workers=0, store=store)
    await store.open()
    try:
        dry = await migrate(source, engine, agents={"trip-bot": "concierge"}, dry_run=True)
        assert (dry.threads, dry.messages, dry.runs) == (2, 3, 2) and dry.skipped == ["thread_c"]
        assert await store.tree_of("thread_a") is None  # a dry run writes nothing

        report = await migrate(source, engine, agents={"trip-bot": "concierge"})
        assert (report.threads, report.messages, report.runs) == (2, 3, 2)
        assert report.active == ["run_2"] and report.skipped == ["thread_c"]
        again = await migrate(source, engine, agents={"trip-bot": "concierge"})  # nothing new: nothing written
        assert (again.threads, again.updated, again.unchanged, again.messages, again.runs) == (0, 0, 2, 0, 0)

        # 1.x carried on: run_2 finished and said something; thread_c is given an agent
        await source.execute("""
            update runs set status = 'completed', completed_at = extract(epoch from timestamptz '2026-01-02 10:01+00')
                where id = 'run_2';
            insert into messages (id, thread_id, role, content, status, created_at, assistant_id) values
                ('msg_5', 'thread_b', 'assistant', '"hello!"', 'completed', '2026-01-02 10:00:30+00', 'helpdesk');
        """)
        later = await migrate(source, engine, agents={"trip-bot": "concierge"}, default_agent="concierge")
        assert (later.threads, later.updated, later.unchanged, later.messages, later.runs) == (1, 1, 1, 2, 1)
    finally:
        await source.close()
        await store.close()

    store = PostgresStore(POSTGRES, min_size=1, max_size=2)
    async with Engine({"concierge": Concierge, "helpdesk": Concierge}, store=store) as engine:
        async with engine.reading("thread_a"):
            t = engine.threads["thread_a"]
            assert t.agent == "concierge" and t.metadata == {"org": "acme"}
            timeline = engine.timeline(ThreadRef("thread_a"))
        kinds = [str(ev.kind) for ev in timeline]
        assert kinds == [THREAD, "message", "message", RUN]
        assert timeline[0].data == {"name": "Trip"}
        said = timeline[2].message
        assert said.id == "msg_2" and said.sender.id == "trip-bot" and said.metadata["llamphouse.v1"]["run_id"] == "run_1"
        assert timeline[1].at.isoformat().startswith("2026-01-01T10:00:01")
        run = timeline[3].data
        assert run["status"] == "completed" and run["usage"] == {"total_tokens": 42}
        assert run["steps"][0]["details"]["tool_calls"][0]["function"]["name"] == "weather"

        async with engine.reading("thread_b"):
            statuses = [ev.data["status"] for ev in engine.timeline(ThreadRef("thread_b")) if str(ev.kind) == RUN]
        assert statuses == ["in_progress", "completed"]  # its later status, recorded too

        ref = await engine.start("concierge", "and in June?", thread=ThreadRef("thread_a"))
        result = await asyncio.wait_for(_ended(engine, ref), 10)
        assert result == ["person:Lisbon in May?", "agent:Mild, 22C.", "service:and in June?"]


async def _ended(engine, ref):
    while str(engine.run(ref).status) not in ("completed", "failed"):
        await asyncio.sleep(0.05)
    rec = engine.run(ref)
    assert str(rec.status) == "completed", rec.error
    return rec.output


@postgres
async def test_the_command(capsys, tmp_path):
    source = await _v1_database()
    await source.close()
    charts = tmp_path / "compass_charts.json"
    charts.write_text(json.dumps([{"id": "c1", "title": "Runs", "sql": "select count(*) from runs"}]))
    with pytest.raises(SystemExit) as done:
        await asyncio.to_thread(main, ["migrate", "--database-url", POSTGRES, "--rename", "trip-bot=concierge", "--dry-run",
              "--charts", str(charts)])
    assert done.value.code == 0
    out = capsys.readouterr().out
    assert "would write 3 messages and 2 run records" in out and "1 threads without runs" in out
    assert "runs are still active in 1.x" in out and "dry run: nothing was written" in out
    with pytest.raises(SystemExit):
        await asyncio.to_thread(main, ["migrate", "--database-url", POSTGRES, "--agent", "concierge", "--charts", str(charts)])
    out = capsys.readouterr().out
    assert "3 new" in out and "wrote 4 messages" in out and "wrote 1 dashboards and charts" in out
    with pytest.raises(SystemExit) as bad:
        await asyncio.to_thread(main, ["migrate", "--database-url", POSTGRES, "--rename", "nope"])
    assert bad.value.code == 2
