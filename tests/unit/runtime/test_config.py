"""Deployment files: agent files and a runtime file, every component swappable; the `llamphouse` command."""

import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from llamphouse.runtime.cli import main as cli
from llamphouse.runtime.config import ConfigError, Deployment
from llamphouse.runtime.queue import MemoryQueue
from llamphouse.runtime.server import build
from llamphouse.runtime.sessions import RollingWindow, Summarizing
from llamphouse.runtime.store import MemoryStore
from llamphouse.sdk import Sender

HERE = Path(__file__).parent / "deploy"
POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")

AGENT_FILE = f"""
apiVersion: llamphouse.io/v1alpha1
kind: AgentDefinition
metadata: {{name: hr-assistant}}
spec:
  entrypoint: {HERE}/hr_agents.py:Hr
---
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {{name: hr-thailand}}
spec:
  definition: hr-assistant@1.2
  display_name: HR Thailand
  config: {{index: "${{HR_INDEX:-hr-th}}", tone: warm}}
  session:
    summarizing: {{summarize: {{import: hr_agents:summarize}}, max_messages: 4, keep: 2}}
  env: {{HR_MODEL: small}}
  secrets: {{HR_TOKEN: hr-token}}
---
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {{name: hr-plain}}
spec:
  entrypoint: {HERE}/hr_agents.py:Hr
"""

RUNTIME_FILE = """
apiVersion: llamphouse.io/v1alpha1
kind: Runtime
data_store: {in_memory: {}}
run_queue: in_memory
file_store: {hr_agents:Shout: {upload_ttl: 60}}
sessions: {rolling_window: {size: 7}}
engine: {workers: 3, lease_ttl: 20}
env: {HR_LOG: debug}
secrets_store:
  hr-token: {provider: env, name: HR_TOKEN_SOURCE}
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


async def test_agent_files_and_a_runtime_file_make_an_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HR_TOKEN_SOURCE", "s3cret")
    agents = write(tmp_path, "agents.yaml", AGENT_FILE)
    runtime = write(tmp_path, "llamphouse.yaml", RUNTIME_FILE)
    found = Deployment.load(runtime, [str(agents)])
    assert set(found.agents) == {"hr-thailand", "hr-plain"}
    assert found.agents["hr-thailand"].display_name == "HR Thailand"
    assert isinstance(found.agents["hr-thailand"].session, Summarizing)
    engine = found.build()
    assert engine.n_workers == 3 and engine.lease_ttl.total_seconds() == 20
    assert isinstance(engine.store, MemoryStore) and isinstance(engine.queue, MemoryQueue)
    assert type(engine.file_store).__name__ == "InMemoryFileStore"  # the custom component, by import path
    assert isinstance(engine._sessions["hr-plain"], RollingWindow) and engine._sessions["hr-plain"].size == 7
    assert os.environ["HR_MODEL"] == "small" and os.environ["HR_TOKEN"] == "s3cret" and os.environ["HR_LOG"] == "debug"
    async with engine:
        for name, expected in (("hr-thailand", "hr-th/warm"), ("hr-plain", "default/plain")):
            run = await engine.start(name, "hi", by=Sender.person("an"))
            for _ in range(100):
                async with engine.reading(run.id):
                    if engine.records[run.id].output is not None:
                        break
                await __import__("asyncio").sleep(0.01)
            assert engine.records[run.id].output == expected


@pytest.mark.parametrize("text, message", [
    ("kind: Runtime\ndata_store: {mongo: {}}", "unknown data_store 'mongo'"),
    ("kind: Runtime\ndata_store: {in_memory: {color: red}}", "unknown option(s) ['color']"),
    ("kind: Runtime\nqueue: {in_memory: {}}", "unknown key(s) ['queue']"),
    ("kind: Runtime\nengine: {wrokers: 2}", "unknown option(s) ['wrokers']"),
    ("kind: Agent\nmetadata: {name: Bad Name}\nspec: {entrypoint: x:y}", "lowercase slug"),
    ("kind: Agent\nmetadata: {name: a}\nspec: {definition: nope}", "no AgentDefinition 'nope'"),
    ("kind: Agent\nmetadata: {name: a}\nspec: {entrypoint: no_such_module:X}", "can't import 'no_such_module'"),
    ("kind: Agent\nmetadata: {name: a}\nspec: {entrypoint: x:y, colour: 1}", "unknown key(s) ['colour']"),
    ("kind: Thing", "unknown kind 'Thing'"),
    ("apiVersion: llamphouse.io/v9\nkind: Runtime", "apiVersion"),
])
def test_mistakes_in_the_files_are_named(tmp_path, text, message):
    path = write(tmp_path, "bad.yaml", text)
    with pytest.raises(ConfigError, match=None) as e:
        Deployment.load(path).build() if "kind: Runtime" in text else Deployment.load(None, [str(path)])
    assert message in str(e.value)


def test_a_config_that_doesnt_fit_the_agent_is_named(tmp_path):
    path = write(tmp_path, "a.yaml", f"""
        kind: Agent
        metadata: {{name: a}}
        spec: {{entrypoint: {HERE}/hr_agents.py:Hr, config: {{indx: x}}}}
    """)
    with pytest.raises(ConfigError, match="config"):
        Deployment.load(None, [str(path)]).build()


def test_an_unset_variable_without_a_default_is_named(tmp_path):
    path = write(tmp_path, "r.yaml", "kind: Runtime\ndata_store: {postgres: {dsn: '${NO_SUCH_VAR_HERE}'}}")
    with pytest.raises(ConfigError, match="NO_SUCH_VAR_HERE"):
        Deployment.load(path).build()


def test_a_python_mapping_and_roles(tmp_path):
    sys.path.insert(0, str(HERE))
    engine = build("hr_agents:AGENTS")
    assert set(engine._agents) == {"py-hr"} and engine.n_workers == 4
    for role in ("api", "worker"):  # in memory, nothing would ever reach the other processes
        with pytest.raises(ConfigError, match="shares work"):
            build("hr_agents:AGENTS", role=role)
    from llamphouse.runtime.bus import MemoryBus

    shared = {"store": MemoryStore(), "queue": MemoryQueue(), "bus": MemoryBus()}
    with pytest.raises(ConfigError):  # in-memory ones are never shared, given or not
        build("hr_agents:AGENTS", role="api", **shared)


def test_check_says_what_there_is(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HR_TOKEN_SOURCE", "x")
    agents = write(tmp_path, "agents.yaml", AGENT_FILE)
    runtime = write(tmp_path, "llamphouse.yaml", RUNTIME_FILE)
    with pytest.raises(SystemExit) as done:
        cli(["check", "--config", str(runtime), "--agents", str(agents)])
    assert done.value.code == 0
    out = capsys.readouterr().out
    assert "hr-thailand" in out and "session=Summarizing" in out and "data_store InMemory" not in out and "ok" in out
    bad = write(tmp_path, "bad.yaml", "kind: Runtime\ndata_store: {mongo: {}}")
    with pytest.raises(SystemExit) as failed:
        cli(["check", "--config", str(bad), "--agents", str(agents)])
    assert failed.value.code == 2 and "mongo" in capsys.readouterr().err


@pytest.mark.skipif(not POSTGRES, reason="no Postgres")
def test_db_upgrade_and_current(capsys):
    cli(["db", "upgrade", "--database-url", POSTGRES])
    cli(["db", "current", "--database-url", POSTGRES])
    assert "e7a1c2d3f4b5" in capsys.readouterr().out
    cli(["db", "upgrade", "--database-url", POSTGRES, "--sql"])
    assert "llamphouse_runs" in capsys.readouterr().out


def test_up_runs_a_process_and_the_1x_form_still_works(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "PYTHONPATH": f"{HERE}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    env.pop("LLAMPHOUSE_DATABASE_URL", None)
    proc = subprocess.Popen([sys.executable, "-m", "llamphouse.runtime", "hr_agents:AGENTS", "--memory",
                             "--port", str(port), "--log-level", "warning"], env=env, cwd=tmp_path)
    try:
        for _ in range(100):
            try:
                health = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).json()
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise AssertionError("did not start")
        assert health["ok"] and health["workers"] == 4
        assert [a["name"] for a in httpx.get(f"http://127.0.0.1:{port}/v2/agents").json()] == ["py-hr"]
    finally:
        proc.terminate()
        proc.wait(10)
