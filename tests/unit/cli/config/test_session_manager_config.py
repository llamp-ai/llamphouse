"""Session managers from llamphouse.yaml: the default for every agent
(top-level session_manager), one agent's own (agents[].session_manager),
the named callables, and what `llamphouse check` reports."""

from types import SimpleNamespace

import pytest

import llamphouse.cli as cli
from llamphouse.cli.config.loader import build_app_from_config, build_session_manager, load_config
from llamphouse.core.sessions import DEFAULT_SESSION_MANAGER, FullHistory, SlidingWindow, Summarizing, session_manager_of

pytestmark = [pytest.mark.unit]

AGENTS = '''
class Plain(Agent):
    async def run(self, context):
        pass

class Own(Agent):
    session_manager = FullHistory()

    async def run(self, context):
        pass

async def summarize(previous, messages):
    return "summary"

def words(text):
    return len(text.split())
'''


def write(tmp_path, yaml: str):
    (tmp_path / "agents.py").write_text(
        "from llamphouse.core import Agent, FullHistory\n" + AGENTS, encoding="utf-8",
    )
    path = tmp_path / "llamphouse.yaml"
    path.write_text(yaml, encoding="utf-8")
    return path


BASE = '''version: "0.1"
definitions:
  - name: plain
    entrypoint: agents.py:Plain
  - name: own
    entrypoint: agents.py:Own
agents:
  - name: plain
    definition: plain
  - name: own
    definition: own
  - name: tuned
    definition: plain
    session_manager:
      summarizing:
        summarizer: agents.py:summarize
        max_messages: 10
        count_tokens: agents.py:words
'''


def test_the_default_applies_to_agents_without_their_own(tmp_path):
    path = write(tmp_path, BASE + "session_manager:\n  sliding_window:\n    max_messages: 5\n    keep_first: 1\n")
    app = build_app_from_config(load_config(path), tmp_path)
    plain, own, tuned = app.agents

    assert isinstance(plain.session_manager, SlidingWindow)
    assert plain.session_manager.max_messages == 5 and plain.session_manager.keep_first == 1
    assert isinstance(own.session_manager, FullHistory)  # its own wins over the default
    assert isinstance(tuned.session_manager, Summarizing)  # its YAML entry wins too
    assert tuned.session_manager.max_messages == 10
    assert tuned.session_manager.count_tokens("a b c") == 3


def test_without_a_default_agents_get_the_latest_20_messages(tmp_path):
    app = build_app_from_config(load_config(write(tmp_path, BASE)), tmp_path)

    assert app.agents[0].session_manager is None
    assert session_manager_of(app.agents[0]) is DEFAULT_SESSION_MANAGER
    assert isinstance(DEFAULT_SESSION_MANAGER, SlidingWindow) and DEFAULT_SESSION_MANAGER.max_messages == 20


def test_openai_summarizers_by_name(tmp_path):
    plain = build_session_manager({"summarizing": {"summarizer": "openai", "max_messages": 8}}, tmp_path)
    tuned = build_session_manager(
        {"summarizing": {"summarizer": {"openai": {"model": "gpt-x", "max_words": 50}}}}, tmp_path,
    )
    assert callable(plain.summarizer) and callable(tuned.summarizer)


def test_bad_entries_say_what_is_wrong(tmp_path):
    with pytest.raises(ValueError, match="summarizer must be"):
        build_session_manager({"summarizing": {"summarizer": "gpt"}}, tmp_path)
    with pytest.raises(ValueError, match="does not accept"):
        build_session_manager({"sliding_window": {"window": 3}}, tmp_path)
    with pytest.raises(ValueError, match="Unknown session_manager"):
        build_session_manager({"rolling": {}}, tmp_path)


def test_llamphouse_check_reports_a_bad_session_manager(tmp_path, capsys):
    path = write(tmp_path, BASE + "session_manager:\n  sliding_window:\n    window: 3\n")

    exit_code = cli._cmd_check(SimpleNamespace(config=str(path), format="text", verbose=False, timeout=5.0))

    assert exit_code != 0
    assert "session_manager: Session_manager 'sliding_window' does not accept" in capsys.readouterr().out


def test_compacting_from_yaml(tmp_path):
    from llamphouse.core.sessions import Compacting

    write(tmp_path, BASE)
    manager = build_session_manager(
        {"compacting": {"summarizer": "agents.py:summarize", "max_tokens": 8000, "reserve_tokens": 2000,
                        "compact_at": 0.8, "compact_to": 0.3}},
        tmp_path,
    )
    assert isinstance(manager, Compacting) and manager.budget == 6000
    assert manager.compact_at == 0.8 and manager.summary_tokens == 600
