import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_hook(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_landing_page_hook_writes_utf8_root_index(tmp_path):
    landing_page = _load_hook("landing_page_hook", "hooks/landing_page.py")
    site_dir = tmp_path / "site" / "docs"

    landing_page.on_post_build({"site_dir": str(site_dir)})

    landing = tmp_path / "site" / "index.html"
    html = landing.read_text(encoding="utf-8")
    assert landing.exists()
    assert '<meta charset="UTF-8"' in html
    assert "LLAMPHouse" in html
    assert "./docs/" in html


def test_sync_examples_builds_docs_and_fixes_stale_headings(tmp_path, monkeypatch):
    sync_examples = _load_hook("sync_examples_hook", "hooks/sync_examples.py")
    examples_dir = tmp_path / "examples"
    docs_file = tmp_path / "docs" / "examples.md"
    example_a = examples_dir / "01_HelloWorld"
    example_b = examples_dir / "13_LLAMPHouseYAML"
    example_a.mkdir(parents=True)
    example_b.mkdir(parents=True)
    docs_file.parent.mkdir(parents=True)

    (example_a / "README.md").write_text(
        """# 99 - Hello World

Run the smallest useful LLAMPHouse agent.

## What you'll learn

- How to define `Agent`
- How to start `llamphouse up`
""",
        encoding="utf-8",
    )
    (example_b / "README.md").write_text(
        """# Example 01 - YAML Runtime

Configure multiple registered agents from YAML.

## What it shows

- `llamphouse.yaml`
- `ConfigStore`
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(sync_examples, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sync_examples, "EXAMPLES_DIR", examples_dir)
    monkeypatch.setattr(sync_examples, "DOCS_FILE", docs_file)
    monkeypatch.setattr(sync_examples, "GITHUB_BASE", "https://example.test/examples")

    sync_examples.sync(verbose=True)

    first_readme = (example_a / "README.md").read_text(encoding="utf-8")
    second_readme = (example_b / "README.md").read_text(encoding="utf-8")
    docs = docs_file.read_text(encoding="utf-8")

    assert first_readme.startswith("# Example 01")
    assert second_readme.startswith("# Example 13")
    assert "[01_HelloWorld](https://example.test/examples/01_HelloWorld)" in docs
    assert "[13_LLAMPHouseYAML](https://example.test/examples/13_LLAMPHouseYAML)" in docs
    assert "`Agent`" in docs
    assert "`llamphouse.yaml`" in docs
