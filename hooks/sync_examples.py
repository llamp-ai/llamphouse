"""
MkDocs hook — keep docs/examples.md in sync with the examples/ directory.

Runs as ``on_pre_build`` so the file is always up-to-date before MkDocs reads it.
Also fixes any stale "Example N —" heading numbers in individual README files.

Can also be run standalone:
    python hooks/sync_examples.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
DOCS_FILE = REPO_ROOT / "docs" / "examples.md"
GITHUB_BASE = "https://github.com/llamp-ai/llamphouse/tree/main/examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numbered_examples() -> list[Path]:
    """Return numbered example dirs sorted by their leading number."""
    def sort_key(p: Path) -> int:
        m = re.match(r"^(\d+)", p.name)
        return int(m.group(1)) if m else 9999

    return sorted(
        [p for p in EXAMPLES_DIR.iterdir() if p.is_dir() and re.match(r"^\d+", p.name)],
        key=sort_key,
    )


def parse_readme(path: Path) -> dict:
    """Extract title, one-line description, and key-feature bullets."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Title: first # heading, strip leading emoji and stale "Example N —" prefix
    title = ""
    for line in lines:
        if line.startswith("# "):
            raw = line[2:].strip()
            raw = re.sub(r"^(?:Example\s+)?\d+\s*[—–-]+\s*", "", raw)
            title = raw
            break

    # Description: first non-empty non-heading paragraph line after the title
    description = ""
    in_body = False
    for line in lines:
        if line.startswith("# "):
            in_body = True
            continue
        if not in_body:
            continue
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            clean = re.sub(r"\*+", "", stripped)
            clean = re.sub(r"`([^`]+)`", r"\1", clean)
            if len(clean) > 120:
                clean = clean[:117].rstrip() + "…"
            description = clean
            break

    # Key features: bullets under "What you'll learn / What it shows / What to expect"
    features: list[str] = []
    capture = False
    for line in lines:
        if re.search(r"what (you.ll learn|it shows|to expect)", line, re.I):
            capture = True
            continue
        if capture:
            if line.startswith("#"):
                break
            m = re.match(r"^\s*[-*]\s+(.+)", line)
            if m:
                features.append(re.sub(r"\*+", "", m.group(1)).strip())
                if len(features) == 3:
                    break

    return {"title": title, "description": description, "features": features}


def fix_readme_heading(readme: Path, number: int) -> bool:
    """Fix stale "Example N —" or bare "N —" prefix in the h1. Returns True if changed."""
    text = readme.read_text(encoding="utf-8")
    new_text = re.sub(
        r"^(# )(?:Example\s+)?\d+\s*[—–-]+\s*",
        rf"\1Example {number:02d} — ",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if new_text != text:
        readme.write_text(new_text, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# Generate docs/examples.md
# ---------------------------------------------------------------------------

def build_examples_md(examples: list[Path]) -> str:
    rows: list[str] = []
    progression_links: list[str] = []

    for i, ex_dir in enumerate(examples, start=1):
        readme_path = ex_dir / "README.md"
        if not readme_path.exists():
            continue

        info = parse_readme(readme_path)
        title = info["title"] or ex_dir.name
        desc = info["description"] or ""
        features = info["features"]

        def feat_label(f: str) -> str:
            ticks = re.findall(r"`([^`]+)`", f)
            if ticks:
                return f"`{ticks[0]}`"
            shortened = re.sub(r"(?i)^how\s+to\s+", "", f).strip()
            if len(shortened) > 30:
                shortened = shortened[:27].rstrip() + "…"
            return f"`{shortened}`"

        link = f"[{ex_dir.name}]({GITHUB_BASE}/{ex_dir.name})"
        key_feat = ", ".join(feat_label(f) for f in features[:2]) if features else ""
        rows.append(f"| {link} | {desc} | {key_feat} |")
        progression_links.append(
            f"{i}. **[{ex_dir.name}]({GITHUB_BASE}/{ex_dir.name})** — {title}"
        )

    table = (
        "| Example | Description | Key features |\n"
        "|---|---|---|\n"
        + "\n".join(rows)
    )
    progression = "\n".join(progression_links)

    return f"""\
# Examples

The [examples/]({GITHUB_BASE}) directory contains runnable samples for every major feature.
Each example includes an `agents.py` (or `server.py`), `client.py`, and `README.md` with instructions.

## Example index

{table}

## Running an example

Most examples follow the same pattern:

```bash
# Navigate to the example
cd examples/01_HelloWorld

# Install dependencies
pip install -r requirements.txt

# Start the server
llamphouse up   # or: python server.py

# In another terminal, run the client
python client.py
```

Some examples require environment variables (e.g., `OPENAI_API_KEY`). Check each example's `README.md` for specific instructions.

## Progression guide

If you're new to LLAMPHouse, we recommend working through the examples in this order:

{progression}

## Next steps

- [Quickstart](getting-started/quickstart.md) — build your first agent from scratch
- [Core Concepts](concepts/agents.md) — understand the fundamentals
- [Guides](guides/streaming.md) — deep dives into specific features
"""


# ---------------------------------------------------------------------------
# Main sync function (shared by hook + CLI script)
# ---------------------------------------------------------------------------

def sync(verbose: bool = False) -> None:
    examples = numbered_examples()
    if not examples:
        print("sync_examples: no numbered example directories found.", file=sys.stderr)
        return

    fixed: list[Path] = []
    for ex_dir in examples:
        num = int(re.match(r"^(\d+)", ex_dir.name).group(1))
        readme = ex_dir / "README.md"
        if readme.exists() and fix_readme_heading(readme, num):
            fixed.append(readme.relative_to(REPO_ROOT))

    DOCS_FILE.write_text(build_examples_md(examples), encoding="utf-8")

    if verbose:
        print(f"  sync_examples: docs/examples.md updated ({len(examples)} examples)")
        if fixed:
            print("  sync_examples: fixed headings in " + ", ".join(str(p) for p in fixed))


# ---------------------------------------------------------------------------
# MkDocs hook
# ---------------------------------------------------------------------------

def on_pre_build(config, **kwargs):
    """Regenerate docs/examples.md before MkDocs reads the docs directory."""
    sync(verbose=True)


if __name__ == "__main__":
    sync(verbose=True)
