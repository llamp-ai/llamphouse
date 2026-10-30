"""The agents this deployment serves: the research desk (examples/v2/01_ResearchDesk).

    python -m llamphouse.runtime app:AGENTS --role api              # the HTTP API and Compass
    python -m llamphouse.runtime app:AGENTS --role worker           # as many as you like
    python -m llamphouse.runtime app:AGENTS --role all --memory     # one process, nothing to install
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_ResearchDesk"))
os.environ.setdefault("LLAMPHOUSE_DATABASE_URL", "postgresql://llamphouse:llamphouse@localhost:5433/llamphouse")
os.environ.setdefault("LLAMPHOUSE_REDIS_URL", "redis://localhost:6380/0")

from agents import AGENTS  # noqa: E402

__all__ = ["AGENTS"]
