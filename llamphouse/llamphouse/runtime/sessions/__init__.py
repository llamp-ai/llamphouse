"""Sessions: an agent's working memory on a thread, shaped by a provider.

    base.py             SessionProvider (the contract), SessionInfo, Revise
    rolling_window.py   RollingWindow - the default: the most recent messages
    summarizing.py      Summarizing - older messages become a summary (your function)

The runtime file names the default (`sessions: {rolling_window: {size: 50}}`); an agent file its own.
"""

from .base import Revise, SessionInfo, SessionProvider
from .rolling_window import RollingWindow
from .summarizing import Summarize, Summarizing

__all__ = ["SessionProvider", "SessionInfo", "Revise", "RollingWindow", "Summarizing", "Summarize"]
