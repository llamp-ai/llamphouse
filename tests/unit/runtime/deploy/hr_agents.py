"""Agents for the deployment-file tests (imported by file path, as an agent file would)."""

from llamphouse.sdk import Agent, Config


class HrConfig(Config):
    index: str = "default"
    tone: str = "plain"


class Hr(Agent[HrConfig]):
    """Answers HR questions."""

    async def run(self, ctx) -> str:
        return f"{ctx.config.index}/{ctx.config.tone}"


def summarize(older) -> str:
    return f"({len(older)} earlier)"


class Shout:  # a custom component: a file store that is the in-memory one, by another name
    def __new__(cls, **kwargs):
        from llamphouse.runtime.files import InMemoryFileStore

        return InMemoryFileStore(**kwargs)


AGENTS = {"py-hr": Hr}
