from llamphouse.core import Agent
from llamphouse.core.context import Context
from llamphouse.core.types.config import StringParam


class ConfigurableAgent(Agent):
    config = [
        StringParam(
            key="tone",
            label="Tone",
            default="neutral",
            description="Response tone.",
        ),
        StringParam(
            key="label",
            label="Label",
            default="default-label",
            description="Runtime label.",
        ),
    ]

    async def run(self, context: Context):
        cfg = context.get_config()
        await context.insert_message(f"{cfg.get('label')}:{cfg.get('tone')}")


async def function_agent(context: Context):
    await context.insert_message("function-agent")


class FactoryAgent(Agent):
    async def run(self, context: Context):
        cfg = context.get_config()
        await context.insert_message(f"factory:{cfg.get('label')}")


def create_agent(deployment_cfg: dict) -> Agent:
    config = deployment_cfg.get("config", {})
    agent = FactoryAgent(
        id=deployment_cfg["name"],
        name=f"factory-{deployment_cfg['name']}",
    )
    agent.settings = config
    return agent
