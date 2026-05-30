from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse import WebhookTrigger


class ReportAgent(Agent):
    """Agent that can be triggered via an external webhook.

    Accepts a JSON payload and logs a summary of what it received.
    In a real use-case this might send a Teams message, call an API,
    or kick off a background job.

    Trigger it with:
        curl -X POST http://127.0.0.1:8000/triggers/report \\
             -H "Authorization: Bearer supersecret" \\
             -H "Content-Type: application/json" \\
             -d '{"customer": "Acme Corp", "event": "trial_expired"}'
    """

    triggers = [
        WebhookTrigger(path="/triggers/report", secret_env="WEBHOOK_SECRET"),
    ]

    async def run(self, context: Context):
        if context.trigger is None:
            # Normal human-initiated run — shouldn't happen with this agent,
            # but handled gracefully just in case.
            await context.insert_message("I'm a webhook-driven agent. Trigger me via POST /triggers/report.")
            return

        data = context.trigger.data
        fired_at = context.trigger.fired_at

        customer = data.get("customer", "unknown customer")
        event = data.get("event", "unknown event")

        summary = (
            f"Webhook received at {fired_at}\n"
            f"Customer : {customer}\n"
            f"Event    : {event}\n"
            f"Payload  : {data}"
        )

        print(f"\n[ReportAgent] {summary}\n")

        # In a real agent you might call an external API here, e.g.:
        #   await send_teams_message(customer, event)

        await context.insert_message(summary)


def main():
    agent = ReportAgent(
        id="report-agent",
        name="Report Agent",
        description="Webhook-triggered agent that processes incoming events.",
        version="0.1.0",
    )

    llamphouse = LLAMPHouse(
        agents=[agent],
        data_store=InMemoryDataStore(),
        adapters=[],  # No A2A/HTTP chat interface needed — webhook only
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
