"""A2A client that intentionally creates a failed runtime trace."""

import asyncio
import os
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    Message,
    MessageSendConfiguration,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from dotenv import load_dotenv
from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.re import parse_env_headers

load_dotenv(override=True)

resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "my-client-error")})
provider = TracerProvider(resource=resource)

endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
if endpoint and not endpoint.rstrip("/").endswith("/v1/traces"):
    endpoint = endpoint.rstrip("/") + "/v1/traces"
headers_raw = (
    os.getenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
    or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    or ""
)
headers = parse_env_headers(headers_raw) if headers_raw else None
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint or None, headers=headers))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my-client-error")
BASE_URL = "http://127.0.0.1:8000"


async def main():
    # This marker is consumed by the example agent and makes the provider
    # call use an invalid model.  The provider failure is then recorded by
    # Llamphouse as an error trace.
    question = "SPOTLIGHT_TRIGGER_AI_ERROR"
    failed = False
    run_id = None

    with tracer.start_as_current_span("my-client-error.run") as span:
        carrier: dict[str, str] = {}
        propagate.inject(carrier)

        async with httpx.AsyncClient(headers=carrier) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=BASE_URL,
            )
            card = await resolver.get_agent_card()

            factory = ClientFactory(
                ClientConfig(httpx_client=httpx_client, streaming=True)
            )
            client = factory.create(card)

            print(f"  User: {question}")
            print("  Expected: failed task")

            message = Message(
                messageId=uuid4().hex,
                role=Role.user,
                parts=[Part(root=TextPart(text=question))],
            )
            configuration = MessageSendConfiguration(
                acceptedOutputModes=["text"],
            )

            async for event in client.send_message(
                message,
                configuration=configuration,
            ):
                if not isinstance(event, tuple):
                    continue

                task, result = event
                run_id = run_id or getattr(task, "id", None)
                if isinstance(result, TaskStatusUpdateEvent):
                    state = getattr(result.status.state, "value", result.status.state)
                    state = str(state).lower()
                    if state == TaskState.failed.value:
                        failed = True
                        print(f"  [status: {state}]")
                elif isinstance(result, TaskArtifactUpdateEvent):
                    # A failure should not produce an artifact, but keep this
                    # client tolerant of a provider-specific event sequence.
                    for part in result.artifact.parts:
                        text = getattr(part.root, "text", None)
                        if text:
                            print(text, end="", flush=True)

            # Some A2A client versions do not surface the final task state in
            # the streamed event sequence. The task id is also the Llamphouse
            # run id, so verify the authoritative error projection through
            # the example's Spotlight route.
            if not failed and run_id:
                trace_url = (
                    f"{BASE_URL}/spotlight/agents/my-assistant/traces"
                )
                for _ in range(20):
                    response = await httpx_client.get(trace_url)
                    if response.is_success:
                        traces = response.json().get("data", [])
                        if any(
                            trace.get("execution_id") == run_id
                            and trace.get("status") == "error"
                            for trace in traces
                        ):
                            failed = True
                            print("\n  [Spotlight trace status: error]")
                            break
                    await asyncio.sleep(0.5)

        if not failed:
            span.set_status(Status(StatusCode.ERROR))
            raise RuntimeError(
                "server did not produce a failed A2A task or error Spotlight trace; "
                "restart examples/16_Spotlight/server.py and try again"
            )

        span.set_status(Status(StatusCode.ERROR))
        print("  Confirmed: A2A task failed; inspect its Spotlight trace.")


if __name__ == "__main__":
    asyncio.run(main())
