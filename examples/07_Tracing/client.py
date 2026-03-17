"""
A2A client with end-to-end tracing.

The client creates a root span, injects the W3C trace-context headers into the
httpx client so that every server-side span is nested under the same trace.
"""

import asyncio
import os
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendStreamingMessageRequest,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from opentelemetry import propagate, trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.util.re import parse_env_headers
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Set up OpenTelemetry tracing ─────────────────────────────────────────
resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "my-client")})
provider = TracerProvider(resource=resource)

endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
if endpoint and not endpoint.rstrip("/").endswith("/v1/traces"):
    endpoint = endpoint.rstrip("/") + "/v1/traces"
headers_raw = os.getenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS") or os.getenv("OTEL_EXPORTER_OTLP_HEADERS") or ""
headers = parse_env_headers(headers_raw) if headers_raw else None

provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint or None, headers=headers)))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my-client")

BASE_URL = "http://127.0.0.1:8000"


async def main():
    with tracer.start_as_current_span("my-client.run") as span:
        # Inject W3C traceparent into default headers so every request
        # the A2A client makes is part of this trace.
        carrier: dict[str, str] = {}
        propagate.inject(carrier)

        async with httpx.AsyncClient(headers=carrier) as httpx_client:

            # ── 1. Discover agent card ───────────────────────────────────
            resolver = A2ACardResolver(
                httpx_client=httpx_client, base_url=BASE_URL
            )
            card = await resolver.get_agent_card()

            print("=== Agent Card ===")
            print(f"  Name:        {card.name}")
            print(f"  Description: {card.description}")
            print(f"  Streaming:   {card.capabilities.streaming}")
            print()

            client = A2AClient(httpx_client=httpx_client, agent_card=card)

            # ── 2. Stream a question ─────────────────────────────────────
            question = (
                "I need to solve the equation `x + 1 = 4`. Can you help me?"
            )
            print(f"  User: {question}")
            print("  Assistant: ", end="", flush=True)

            span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
            span.set_attribute("input.value", question)

            chunks: list[str] = []

            async for chunk in client.send_message_streaming(
                SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        message={
                            "role": "user",
                            "parts": [{"kind": "text", "text": question}],
                            "messageId": uuid4().hex,
                        }
                    ),
                )
            ):
                result = chunk.root.result
                if isinstance(result, TaskArtifactUpdateEvent):
                    for part in result.artifact.parts:
                        if hasattr(part.root, "text") and part.root.text:
                            chunks.append(part.root.text)
                            print(part.root.text, end="", flush=True)
                elif isinstance(result, TaskStatusUpdateEvent):
                    if (
                        result.final
                        and result.status.state.value != "completed"
                    ):
                        msg = ""
                        if result.status.message and result.status.message.parts:
                            for p in result.status.message.parts:
                                if hasattr(p.root, "text"):
                                    msg += p.root.text
                        print(
                            f"\n  [status: {result.status.state.value}] {msg}",
                            flush=True,
                        )

            print("\n")

            span.set_attribute("output.value", "".join(chunks))
            span.set_status(Status(StatusCode.OK))


if __name__ == "__main__":
    asyncio.run(main())