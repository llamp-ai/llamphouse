"""
Speed-test client for Example 10.

Compares the AsyncWorker vs DistributedWorker by benchmarking servers
you start yourself.  Uses the A2A streaming protocol.

Usage
─────
  1. Start both servers in separate terminals:
       python server.py --mode async --port 8000
       python server.py --mode distributed --port 8100

  2. Run the comparison:
       python client.py                     # compares 8000 vs 8100
       python client.py --runs 20           # 20 runs per mode
       python client.py --port 8000         # benchmark a single server
"""

import argparse
import asyncio
import time
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    Message,
    Part,
    TextPart,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)


async def send_and_collect(client, idx: int) -> dict:
    """Send a message via A2A streaming and return timing + reply."""
    t0 = time.perf_counter()

    msg = Message(
        messageId=uuid4().hex,
        role=Role.user,
        parts=[Part(root=TextPart(text=f"Hello from run {idx}"))],
    )

    reply = ""
    status = "unknown"

    async for event in client.send_message(msg):
        if isinstance(event, tuple):
            _task, streaming_event = event

            if isinstance(streaming_event, TaskArtifactUpdateEvent):
                for part in streaming_event.artifact.parts:
                    if hasattr(part.root, "text") and part.root.text:
                        reply += part.root.text

            elif isinstance(streaming_event, TaskStatusUpdateEvent):
                if streaming_event.final:
                    status = streaming_event.status.state.value

    elapsed = time.perf_counter() - t0
    return {"idx": idx, "status": status, "elapsed": elapsed, "reply": reply}


async def run_concurrent(base_url: str, n: int) -> list[dict]:
    """Fire N concurrent A2A requests and collect results."""
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        card = await resolver.get_agent_card()

        factory = ClientFactory(
            ClientConfig(httpx_client=httpx_client, streaming=True)
        )
        client = factory.create(card)

        tasks = [send_and_collect(client, i) for i in range(n)]
        return await asyncio.gather(*tasks)


def print_results(label: str, results: list[dict], wall_time: float):
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")
    succeeded = sum(1 for r in results if r["status"] == "completed")
    failed = len(results) - succeeded
    per_run = [r["elapsed"] for r in results]
    print(f"  Runs        : {len(results)}")
    print(f"  Succeeded   : {succeeded}")
    if failed:
        print(f"  Failed      : {failed}")
    print(f"  Wall time   : {wall_time:.2f}s")
    print(f"  Avg per run : {sum(per_run) / len(per_run):.2f}s")
    print(f"  Min per run : {min(per_run):.2f}s")
    print(f"  Max per run : {max(per_run):.2f}s")
    print(f"  Throughput  : {len(results) / wall_time:.1f} runs/s")
    print(f"{'─' * 60}")
    for r in sorted(results, key=lambda x: x["idx"]):
        icon = "✓" if r["status"] == "completed" else "✗"
        print(f"  [{icon}] run {r['idx']:>2d}  {r['elapsed']:.2f}s  {r['reply']}")
    print()
    return wall_time


async def run_benchmark(port: int, n: int, label: str) -> float | None:
    """Fire n concurrent runs via A2A and print results. Returns wall time."""
    base_url = f"http://127.0.0.1:{port}"
    print(f"\n⏳ Running {n} concurrent runs against {label} (port {port})...")
    try:
        t0 = time.perf_counter()
        results = await run_concurrent(base_url, n)
        wall = time.perf_counter() - t0
        return print_results(label, results, wall)
    except Exception as e:
        print(f"\n  ✗ Could not connect to port {port}: {e}")
        return None


async def async_main():
    parser = argparse.ArgumentParser(description="Speed test: AsyncWorker vs DistributedWorker")
    parser.add_argument("--runs", type=int, default=10, help="Number of concurrent runs (default: 10)")
    parser.add_argument("--port", type=int, default=None,
                        help="Benchmark a single server on this port")
    parser.add_argument("--async-port", type=int, default=8000,
                        help="Port for AsyncWorker server (default: 8000)")
    parser.add_argument("--distributed-port", type=int, default=8100,
                        help="Port for DistributedWorker server (default: 8100)")
    args = parser.parse_args()

    # ── Single server mode ──────────────────────────────────────────────────────
    if args.port:
        await run_benchmark(args.port, args.runs, f"Server on port {args.port}")
        return

    # ── Comparison mode ─────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        AsyncWorker  vs  DistributedWorker               ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Make sure both servers are running:                     ║")
    print(f"║    Terminal 1: python server.py --mode async --port {args.async_port:<5}║")
    print(f"║    Terminal 2: python server.py --mode distributed \\    ║")
    print(f"║                  --port {args.distributed_port:<5}                            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    async_wall = await run_benchmark(args.async_port, args.runs, "ASYNC WORKER (all-in-one)")
    distributed_wall = await run_benchmark(args.distributed_port, args.runs,
                                            "DISTRIBUTED WORKER (Redis queue)")

    # ── Summary ─────────────────────────────────────────────────────────────────
    if async_wall or distributed_wall:
        print("═" * 60)
        print("  COMPARISON SUMMARY")
        print("═" * 60)
        if async_wall:
            print(f"  AsyncWorker       : {async_wall:.2f}s  ({args.runs / async_wall:.1f} runs/s)")
        if distributed_wall:
            print(f"  DistributedWorker : {distributed_wall:.2f}s  ({args.runs / distributed_wall:.1f} runs/s)")
        if async_wall and distributed_wall:
            ratio = async_wall / distributed_wall
            if ratio > 1:
                print(f"\n  ⚡ DistributedWorker was {ratio:.1f}x faster")
            elif ratio < 1:
                print(f"\n  ℹ️  AsyncWorker was {1/ratio:.1f}x faster (expected for small workloads)")
            else:
                print(f"\n  ≈  Both performed similarly")
            print()
            print("  Note: For async I/O tasks like this, both perform similarly.")
            print("  The DistributedWorker shines when you need to:")
            print("    • Scale workers horizontally (multiple processes/machines)")
            print("    • Isolate the API from heavy compute (CPU-bound models)")
            print("    • Survive worker crashes (Redis auto-reclaims unfinished runs)")
        print()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()