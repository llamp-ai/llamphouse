# Example 10 - AsyncWorker vs DistributedWorker

Compares the two worker modes in LLAMPHouse using the A2A streaming protocol.
The client uses the current `a2a` SDK protobuf types for messages and `httpx`
to read the LLAMPHouse A2A JSON-RPC/SSE stream.

|                          | AsyncWorker               | DistributedWorker                   |
| ------------------------ | ------------------------- | ----------------------------------- |
| **Processes**      | All-in-one (API + worker) | API and worker(s) run separately    |
| **Queue**          | InMemoryQueue             | Redis Streams                       |
| **Scaling**        | Single process only       | Add more worker processes           |
| **Crash recovery** | Runs lost on crash        | Redis auto-reclaims unfinished runs |
| **Redis required** | No                        | Yes                                 |

## Architecture

```
AsyncWorker (all-in-one):
┌────────────────────────────┐
│  API Server + AsyncWorker  │   ← single process
└────────────────────────────┘

DistributedWorker (split):
┌──────────────┐       ┌───────┐       ┌──────────────────┐
│  API Server  │──────▶│ Redis │◀──────│  Worker Process  │
│  (API only)  │       │       │       │  (concurrency=10)│
└──────────────┘       └───────┘       └──────────────────┘
                                       ┌──────────────────┐
                                 ◀──── │  Worker Process 2 │  ← scale out
                                       └──────────────────┘
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt

# Optional for local runs when no OTel Collector is running on port 4318
export LLAMPHOUSE_TRACING_ENABLED=false

# Redis (only needed for distributed mode)
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### Side-by-Side Comparison

The client benchmarks both modes in one run, but **you must start both servers
yourself first** — the client only fires requests, it does not spawn servers.

Open three terminals:

```bash
# Terminal 1 — AsyncWorker on port 8000
uv run python server.py --mode async --port 8000

# Terminal 2 — DistributedWorker on port 8100 (needs Redis)
uv run python server.py --mode distributed --port 8100

# Terminal 3 — run the benchmark
uv run python client.py               # 10 concurrent runs (default)
uv run python client.py --runs 20     # more runs
```

### Single-Mode

Benchmark just one server:

```bash
# ── AsyncWorker (no Redis needed) ──
uv run python server.py --mode async --port 8000
uv run python client.py --port 8000    # in another terminal

# ── DistributedWorker (needs Redis) ──
uv run python server.py --mode distributed --port 8100
uv run python client.py --port 8100     # in another terminal
```

Both modes use `InMemoryDataStore`. In distributed mode, `server.py` starts an
in-process `DistributedWorker`, so the API and worker share the same store
object. A true split-process deployment needs a shared persistent store such as
Postgres; see the deployment guide for that setup.

## Expected Output

```
╔══════════════════════════════════════════════════════════╗
║        AsyncWorker  vs  DistributedWorker               ║
╚══════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════
  ASYNC WORKER (all-in-one)
════════════════════════════════════════════════════════════
  Runs        : 10
  Succeeded   : 10
  Wall time   : 1.45s
  Throughput  : 6.9 runs/s

════════════════════════════════════════════════════════════
  DISTRIBUTED WORKER (API + worker)
════════════════════════════════════════════════════════════
  Runs        : 10
  Succeeded   : 10
  Wall time   : 1.52s
  Throughput  : 6.6 runs/s

════════════════════════════════════════════════════════════
  COMPARISON SUMMARY
════════════════════════════════════════════════════════════
  AsyncWorker       : 1.45s  (6.9 runs/s)
  DistributedWorker : 1.52s  (6.6 runs/s)

  Note: For async I/O tasks like this, both perform similarly.
  The DistributedWorker shines when you need to:
    • Scale workers horizontally (multiple processes/machines)
    • Isolate the API from heavy compute (CPU-bound models)
    • Survive worker crashes (Redis auto-reclaims unfinished runs)
```

## Split-Process Workers

The included `worker.py` is a reference for the real split-process shape, but
this example intentionally keeps the local benchmark in one process with
`InMemoryDataStore`. When the API and worker are separate processes, use a
shared persistent data store so both processes can read the same runs.
