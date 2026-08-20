# Example 10 — AsyncWorker vs DistributedWorker

Compares the two worker modes in llamphouse using the A2A streaming protocol:

| | AsyncWorker | DistributedWorker |
|---|---|---|
| **Processes** | All-in-one (API + worker) | API and worker(s) run separately |
| **Queue** | InMemoryQueue | Redis Streams |
| **Scaling** | Single process only | Add more worker processes |
| **Crash recovery** | Runs lost on crash | Redis auto-reclaims unfinished runs |
| **Redis required** | No | Yes |

## Architecture

```
AsyncWorker (all-in-one):
+----------------------------+
|  API Server + AsyncWorker  |   <- single process
+----------------------------+

DistributedWorker (split):
+--------------+       +-------+       +------------------+
|  API Server  |------>| Redis |<------|  Worker Process  |
|  (API only)  |       |       |       |  (concurrency=10)|
+--------------+       +-------+       +------------------+
                                       +------------------+
                                 <---- |  Worker Process 2 |  <- scale out
                                       +------------------+
```

## Quick Start

Run commands from this directory:

```bash
cd examples/10_DistributedWorker
```

### Prerequisites

```bash
pip install -r requirements.txt

# Redis (only needed for distributed mode)
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

If the Redis container already exists:

```bash
docker start redis
```

### Run Server

Async mode does not need Redis:

```bash
python server.py --mode async --port 8000
```

Distributed mode needs Redis:

```bash
python server.py --mode distributed --port 8100
```

### Automated Comparison

Start both server modes in separate terminals first:

```bash
python server.py --mode async --port 8000
python server.py --mode distributed --port 8100
```

Then run the comparison in another terminal:

```bash
python client.py               # 10 concurrent runs (default)
python client.py --runs 20     # more runs
```

### Manual Mode

You can also benchmark each mode independently:

```bash
# AsyncWorker (no Redis needed)
python server.py --mode async --port 8000
python client.py --port 8000    # in another terminal

# DistributedWorker (needs Redis)
python server.py --mode distributed --port 8100
python client.py --port 8100    # in another terminal
```

For a real split-process setup, use Postgres so the API process and worker
process share run state:

```bash
set REDIS_URL=redis://localhost:6379/0
set DATA_STORE=postgres
set DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/llamphouse

alembic upgrade head

python server.py --mode distributed --port 8000  # API process
python worker.py                                 # worker process
```

`DATA_STORE=memory` is useful for the local all-in-one comparison only. It
should not be used when API and worker processes are separated.

## Expected Output

```
============================================================
  ASYNC WORKER (all-in-one)
============================================================
  Runs        : 10
  Succeeded   : 10
  Wall time   : 1.45s
  Throughput  : 6.9 runs/s

============================================================
  DISTRIBUTED WORKER (API + worker)
============================================================
  Runs        : 10
  Succeeded   : 10
  Wall time   : 1.52s
  Throughput  : 6.6 runs/s

============================================================
  COMPARISON SUMMARY
============================================================
  AsyncWorker       : 1.45s  (6.9 runs/s)
  DistributedWorker : 1.52s  (6.6 runs/s)

  Note: For async I/O tasks like this, both perform similarly.
  The DistributedWorker shines when you need to:
    - Scale workers horizontally (multiple processes/machines)
    - Isolate the API from heavy compute (CPU-bound models)
    - Survive worker crashes (Redis auto-reclaims unfinished runs)
```

## Scaling Workers

Run multiple worker processes to scale horizontally:

```bash
python worker.py --concurrency 5    # terminal A
python worker.py --concurrency 5    # terminal B
```

Both workers share the load via Redis consumer groups. If one crashes,
the other automatically picks up its unfinished runs.

