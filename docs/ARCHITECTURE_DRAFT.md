# LLAMPHouse Platform Architecture — Draft

> **Status:** Draft · February 2026
> **Purpose:** Define the three pillars of the LLAMPHouse platform, the components
> that power each, and how they connect.

---

## Overview

LLAMPHouse is structured around **three pillars**, each serving a different persona:

| Pillar | Persona | Purpose |
|---|---|---|
| **Agent Runtime** | DevOps / Platform | Self-hosted, scalable infrastructure for running agents |
| **Spotlight** (Business Dashboard) | Product / Ops / CS | Conversation insights, analytics, data extraction |
| **Compass** (Developer Dashboard) | Developer / ML Eng | Tracing, evaluation, configuration, debugging |

```
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
  PRIVATE ZONE (customer VPC / on-prem)
│                                                                     │
  ┌─────────────────┐  ┌─────────────────┐
│ │  API Server      │  │  Workers (N)    │  ← same image, different cmd   │
  │  HTTP / SSE      │  │  Run execution  │
│ │  Route handling  │  │  LLM calls      │                                │
  │  Adapters        │  │  Tool calls     │
│ └────────┬─────────┘  └───────┬─────────┘                                │
           │                    │
│   ┌──────┴──────┐   ┌────────┴─┐                                         │
    │  Postgres   │   │  Redis   │   ← customer-managed or bundled
│   └─────────────┘   └──────────┘                                         │
└ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
             │ OTel export / API
┌ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
  PLATFORM ZONE (LLAMPHouse-managed or self-hosted)
│                                                                     │
  ┌──────────────────┐  ┌───────────────────┐
│ │ Spotlight        │  │ Compass           │                         │
  │  Analytics       │  │  Trace viewer     │
│ │  Conversation    │  │  Eval framework   │                         │
  │    search        │  │  Config store     │
│ │  Sentiment &     │  │  Playground       │                         │
  │    extraction    │  │  Run comparisons  │
│ │  Usage metrics   │  │  Prompt editor    │                         │
  └────────┬─────────┘  └────────┬──────────┘
│          │                     │                                    │
    ┌──────┴──────┐      ┌───────┴────────┐
│   │  Postgres   │      │  ClickHouse    │                           │
    │  (state +   │      │  (traces +     │
│   │   search)   │      │   analytics)   │                           │
    └─────────────┘      └────────────────┘
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

---

## Pillar 1: Agent Runtime

> *What you ship to production. Runs agents, handles traffic, scales horizontally.*

### What exists today

| Component | Implementation | Status |
|---|---|---|
| HTTP server | FastAPI + Uvicorn | ✅ Done |
| API surface | Assistants API adapter, A2A adapter | ✅ Done |
| Workers | AsyncWorker (in-process, concurrent) | ✅ Done |
| Data store | InMemoryDataStore, PostgresDataStore | ✅ Done |
| Run queue | InMemoryQueue | ✅ Done |
| Streaming | SSE via event queues | ✅ Done |
| Auth | Pluggable BaseAuth middleware | ✅ Done |
| Multi-agent | A2A protocol, orchestrator pattern | ✅ Done |
| Retention | Configurable purge policy | ✅ Done |
| Config store | InMemoryConfigStore | ✅ Done |
| Distributed queue | RedisQueue (Redis Streams with consumer groups) | ✅ Done |
| Distributed worker | DistributedWorker (Redis Streams consumer) | ✅ Done |
| Cross-worker streaming | RedisEventQueue (Redis Pub/Sub) | ✅ Done |
| CLI entrypoints | `llamphouse serve` / `llamphouse worker` / `serve --workers` | ✅ Done |
| Docker Compose | Runtime + Postgres + Redis (dev & prod) | ✅ Done |
| Queue-level rate limiting | Rate limit config on queue enqueue | ✅ Done |

### What to build for scale

| Component | Purpose | Recommended tech | Status |
|---|---|---|---|
| **Distributed queue** | Decouple API from workers; scale workers independently | **Redis Streams** | ✅ Done |
| **Distributed worker** | Run workers as separate processes/containers | `DistributedWorker` that consumes from Redis Streams | ✅ Done |
| **Streaming across workers** | SSE events from any worker back to the right API instance | **Redis Pub/Sub** as event bus between workers and API | ✅ Done |
| **Rate limiting** | Per-tenant / per-assistant throttling | Redis-backed token bucket (or use the queue's backpressure) | 🟡 Queue-level done; HTTP middleware not yet |
| **Horizontal API** | Multiple API server instances behind a load balancer | Stateless FastAPI + shared Postgres + Redis | Not yet |

#### Runtime process modes

The runtime ships as **one image** with different entrypoints:

| Command | What it runs | When to use |
|---|---|---|
| `llamphouse serve --workers` | API server + in-process AsyncWorker | Dev, small deploys, `pip install` usage |
| `llamphouse serve` | API server only (enqueues to Redis, no local worker) | Production API pods |
| `llamphouse worker` | Worker only (consumes from Redis Streams, no HTTP) | Production worker pods |

This means:
- **Dev / small scale:** Single process does everything (current behavior)
- **Production:** Separate API and worker containers from the same image, different command
- Workers are CPU/memory heavy (LLM calls, tool execution) → scale independently
- API pods stay responsive for SSE connections — no contention with worker load
- Worker crash doesn't kill the API; scale workers to zero during off-hours

#### Scaling architecture

```
                    Load Balancer
                    ┌─────┴─────┐
              ┌─────┤           ├─────┐
              │     │           │     │
         API Pod 1  API Pod 2  API Pod 3
         (serve)    (serve)    (serve)
              │     │           │     │
              └──┬──┘           └──┬──┘
                 │                 │
          ┌──────┴──────┐   ┌─────┴─────┐
          │   Redis     │   │ Postgres  │
          │ (queue +    │   │ (state)   │
          │  pub/sub)   │   │           │
          └──────┬──────┘   └───────────┘
                 │
          ┌──────┴──────────────┐
          │      │      │      │
       Worker  Worker Worker Worker
        Pod 1  Pod 2  Pod 3  Pod 4
       (worker)(worker)(worker)(worker)
```

All pods run the **same `llamphouse/runtime` image** — just with a different command.

#### Key decisions

| Decision | Recommendation | Why |
|---|---|---|
| One image, multiple modes | `serve` / `worker` / `serve --workers` | No code duplication. Same pattern as Celery/Django, Sidekiq/Rails, BullMQ/Express. |
| Queue tech | **Redis Streams** | Simple to operate, built-in consumer groups, most teams already run Redis. |
| State store | **Postgres** (keep as-is) | Already implemented, proven, supports the query patterns needed for both dashboards. |
| Event bus for streaming | **Redis Pub/Sub** | Worker publishes SSE events to a channel keyed by `assistant_id:thread_id`. The API pod subscribed to that channel streams them to the client. |

---

## Pillar 2: Spotlight (Business Dashboard)

> *For product managers, ops teams, and customer success. Answers: "What are users saying? How are agents performing? What should we improve?"*

### Features

| Feature | Description | Data source |
|---|---|---|
| **Conversation explorer** | Browse / search all threads with filters (date, assistant, status, sentiment) | Postgres (threads, messages) |
| **Full-text search** | Search message content across all conversations | Postgres `tsvector` + GIN index |
| **Data extraction pipeline** | Auto-extract metadata per conversation: sentiment, topic, intent, entities, language | LLM-as-judge or lightweight classifier |
| **Extracted data views** | Filter/aggregate by extracted fields (e.g., "show all negative-sentiment conversations") | Postgres (extracted data table) |
| **Usage analytics** | Runs per day, avg response time, token usage, error rates, assistant usage breakdown | ClickHouse |
| **Alerts** | Notify on spikes in errors, negative sentiment, or unusual patterns | Cron job + webhook/email |
| **Export** | CSV/JSON export of conversations and extracted data | API endpoint |

### Components to build

#### 1. Extraction pipeline

A background process that runs after each completed run and extracts structured data.

```python
class BaseExtractor(ABC):
    """Runs after each completed run to extract metadata."""
    
    @abstractmethod
    async def extract(self, thread: ThreadObject, messages: list[MessageObject], run: RunObject) -> dict:
        """Return extracted fields, e.g. {"sentiment": "positive", "topic": "billing"}."""
        pass

class LLMExtractor(BaseExtractor):
    """Uses an LLM to extract sentiment, topic, intent, etc."""
    ...

class RuleBasedExtractor(BaseExtractor):
    """Keyword/regex based extraction (fast, no LLM cost)."""
    ...
```

**When it runs:**
- Hook into the worker's run-completion flow (after `update_run_status(COMPLETED)`)
- Run extractors asynchronously (don't block the response)
- Store results in an `extractions` table linked to `run_id`

**Storage:**

```sql
CREATE TABLE extractions (
    id          TEXT PRIMARY KEY,
    run_id      TEXT REFERENCES runs(id) ON DELETE CASCADE,
    thread_id   TEXT REFERENCES threads(id) ON DELETE CASCADE,
    data        JSONB NOT NULL,          -- {"sentiment": "positive", "topic": "billing", ...}
    extractor   TEXT NOT NULL,           -- "llm" | "rule_based"
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_extractions_thread ON extractions(thread_id);
CREATE INDEX idx_extractions_data ON extractions USING GIN(data);
```

#### 2. Keyword search

Use **Postgres full-text search** — no extra infrastructure required.

```sql
-- Add a tsvector column + GIN index to the messages table
ALTER TABLE messages ADD COLUMN search_vec tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(content_text, ''))) STORED;

CREATE INDEX idx_messages_search ON messages USING GIN(search_vec);
```

Query example:

```sql
SELECT m.*, ts_rank(search_vec, q) AS rank
FROM   messages m, plainto_tsquery('english', :query) q
WHERE  search_vec @@ q
ORDER  BY rank DESC
LIMIT  50;
```

This covers keyword search with ranking, works well up to millions of rows, and adds zero operational overhead.

#### 3. Analytics

Use **ClickHouse** for all analytical queries (usage stats, token consumption, response times, error rates). ClickHouse is already in the stack for traces (see Pillar 3), so we reuse it here — no additional infra.

For the earliest MVP, simple Postgres aggregate queries are fine. Migrate analytical reads to ClickHouse as volume grows.

#### 4. Frontend

Both Spotlight and Compass use **Vue 3** (+ Vite for the build). The compiled static assets are shipped inside the Docker image and served by the dashboard adapters — no separate frontend deployment needed.

The existing Alpine.js mini-dashboard stays as the lightweight "built-in" dev view that comes with `pip install llamphouse`. The full Vue dashboards are the Docker-shipped experience.

---

## Pillar 3: Compass (Developer Dashboard)

> *For developers and ML engineers. Answers: "Why did the agent do that? How can I make it better? Which config produces the best results?"*

### Features

| Feature | Description | Data source |
|---|---|---|
| **Trace viewer** | Visualize the full trace of a run: LLM calls, tool calls, latencies, token counts | OpenTelemetry spans |
| **Config store** | View/edit assistant config params, snapshot per run | Config store (done ✅) |
| **Run comparison** | Side-by-side view of runs with different configs | Runs + config_values |
| **Prompt playground** | Test prompt changes interactively before deploying | Creates runs via API |
| **Evaluation framework** | Score runs against criteria (auto or human) | New eval store |
| **Logs** | Structured logs per run with filtering | OpenTelemetry logs or Postgres |

### Components to build

#### 1. Trace viewer

You already emit OpenTelemetry spans for every operation. Store them in **ClickHouse** for fast querying over large volumes.

**Why ClickHouse for traces:**
- Columnar storage is ideal for wide, append-only span data
- Sub-second aggregation queries over millions of spans
- Handles the analytics workload too (shared infra with Pillar 2)
- The OTel Collector has a native ClickHouse exporter

**Architecture:**

```
Agent Runtime
    │
    │  OTel SDK (gRPC/HTTP)
    ▼
OTel Collector  ──►  ClickHouse
    │                     ▲
    │                     │
    ▼                     │
Compass (Vue)  ───────────┘
```

**ClickHouse schema** (follows the OTel semantic conventions):

```sql
CREATE TABLE otel_traces (
    Timestamp        DateTime64(9),
    TraceId          String,
    SpanId           String,
    ParentSpanId     String,
    SpanName         LowCardinality(String),
    SpanKind         LowCardinality(String),
    ServiceName      LowCardinality(String),
    Duration         Int64,               -- nanoseconds
    StatusCode       LowCardinality(String),
    StatusMessage    String,
    SpanAttributes   Map(LowCardinality(String), String),
    ResourceAttributes Map(LowCardinality(String), String),
    Events           Nested(Timestamp DateTime64(9), Name LowCardinality(String), Attributes Map(LowCardinality(String), String))
) ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (ServiceName, SpanName, toUnixTimestamp64Nano(Timestamp))
TTL toDateTime(Timestamp) + INTERVAL 30 DAY;
```

Compass's trace viewer queries ClickHouse directly via its HTTP interface. For users who prefer Jaeger/Tempo, they can point the OTel Collector there instead — the runtime doesn't care.

#### 2. Evaluation framework

Allow developers to score runs — either automatically (LLM-as-judge) or manually.

```python
class BaseEvaluator(ABC):
    """Scores a completed run."""
    
    @abstractmethod
    async def evaluate(self, thread: ThreadObject, messages: list[MessageObject], 
                       run: RunObject, config: dict) -> EvalResult:
        pass

class EvalResult(BaseModel):
    scores: dict[str, float]   # e.g. {"relevance": 0.9, "helpfulness": 0.8}
    reasoning: str | None      # optional explanation
    evaluator: str             # "llm_judge" | "human" | "custom"
```

**Storage:**

```sql
CREATE TABLE evaluations (
    id          TEXT PRIMARY KEY,
    run_id      TEXT REFERENCES runs(id) ON DELETE CASCADE,
    scores      JSONB NOT NULL,
    reasoning   TEXT,
    evaluator   TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

**Dashboard view:** Show eval scores next to each run, enable sorting/filtering by score, and chart score trends over time.

#### 3. Run comparison

A dashboard view that puts two or more runs side-by-side:

```
┌─────────────────────────┬─────────────────────────┐
│ Run A                   │ Run B                   │
│ config: temp=0.7        │ config: temp=1.5        │
│         tone=neutral    │         tone=pirate     │
│                         │                         │
│ Score: 0.9              │ Score: 0.7              │
│                         │                         │
│ "Quantum computing is   │ "Arrr! Quantum computin'│
│  a field of..."         │  be a field where..."   │
└─────────────────────────┴─────────────────────────┘
```

This is purely a frontend feature — the data (runs, config_values, eval scores) already exists.

#### 4. Prompt playground

An interactive editor in the dashboard where developers can:
1. Pick an assistant
2. Edit config values (using the config form — already built)
3. Type a test message
4. Hit "Run" → creates a real thread + run via the API
5. See the response inline
6. Compare with previous playground runs

---

## Component map

### Python package (`pip install llamphouse`)

```
llamphouse/
  core/
    # ── Runtime (Pillar 1) ──────────────────────────────
    llamphouse.py              # Main app (exists)
    assistant.py               # Base assistant (exists)
    context.py                 # Run context (exists)
    workers/
      base_worker.py           # ABC (exists)
      async_worker.py          # In-process (exists)
      distributed_worker.py    # Redis Streams consumer (exists)
    queue/
      base_queue.py            # ABC (exists)
      in_memory_queue.py       # (exists)
      redis_queue.py           # Redis Streams impl (exists)
    streaming/
      event_queue/
        redis_event_queue.py   # Cross-worker SSE via pub/sub (exists)
    data_stores/
      base_data_store.py       # ABC (exists)
      in_memory_store.py       # (exists)
      postgres_store.py        # (exists)
    auth/                      # (exists)
    config_store/
      base.py                  # ABC (exists)
      in_memory_store.py       # (exists)
      postgres_store.py        # NEW — persistent config
    types/
      config.py                # Param types (exists)

    # ── Extraction (Pillar 2) ──────────────────────────
    extraction/                # NEW
      base_extractor.py        # ABC
      llm_extractor.py         # LLM-as-judge extraction
      pipeline.py              # Runs extractors after run completion

    # ── Evaluation (Pillar 3) ──────────────────────────
    evaluation/                # NEW
      base_evaluator.py        # ABC
      llm_evaluator.py         # LLM-as-judge scoring
      store.py                 # Eval result storage

    # ── Tracing (Pillar 3) ─────────────────────────────
    tracing/
      __init__.py              # OTel setup (exists)
      # Spans are exported to ClickHouse via the OTel Collector
      # (no custom exporter needed in the Python package)

    # ── CLI entrypoints ────────────────────────────────
    cli.py                     # `llamphouse serve`, `llamphouse worker` (exists)

    # ── Adapters ───────────────────────────────────────
    adapters/
      assistant_api/           # OpenAI Assistants API (exists)
      a2a/                     # Agent-to-Agent (exists)
      dashboard/               # Lightweight Alpine.js dashboard (exists)
```

### Docker images

```
docker/
  runtime/                     # llamphouse/runtime (one image, multiple modes)
    Dockerfile                 # Python + llamphouse package
    entrypoint.sh              # Dispatches to `serve` or `worker`
    otel-collector-config.yaml # Ships spans to ClickHouse
  spotlight/                    # llamphouse/spotlight
    Dockerfile                 # Vue app + API backend
    frontend/                  # Vue 3 + Vite source
    api/                       # FastAPI read-only API
  compass/                     # llamphouse/compass
    Dockerfile                 # Vue app + API backend
    frontend/                  # Vue 3 + Vite source
    api/                       # FastAPI API (config, evals, traces)
  docker-compose.yml           # Full stack (all-in-one mode)
  docker-compose.prod.yml      # Production (split API + workers)
  docker-compose.runtime.yml   # Private zone only (runtime + Postgres + Redis)
```

---

## Implementation priority

### Phase 1 — Foundation (now → 4 weeks)

| # | Item | Pillar | Effort | Impact |
|---|---|---|---|---|
| 1 | Docker Compose: runtime + Postgres + Redis | Infra | ✅ Done | One-command local dev & deploy |
| 2 | PostgresConfigStore | Dev | S | Persistent config across restarts |
| 3 | Config snapshot on runs | Dev | ✅ Done | Compare runs |
| 4 | Postgres keyword search (`tsvector`) | Biz | S | Conversation search |
| 5 | OTel Collector + ClickHouse setup (Docker) | Dev | M | Trace storage |

### Phase 2 — Compass (4–8 weeks)

| # | Item | Pillar | Effort | Impact |
|---|---|---|---|---|
| 6 | Compass scaffold (Vite + Vue 3) | Dev | M | Foundation for all dev UI |
| 7 | Trace viewer (reads from ClickHouse) | Dev | L | Self-hosted tracing |
| 8 | Run comparison view | Dev | M | Side-by-side config comparison |
| 9 | Evaluation framework (base + LLM evaluator) | Dev | M | Systematic quality measurement |
| 10 | Prompt playground | Dev | M | Interactive testing |

### Phase 3 — Spotlight (8–12 weeks)

| # | Item | Pillar | Effort | Impact |
|---|---|---|---|---|
| 11 | Extraction pipeline (base + LLM extractor) | Biz | M | Unlock business insights |
| 12 | Spotlight scaffold (Vite + Vue 3) | Biz | M | Foundation for biz UI |
| 13 | Usage analytics (ClickHouse queries + charts) | Biz | M | Ops visibility |
| 14 | Extracted data views + filtering | Biz | M | Product insights |
| 15 | Alerts (error spikes, sentiment drops) | Biz | M | Proactive monitoring |

### Phase 4 — Scale (12+ weeks)

| # | Item | Pillar | Effort | Impact |
|---|---|---|---|---|
| 16 | Redis Streams queue + distributed worker | Runtime | ✅ Done | Horizontal scaling |
| 17 | Redis Pub/Sub event queue (cross-worker SSE) | Runtime | ✅ Done | Stream from any worker |
| 18 | Rate limiting middleware | Runtime | 🟡 Partial | Multi-tenant safety (queue-level rate limiting done; HTTP middleware not yet) |
| 19 | Helm chart / K8s manifests | Infra | M | Production-grade deployment |

---

## Tech stack summary

| Layer | Component | Technology |
|---|---|---|
| **API** | HTTP server | FastAPI + Uvicorn |
| **API** | Protocol adapters | Assistants API, A2A, Dashboard(s) |
| **Compute** | In-process workers | AsyncWorker (asyncio) |
| **Compute** | Distributed workers | Redis Streams consumer |
| **State** | Primary database | PostgreSQL |
| **State** | Keyword search | Postgres `tsvector` + GIN index |
| **Analytics** | Traces + analytics | ClickHouse |
| **Messaging** | Run queue | Redis Streams |
| **Messaging** | SSE event bus | Redis Pub/Sub |
| **Observability** | Tracing pipeline | OpenTelemetry SDK → OTel Collector → ClickHouse |
| **Extraction** | Data pipeline | LLM-as-judge (async, post-run) |
| **Evaluation** | Quality scoring | LLM-as-judge + human review |
| **Frontend** | Built-in mini dashboard | Alpine.js (ships with pip package) |
| **Frontend** | Spotlight (business dashboard) | Vue 3 + Vite (Docker image) |
| **Frontend** | Compass (developer dashboard) | Vue 3 + Vite (Docker image) |
| **Packaging** | Containers | Docker / Docker Compose / Helm |

---

## Deployment models

### Model 1: All-in-one (`docker compose up`)

Everything runs on a single machine — great for development and small deployments.

```yaml
# docker-compose.yml
services:
  runtime:              # llamphouse/runtime — `llamphouse serve --workers`
  spotlight:            # llamphouse/spotlight (Vue + API)
  compass:              # llamphouse/compass (Vue + API)
  postgres:             # State store
  redis:                # Queue + pub/sub
  clickhouse:           # Traces + analytics
  otel-collector:       # Receives spans, writes to ClickHouse
```

For production on a single machine, split API and workers:

```yaml
# docker-compose.prod.yml
services:
  api:                  # llamphouse/runtime — `llamphouse serve`
  worker:               # llamphouse/runtime — `llamphouse worker`
    replicas: 3         # scale workers independently
  spotlight:            # llamphouse/spotlight
  compass:              # llamphouse/compass
  postgres:
  redis:
  clickhouse:
  otel-collector:
```

### Model 2: Split deployment (private runtime, hosted dashboards)

The agent runtime stays inside the customer's VPC (handles user data, API keys, LLM calls). The dashboards run in a separate zone — either LLAMPHouse-hosted or in a shared cluster.

```yaml
# docker-compose.runtime.yml  ← runs in customer VPC
services:
  api:                  # llamphouse/runtime — `llamphouse serve`
  worker:               # llamphouse/runtime — `llamphouse worker` (scale as needed)
  postgres:
  redis:
  otel-collector:       # Exports spans out to the platform zone

# docker-compose.platform.yml  ← runs in platform zone
services:
  spotlight:
  compass:
  clickhouse:           # Receives spans from customer's OTel Collector
  postgres:             # Dashboards' own read-replica or separate DB
```

**Data flow between zones:**

| Data | Direction | Protocol | Notes |
|---|---|---|---|
| Spans/traces | Runtime → Platform | OTel Collector (gRPC/HTTP) | OTel Collector in private zone pushes to platform ClickHouse |
| Run/thread/message data | Runtime → Platform | Postgres logical replication or API sync | Read-only replica for dashboards |
| Config updates | Platform → Runtime | API call (HTTPS) | Compass writes config back to runtime |

### Model 3: Kubernetes

For production-scale deployments, ship Helm charts:

```
helm/
  llamphouse/
    Chart.yaml
    values.yaml                 # image: llamphouse/runtime (shared)
    templates/
      api-deployment.yaml       # `llamphouse serve` — HPA on request rate
      api-service.yaml
      worker-deployment.yaml    # `llamphouse worker` — HPA on queue depth
      spotlight.yaml
      compass.yaml
      otel-collector.yaml
      # Postgres, Redis, ClickHouse via sub-charts or external
```

---

## Docker images

| Image | Contents | Modes | Base | Size target |
|---|---|---|---|---|
| `llamphouse/runtime` | FastAPI app, workers, OTel SDK | `serve`, `worker`, `serve --workers` | `python:3.12-slim` | < 200 MB |
| `llamphouse/spotlight` | Vue static build + FastAPI API (Spotlight) | — | `python:3.12-slim` | < 150 MB |
| `llamphouse/compass` | Vue static build + FastAPI API (Compass) | — | `python:3.12-slim` | < 150 MB |
| `llamphouse/otel-collector` | OTel Collector with ClickHouse exporter | — | `otel/opentelemetry-collector-contrib` | ~ 100 MB |

The runtime is **one image with three modes** — not separate images for API and worker. Different containers just run different commands:

```yaml
# In docker-compose
api:
  image: llamphouse/runtime:latest
  command: ["llamphouse", "serve"]

worker:
  image: llamphouse/runtime:latest
  command: ["llamphouse", "worker"]
  deploy:
    replicas: 3
```

Each image is built in CI, tagged with the version, and pushed to a registry (Docker Hub or GitHub Container Registry).

The `pip install llamphouse` package continues to work standalone for development — it includes the runtime + Alpine.js mini-dashboard and defaults to `serve --workers` mode (everything in one process).

---

## Design principles

1. **Everything is pluggable** — ABCs with sensible defaults. Users swap implementations without changing app code.
2. **Zero mandatory infra** — `pip install llamphouse` works with InMemory defaults. Postgres, Redis, ClickHouse are opt-in (and bundled in Docker Compose).
3. **`docker compose up` to full platform** — One command gives you the runtime, both dashboards, and all backing services.
4. **Private-first runtime** — The agent runtime (user data, API keys, LLM traffic) runs in the customer's infrastructure. Dashboards can run separately.
5. **OpenTelemetry native** — All observability flows through OTel. Spans go to ClickHouse by default, but users can point the Collector at any backend.
6. **Config as code** — Agent configuration is defined in Python (class attributes), versioned in git, and editable at runtime via the dashboard.
7. **Vue for rich UI, Alpine for embedded** — The full dashboards use Vue 3 for a rich experience. The pip-shipped mini-dashboard stays Alpine.js for zero-dependency simplicity.
