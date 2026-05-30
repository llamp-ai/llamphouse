# Plan — Compass for Developers

Status: **draft / not implemented**
Target release: TBD (multi-phase, ships incrementally)
Scope: turn Compass from a read-only observability dashboard into an interactive iteration surface for developers building agents on LLAMPHouse.

---

## 1. Motivation

Compass today answers "what happened?" via stat cards, sortable tables, dashboards, and charts. That's BI software. The people who actually use Compass — engineers iterating on agent code — need it to answer **"what should I do next?"** and let them act on the answer in place.

Concretely, devs need to:

- Try an idea against an agent without hand-writing a client (**Playground**)
- Re-run an old input against new code to verify a fix (**Replay**)
- Trigger an agent from the same UI that triggered it in prod (**Webhook actions**)
- Slice the data however they want without waiting on us to ship a view (**SQL editor**)
- Mark good / bad runs and track quality over time (**Scores**)
- Pin a set of inputs as a regression benchmark (**Datasets**)
- Customise the home page so what they care about is what they see (**Editable Overview**)

These features compose: Replay rides on Playground; Datasets enable batch Replay; Scores make Dataset runs evaluable; the Editable Overview surfaces all of the above as cards.

---

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Widget** | A self-contained visual unit (chart, table, stat card, custom HTML). Today's `ChartWidget` is one. |
| **Dashboard** | A named collection of widgets with a layout. Already exists. |
| **Overview** | The home page. Becomes a special, user-editable dashboard with a sensible default. |
| **Playground** | A view that creates and runs an agent run from the UI. Inputs, config, model, system prompt are all editable. |
| **Replay** | A Playground session pre-filled from an existing run. |
| **Action** | A button declared on an agent (or globally) that fires a side effect — initially webhook POSTs. |
| **Score** | A named numeric or categorical label attached to a run, with a `source` (human / agent / external). |
| **Dataset** | A named set of `(input, optional_expected_output)` pairs. Stored, versioned, runnable. |

---

## 3. Features

### 3.1 Editable Overview (small, foundation)

**Goal:** the home page is a dashboard the user can edit, same as any other.

**Approach:**
- The current hard-coded Overview becomes a special row of system widgets ("Recent Runs", "Active Runs", "Stats") plus an empty "drop your widgets here" area.
- Backend: introduce a reserved dashboard id `__overview__` per workspace (or per user, see open questions). Same `DashboardStore` schema as today.
- Frontend: `OverviewView` reuses `DashboardView`'s renderer. An "Edit Overview" toggle switches into edit mode.
- "Import from Dashboard" picker: choose any widget from any existing dashboard → it's cloned (not referenced) into the Overview.

**System widgets:**
- `recent_runs` — last N runs, click to detail (already implemented)
- `recent_threads` — last N threads
- `active_runs` — runs with status `in_progress`, auto-refreshing
- `failures_last_hour` — count + tap to filter
- `your_agents` — agents whose code/config changed since their last run (needs a change-detection helper)

**Out of scope for v1:** sharing dashboards between users (single-user installs are the norm).

### 3.2 Playground — call agents from the UI

**Goal:** the user picks an agent, types/edits an input, optionally tweaks config, hits Run, watches the response stream in.

**Approach:**
- New view `/playground` and matching route.
- **Agent picker** — populated from `req.app.state.assistants`.
- **Input editor** — text area for the user message, plus an "Advanced" disclosure for system prompt override, config values, model override (subject to what the agent allows).
- **Config values editor** — for agents with `BaseConfigStore` params, render them with the existing form widgets used by Dashboards.
- **Run button** creates a new thread + run via the existing OpenAI-compat or A2A adapter (whichever is mounted), then streams events into a familiar `RunDetailView`-style panel.
- **History pane** — left rail listing recent Playground runs (filter to runs whose `metadata.source == "playground"` so they don't pollute the main Runs view by default).

**Backend addition:** none new for the run itself — the existing run-creation flow handles it. Add `metadata.source = "playground"` so we can identify these runs later. Optionally tag the user (when auth is on).

**UX details:**
- Cmd/Ctrl+Enter to submit
- Persist last input per agent in localStorage
- "Branch from here" button on any assistant message: copies the conversation up to that point into a fresh Playground session

### 3.3 Replay — Playground session pre-filled from an existing run

**Goal:** on any run, click "Replay" → opens the Playground with that run's input, system prompt, and config pre-populated, ready to edit or run as-is.

**Approach:**
- New backend endpoint `POST /api/runs/{run_id}/replay-payload` that returns `{ agent_id, input_message, system_prompt, config_values, model, tools }` extracted from the original run.
- "Replay" button on `RunDetailView` → router-pushes to `/playground?from_run=<run_id>`.
- Playground reads `from_run` and fetches the payload, fills the form. User can edit anything before hitting Run.
- The new run's `metadata.replayed_from = <original_run_id>`. Surface in the run header: "Replay of `run_abc…`".

**Composability:**
- Dataset items (§ 3.7) can each be one-click replayed.
- A future "Replay all" button on a dataset uses the same primitive.

### 3.4 Actionable buttons (Webhook actions)

**Goal:** declare buttons on an agent that, when clicked in Compass, fire a webhook (or run the agent with a fixed payload).

**Approach (two flavours, share UI):**

**Flavour A — Trigger a webhook-triggered run.**
When the Trigger system from `PLAN_LIFECYCLE_EVENTS.md` lands, any `WebhookTrigger` declared on an agent appears as a button in Compass:
```
ReportAgent
  ▸ Trigger: POST /triggers/report  [▶ Fire]
```
Clicking opens a modal with a JSON body editor, optional auth header field, and a "Fire" primary button.  The new run shows up under the agent's history.

**Flavour B — Outbound action buttons.**
Agents can declare:
```python
class MyAgent(Agent):
    actions = [
        WebhookAction(
            label="Notify Slack",
            url="https://hooks.slack.com/…",
            body_template=lambda run: {"text": f"Run {run.id} done"},
        ),
    ]
```
These render as buttons on the run detail page. Click → POST. Result shown inline.

**v1 scope:** ship Flavour A (depends on Triggers landing). Flavour B is a thin add-on once `BaseSubscriber` exists.

### 3.5 SQL editor

**Goal:** ad-hoc SELECT queries from a dev console.

**Approach:**
- New top-level tab `/sql`.
- Reuses the existing dashboard SQL execution path (`_run_postgres_query` / `_build_and_run_sqlite`) including the `_check_sql` allowlist (SELECT-only).
- Code editor with SQL syntax highlighting (CodeMirror or Monaco — both already in similar Vue projects).
- Results table reuses `DataTable`.
- **Query history** persisted in localStorage; **Saved queries** persisted server-side under a new `saved_queries` table.
- **Schema sidebar** — list of tables and columns. Click → inserts at cursor.
- **"Open in dashboard"** — promotes a working query to a `ChartWidget` query, dropping into the active dashboard.

**Safety:**
- Hard cap on rows returned (1000 default, override per-query up to 10000).
- Hard cap on query duration (statement_timeout via Postgres; soft timeout for SQLite).
- Reject anything that isn't `SELECT` / `WITH` / `EXPLAIN`.

### 3.6 Scores — labels attached to runs

**Goal:** mark runs with named, typed values that can be filtered, aggregated, and charted.

**Schema:**
```
scores:
  id           UUID PK
  run_id       FK → runs.id  ON DELETE CASCADE
  name         TEXT   (e.g. "correctness", "user_rating", "tokens_per_dollar")
  value_num    DOUBLE PRECISION   nullable  (for numeric scores)
  value_text   TEXT               nullable  (for categorical / boolean scores)
  source       TEXT   (e.g. "human", "agent:eval-judge", "external")
  notes        TEXT   nullable
  created_at   TIMESTAMPTZ
  created_by   TEXT   nullable
  INDEX (run_id, name)
  INDEX (name, value_num)
```

**API:**
- `POST /api/runs/{run_id}/scores` — add/update a score.
- `GET /api/runs/{run_id}/scores` — list.
- `DELETE /api/scores/{score_id}` — remove.

**UI:**
- **Scores panel on `RunDetailView`** — chips showing each score, with inline edit. New score: name + value form.
- **Score filter** in the Runs page filter builder: "has score X", "score X > 0.5", "score X = 'good'".
- **Score column** opt-in in the Runs table.
- **Score widget** for Dashboards / Overview: histograms, time series, leaderboards by agent.

**Python side:**
- `context.add_score(name, value, source="agent", notes=None)` — agents can self-score from inside their `run()` (handy for built-in critique/judge agents).
- Optional `@score(name="...", judge=lambda ctx: ...)` decorator that runs after `agent.run()` and writes a score.

### 3.7 Datasets

**Goal:** named, versioned collections of test inputs (+ optional expected outputs) that can be run against an agent in batch and scored.

**Schema:**
```
datasets:
  id           UUID PK
  name         TEXT UNIQUE
  description  TEXT
  created_at   TIMESTAMPTZ
  updated_at   TIMESTAMPTZ

dataset_items:
  id            UUID PK
  dataset_id    FK → datasets.id  ON DELETE CASCADE
  input         JSONB              (the user message / payload)
  expected      JSONB nullable     (optional reference output)
  metadata      JSONB nullable
  position      INT                (preserve order)
  INDEX (dataset_id, position)

dataset_runs:
  id            UUID PK
  dataset_id    FK → datasets.id
  agent_id      TEXT
  agent_version TEXT
  status        TEXT   (queued | running | completed | failed | cancelled)
  started_at    TIMESTAMPTZ nullable
  completed_at  TIMESTAMPTZ nullable
  config_snapshot JSONB nullable

dataset_run_items:
  dataset_run_id  FK → dataset_runs.id
  dataset_item_id FK → dataset_items.id
  run_id          FK → runs.id        nullable (set once dispatched)
  status          TEXT
  PRIMARY KEY (dataset_run_id, dataset_item_id)
```

**API:**
- `POST /api/datasets` — create.
- `POST /api/datasets/{id}/items` — add items (bulk).
- `POST /api/datasets/{id}/run` — kick off a `dataset_run` against an agent; one LLAMPHouse run per item, dispatched through the existing run queue.
- `GET /api/dataset-runs/{id}` — status + per-item results, with score rollups.

**Constructor flows:**
- **From scratch**: empty dataset → add items manually.
- **From existing runs**: select runs in the Runs view → "Save as dataset" → snapshots input + (optionally) the original output as the `expected`.
- **From file**: JSONL / CSV upload via the Datasets page.

**UI:**
- New `/datasets` page: list of datasets, click → dataset detail with items table.
- **"Run against agent"** button: agent picker + config overrides → kicks off a `dataset_run`.
- **Dataset run view**: live progress, per-item input/output/diff, score histogram.
- **Replay this item** → Playground.

**Eval composition:** if a dataset has `expected` set on items, Scores can be auto-applied per item using a configurable judge (string equality, regex, LLM judge — pluggable). Out of scope for v1; the data model already supports it.

---

## 4. Cross-cutting concerns

### 4.1 New tables and migrations

Tables to add over the whole plan:
- `scores`
- `datasets`
- `dataset_items`
- `dataset_runs`
- `dataset_run_items`
- `saved_queries` (for SQL editor)

One Alembic migration per feature; no breaking changes to existing tables.

### 4.2 Data store extensions

Each new table needs:
- `BaseDataStore` abstracts: `insert_score`, `list_scores_for_run`, `delete_score`, `insert_dataset`, `list_datasets`, etc.
- Postgres + In-memory implementations.
- The same pagination / filter / `include_total` contract already in place for threads and runs.

### 4.3 Auth & multi-user

Currently single-user / dev-mode. When auth lands (`AuthResult` already exists), Scores and Saved Queries get a `created_by` field; Datasets are owned. Out of scope for v1 but the schemas above accommodate it.

### 4.4 Performance

- Score histograms in Dashboards: precompute via SQL `GROUP BY` rather than in Python.
- Dataset runs use the existing run queue — no new dispatch infrastructure.
- Playground SSE reuses the existing event queue.

---

## 5. Stepped development plan

Phases ordered by dependency and user value.  Each step is roughly one PR with a clear "Done when" criterion.

### Phase 1 — SQL editor (standalone, highest leverage for power users)

- [ ] **1. `saved_queries` table + migration.** *Done when:* `alembic upgrade head` applies; CRUD over `BaseDataStore`.
- [ ] **2. `GET/POST/DELETE /api/saved-queries`** routes; reuse existing `_check_sql`. *Done when:* round-trip via curl.
- [ ] **3. `/sql` Compass page** with code editor (Monaco), Run button, results table. *Done when:* querying `runs` table returns rows.
- [ ] **4. Schema sidebar** populated from `INFORMATION_SCHEMA` (Postgres) / `sqlite_master`. *Done when:* clicking a table name inserts `SELECT * FROM <table> LIMIT 100`.
- [ ] **5. Query history** in localStorage; **Saved queries** UI list. *Done when:* both persist across reloads.
- [ ] **6. "Open as dashboard widget"** action — saves the query as a `ChartWidget` and navigates to the active dashboard. *Done when:* end-to-end flow works.

### Phase 2 — Playground (foundational for Replay, Datasets)

- [ ] **7. `/playground` route + view scaffold** with agent picker. *Done when:* renders empty state for any registered agent.
- [ ] **8. Input editor + Run button** that creates a thread + run via the existing API. *Done when:* a free-form message round-trips to the agent and the response renders.
- [ ] **9. Streaming output panel** — reuses the existing SSE event queue. *Done when:* messages stream in word-by-word.
- [ ] **10. Config-values editor** populated from `BaseConfigStore` params on the selected agent. *Done when:* changing a value alters the next run's behaviour.
- [ ] **11. System prompt / model override** in an "Advanced" disclosure. *Done when:* override is honoured by the run.
- [ ] **12. Playground history rail** filtered by `metadata.source = "playground"`. *Done when:* rerun-from-history works.
- [ ] **13. Keyboard shortcuts** (⌘/Ctrl+Enter to submit, ↑ to recall last input). *Done when:* both bindings live.

### Phase 3 — Replay (Playground extension)

- [ ] **14. `POST /api/runs/{run_id}/replay-payload`** returns the run's input + config snapshot. *Done when:* JSON shape matches Playground form contract.
- [ ] **15. "Replay" button on `RunDetailView`** → opens Playground with `?from_run=<id>`. *Done when:* form is pre-filled.
- [ ] **16. `metadata.replayed_from`** stamping on the new run; banner in the new run's detail view linking back. *Done when:* lineage is clickable in both directions.

### Phase 4 — Scores

- [ ] **17. `scores` table + migration.** *Done when:* schema matches §3.6.
- [ ] **18. Data store methods** (`insert_score`, `list_scores_for_run`, `delete_score`, `list_scores_by_name`). *Done when:* both Postgres and in-memory implementations covered.
- [ ] **19. Scores REST API** (`POST` / `GET` / `DELETE`). *Done when:* contract tests pass.
- [ ] **20. Scores panel on `RunDetailView`** — chips with inline add/edit/delete. *Done when:* round-trip works.
- [ ] **21. `context.add_score(...)`** for agents to self-score during `run()`. *Done when:* an example agent writes a score and Compass shows it.
- [ ] **22. Score-aware filters** in `Runs` view (extend the filter allowlist to `score:<name>`). *Done when:* "`correctness > 0.8`" returns only matching runs.
- [ ] **23. Score widget** for Dashboards (histogram, time-series). *Done when:* widget renders against live data.

### Phase 5 — Editable Overview

- [ ] **24. Promote Overview to a Dashboard** — reserved id `__overview__`, seeded on first load with default system widgets. *Done when:* opening Overview shows the same content as before.
- [ ] **25. Edit mode toggle** on Overview, reusing existing Dashboard editing controls. *Done when:* widgets can be added, moved, resized, removed.
- [ ] **26. "Import from Dashboard"** picker — pick a widget from any other dashboard, clone into Overview. *Done when:* changes to the source widget don't affect the cloned one.
- [ ] **27. New system widgets**: `active_runs`, `failures_last_hour`, `your_agents` (agents whose code changed). *Done when:* each renders correct numbers against a seeded fixture.

### Phase 6 — Datasets

- [ ] **28. `datasets`, `dataset_items`, `dataset_runs`, `dataset_run_items` tables + migration.** *Done when:* `alembic upgrade head` applies cleanly.
- [ ] **29. Dataset CRUD API** (`POST/GET/DELETE /api/datasets`, `POST /api/datasets/{id}/items`). *Done when:* curl round-trip.
- [ ] **30. `/datasets` page** — list, create, detail with items table. *Done when:* CRUD works from the UI.
- [ ] **31. "Save as dataset"** action on the Runs view — turns the current selection into a new dataset (snapshots input, optionally output as `expected`). *Done when:* item count matches the selection.
- [ ] **32. `POST /api/datasets/{id}/run`** dispatches one LLAMPHouse run per item through the existing queue. *Done when:* a 5-item dataset produces 5 runs with `metadata.dataset_run_id` stamped.
- [ ] **33. Dataset run detail view** — live progress, per-item input/output, links to each run. *Done when:* opening a 5-item run shows status for each.
- [ ] **34. Per-item Replay** → opens Playground prefilled from the dataset item (not the run). *Done when:* edited replays don't mutate the dataset.

### Phase 7 — Webhook actions

> Depends on the Trigger system from `PLAN_LIFECYCLE_EVENTS.md` shipping first.

- [ ] **35. Trigger panel on agent detail page** — lists each `WebhookTrigger` declared on the agent. *Done when:* triggers are visible per agent.
- [ ] **36. "Fire" modal** with JSON body editor and auth header field. *Done when:* clicking Fire posts to the trigger's path and a new run appears.
- [ ] **37. Trigger history** under each trigger — list of runs fired by it. *Done when:* per-trigger run list is populated.
- [ ] **38. (Optional) Outbound `WebhookAction`** — agent-declared buttons on the run detail page that POST elsewhere. *Done when:* clicking a button POSTs and shows the response inline.

### Cross-cutting (apply to every step)

- [ ] Unit tests at each step touching code; integration tests at phase boundaries (steps 9, 16, 23, 32).
- [ ] No `# TODO` left in shipped code.
- [ ] Migrations are forward-only and reviewed before merge.
- [ ] Public API additions go through `llamphouse/__init__.py`.
- [ ] Every new view answers "what can the user do here?" — not just "what can they read?"

---

## 6. Open questions

1. **Workspace / user scope.** Multi-user installs need per-user dashboards, saved queries, datasets. v1 ignores this; v2 adds a `created_by` filter everywhere. Schemas above accommodate it.
2. **Score schema split.** One row per `(run_id, name)` (upsert semantics) or one row per `(run_id, name, source)` (append-only audit log)? Append-only is more honest but the UI gets fiddlier. Default to upsert; revisit if eval pipelines need history.
3. **Dataset versioning.** Mutating a dataset between runs means a "dataset_run" can't be replayed exactly. Two options:
   - Snapshot the items into `dataset_run_items` at dispatch time (extra storage)
   - Version datasets explicitly (`datasets(id, version)`, items reference a version)

   Snapshot is simpler. Versioning is more honest. Recommend snapshot for v1.
4. **Playground auth.** When auth is on, anyone with Compass access can fire arbitrary runs. Acceptable for single-tenant dev installs; needs role-gating for shared installs.
5. **SQL editor blast radius.** Read-only SQL is fine, but a giant `SELECT * FROM messages` can OOM the server. Hard row cap + per-query timeout, set conservatively.
6. **Webhook actions and outbound `WebhookAction` semantics.** Synchronous (await response, show inline) or fire-and-forget? Sync is more useful for dev workflows; async fits later production patterns.

---

## 7. Non-goals (for this plan)

- Production-grade RBAC / multi-tenancy.
- Reproducing LangSmith / Helicone wholesale — focus on the dev-loop subset.
- LLM-judge eval framework (datasets just hold the data; judges come later as plug-ins).
- Realtime collaboration on dashboards.
- Mobile UI.

---

## 8. The smallest first PR

If you want a single PR that proves the direction, ship **Playground end-to-end** (Phase 2). It's the largest leverage step: it teaches the team how to mount a new top-level view with streaming, exercises every existing primitive (agents, ConfigStore, SSE), and unblocks Replay (one endpoint + a button) and Datasets (batch dispatch on top of the same flow).

Everything else in the plan is conceptually similar to features that already exist (Dashboards, Filters, RunDetailView). Playground is the only step that introduces a genuinely new interaction model — get it right and the rest is incremental.
