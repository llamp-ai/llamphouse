# Triggers

Nothing starts a run by itself: a person, an app or another agent does. Triggers are the other things that may:
an HTTP call from another system (a **webhook**) and the clock (a **schedule**). They are declared in an agent's
file. Agents can also schedule runs themselves, with the **Scheduler** capability.

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: orders}
spec:
  entrypoint: agents.py:Orders
  triggers:
    - webhook:
        path: /hooks/orders                  # POST here
        secret_env: ORDERS_SECRET            # Authorization: Bearer <secret>
        thread: {label: $.order.id}          # one thread per order
        message: {text: $.summary}           # the input's text; the whole payload comes along as data
        thread_metadata: {customer: $.customer.id}
        idempotency: {key: $.event_id}       # the same event twice: one run
        busy: send                           # the order's thread is busy: send it to the active run
    - schedule:
        cron: "0 7 * * MON-FRI"
        timezone: Europe/Brussels
        input: {text: Morning check of open orders}
        thread: {label: morning}             # the same thread every time
        busy: skip
```

A trigger's message comes from a sender of kind `trigger` whose id is the trigger's name, and the run's caller
is `trigger:<name>`. In agent code:

```python
from llamphouse.sdk import Agent, SenderKind


class Orders(Agent):
    """Keeps an eye on orders."""

    async def run(self, ctx) -> None:
        if ctx.input.sender.kind == SenderKind.TRIGGER:
            payload = ctx.input.data          # the webhook's JSON body (a schedule's `input.data`)
            ...
```

## Webhooks

A webhook mounts `POST <path>` on the API processes, outside `/v2` and its auth: each webhook checks its own
secret. The body must be JSON.

| Option | |
|---|---|
| `path` | required: where it is mounted (unique across the deployment) |
| `name` | the trigger's name (default from the path: `webhook:hooks:orders` for `/hooks/orders`) |
| `secret` / `secret_env` | a bearer token, given or read from an environment variable: `Authorization: Bearer <secret>` |
| `signature` | an HMAC-SHA256 of the raw body instead: `{header: X-Hub-Signature-256, secret_env: ..., prefix: "sha256="}` (header and prefix shown are the defaults) |
| `public` | `true` for an endpoint with neither (it must be said explicitly) |
| `message` | `{text: <path>}`: the input's text, from the payload |
| `thread` | `{label: <path>}`: the thread for this value (made once, the same one every time); default: a new thread per call |
| `thread_metadata` | labels for a new thread, from the payload: `{key: <path>}` |
| `idempotency` | `{key: <path>}`: the same key again gets the first answer back (`200`, `Idempotent-Replay: true`), and starts nothing |
| `busy` | what happens when the thread's run is still active: `send` (default: to its inbox), `skip`, or `reject` (`409`) |
| `tenant` | the tenant its threads belong to, under tenant access |

Paths into the payload look like `$.order.id` or `order.items.0.sku`.

What a call gets back:

| Status | |
|---|---|
| `202` | `{"thread": "...", "run": "...", "action": "started" \| "sent" \| "skipped"}` |
| `200` | the same event again (idempotency): the first answer |
| `400` | the body is not JSON |
| `401` / `403` | the secret or signature is wrong or missing |
| `409` | `busy: reject` and the thread is busy, or the same event is being handled right now |
| `422` | the payload lacks the thread label or idempotency key |
| `503` | the secret is not configured |

```bash
curl -s localhost:8000/hooks/orders -H "Authorization: Bearer $ORDERS_SECRET" -H 'content-type: application/json' \
     -d '{"event_id": "evt_1", "order": {"id": "A-17"}, "customer": {"id": "c-9"}, "summary": "Order A-17 shipped"}'
# {"thread": "thread_...", "run": "run_...", "action": "started"}
```

Signed webhooks (GitHub, Stripe-style):

```yaml
- webhook:
    path: /hooks/github
    signature: {header: X-Hub-Signature-256, secret_env: GITHUB_WEBHOOK_SECRET}
    thread: {label: $.repository.full_name}
    idempotency: {key: $.hook_id}
```

Compass shows every webhook with a ready `curl` call.

## Schedules

| Option | |
|---|---|
| `cron` | a cron expression (below), in `timezone` |
| `every` | an interval: `30s`, `15m`, `2h`, `1d`, or seconds |
| `at` | once, at an ISO time (`2026-12-01T09:00:00Z`) |
| `timezone` | for `cron` (default `UTC`): `0 7 * * *` is 07:00 there, summer or winter |
| `input` | the run's input: `{text: ..., data: {...}}`, or a string |
| `thread` | `{label: <name>}`: the same thread every time; default a new thread per fire |
| `busy` | `skip` (default), `send` or `reject` |
| `name` | the trigger's name (default `schedule:<index>`) |
| `tenant` | the tenant its threads belong to |

Exactly one of `cron`, `every` and `at`.

Cron expressions have five fields (minute, hour, day of month, month, day of week). Each is `*`, a number, a range
`a-b`, a step `*/n` or `a-b/n`, or a list of those; months and days also by name (`JAN`-`DEC`, `SUN`-`SAT`; 0 and
7 are Sunday). When both day fields are restricted, a day matches either. Shorthands: `@yearly`, `@monthly`,
`@weekly`, `@daily`, `@hourly`.

How schedules run:

- Schedules are kept in the data store. The workers' sweeper fires one when it is due, exactly once across all
  workers: whichever sees it first moves it on.
- A fire missed while no worker ran happens once when one runs again (not once per missed time).
- A changed schedule in the agent file starts over; a removed one is dropped, on the next start.
- In Compass, the Triggers page and each agent's page list schedules with their next time and a **Fire now**
  button.

## The Scheduler capability

An agent can schedule a run of an agent for later, kept by the store, so it happens even if this process is
gone:

```python
from datetime import datetime, timezone

from llamphouse.sdk import Agent, Scheduler, requires


@requires(Scheduler)
class Reminders(Agent):
    """Sets reminders."""

    async def run(self, ctx) -> None:
        scheduler = ctx.require(Scheduler)
        key = await scheduler.schedule("reminder", delay=3600, key=f"remind-{ctx.thread.id}", who="An", what="call Bo")
        await scheduler.schedule("reminder", at=datetime(2026, 12, 1, 9, tzinfo=timezone.utc), what="year end")
        ...
        await scheduler.cancel(key)
```

`schedule(agent, *, at=None, delay=None, key=None, **data)`:

- starts a run of `agent` at `at` (or after `delay` seconds) on a new thread, with `data` as its input
  (`ctx.input.data` there);
- `key` names it: scheduling the same key again replaces it, and `cancel(key)` drops it; it returns the key (a
  random one when none is given);
- it is recorded like a step: a run that executes again doesn't schedule twice;
- the agent's `permissions.agents` applies, as for `ctx.call`.

`@requires(Scheduler)` makes the runtime refuse to start the agent where there is no scheduler, instead of
failing mid-run. Without it, `ctx.capability(Scheduler)` returns `None` where it isn't offered.

## `engine.fire`

Both triggers use one primitive, which apps can call too:

```python
from llamphouse.sdk import Message, Sender

result = await engine.fire("orders", Message.create(Sender.trigger("import"), "Re-check order A-17"),
                           label="A-17", busy="send")
# {"thread": "...", "run": "...", "action": "started" | "sent" | "skipped"}
```

A label's thread is made once, whichever process comes first.

## See also

- [Example 11_Triggers](../../examples/11_Triggers)
- [Subscriptions](subscriptions.md): the other direction, telling outside systems what runs did
