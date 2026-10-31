# Triggers

An order desk that other systems and the clock start, not a person. All of it is in the agent file's
`triggers:`: a webhook the shop POSTs its order events to (signed with an HMAC of the body, one thread per
order, the event's summary as the input, idempotent on the event's id), a webhook for an ops tool with a
bearer token, and two schedules - a cron digest every weekday morning and a check every minute. `client.py`
plays the shop and the ops tool, then fires the digest now, as Compass's *Fire now* button does. No model and
no key.

## Run

```bash
llamphouse up --memory --agents agents.yaml      # http://127.0.0.1:8000/compass
python client.py                                 # in a second terminal
```

```
The shop's webhook (HMAC-signed, a thread per order, idempotent on event_id):
  202  order.created A-1001: {'thread': 'thread_0939...', 'run': 'run_1ca2...', 'action': 'started'}
        -> Order A-1001 for acme: new order, 3 items (events so far: order.created)
  200 (Idempotent-Replay)  the same event again: {'thread': 'thread_0939...', 'run': 'run_1ca2...', 'action': 'started'}
  202  order.paid A-1001: {'thread': 'thread_0939...', 'run': 'run_c758...', 'action': 'started'}
        -> Order A-1001 for acme: paid (events so far: order.created, order.paid)
  202  order.created A-1002: {'thread': 'thread_a7e6...', 'run': 'run_b13b...', 'action': 'started'}
        -> Order A-1002 for globex: new order, 1 item (events so far: order.created)
  401  signed with the wrong secret: {'error': 'bad signature'}
  acme's threads (GET /v2/threads?metadata.customer=acme): ['thread_0939...'] - A-1001's is thread_0939...

The ops webhook (bearer token):
  202  with the token: {'thread': 'thread_4672...', 'run': 'run_446e...', 'action': 'started'}
        -> [ops] noted: shop maintenance at 22:00
  403  a wrong token: {'error': 'invalid token'}
  401  no token: {'error': 'Authorization: Bearer <secret> required'}

Schedules (Compass: Triggers, or Agents -> orders -> Triggers):
  trigger:orders:daily-digest      cron 0 7 * * MON-FRI (Europe/Amsterdam) next 2026-11-02T06:00:00+00:00
  trigger:orders:stuck-check       every 60s                            next 2026-10-31T11:01:57+00:00
  fire now: {'id': 'trigger:orders:daily-digest', 'fired': True, 'note': None}
        -> [daily-digest] Daily digest: nothing stuck, all orders moving.
```

The secrets have defaults for the demo (`shop-secret`, `ops-token`); set `SHOP_SECRET` and `OPS_TOKEN` for
both the server and the client to use others. `LLAMPHOUSE_URL` points the client elsewhere (default
`http://127.0.0.1:8000`). Leave the server running a minute and `stuck-check` fires by itself.

## In Compass

- **Triggers**: every webhook (its path, how it is checked, its label and idempotency key, a curl to try it)
  and every schedule with when it fires next, and **Fire now**. The same per agent under Agents -> `orders` ->
  Triggers.
- **Threads**: one per order, labelled `llamphouse.label` and `customer`; open A-1001's to see both events
  and the state they built. Each run's input comes from the trigger (`trigger` sender, named `shop`, `ops` or
  `orders:daily-digest`).

## The code

```yaml
# agents.yaml
triggers:
  - webhook:
      name: shop
      path: /hooks/shop
      signature: {header: X-Shop-Signature, secret_env: SHOP_SECRET, prefix: "sha256="}
      thread: {label: $.order.id}            # one thread per order
      message: {text: $.summary}             # the input's text; the whole payload comes along as data
      thread_metadata: {customer: $.order.customer}
      idempotency: {key: $.event_id}         # the same event again: the first answer, 200
      busy: send                             # the order's run is busy: the event goes to its inbox
  - webhook: {name: ops, path: /hooks/ops, secret_env: OPS_TOKEN, message: {text: $.text}}
  - schedule: {name: daily-digest, cron: "0 7 * * MON-FRI", timezone: Europe/Amsterdam,
               input: {text: Daily digest}, thread: {label: digest}}
  - schedule: {name: stuck-check, every: 1m, input: {text: Look for stuck orders},
               thread: {label: stuck-check}, busy: skip}
```

```python
class OrdersAgent(Agent):
    async def run(self, ctx) -> str:
        sender = ctx.input.sender                 # kind trigger, id: the trigger's name
        if sender.id == "shop":
            event = ctx.input.data                # the payload the shop POSTed
            ...
```

- A webhook is checked by a bearer secret (`secret` / `secret_env`), a `signature` (an HMAC-SHA256 of the
  raw body), or `public: true`. Webhooks are mounted next to the API, outside its auth: each checks its own
  secret. The answer is `{thread, run, action}`, 202.
- `busy` says what happens when the thread's run is still busy: `send` the message to it, `skip` it, or
  `reject` it (409). Schedules default to `skip`.
- `cron` (with `timezone`), `every` (`30s`, `15m`, `1h`) or once `at` (an ISO time). Schedules are kept by the
  store and fired once by whichever worker sees them due first; a fire missed while nothing ran happens once
  when one runs again.
- Agents schedule runs themselves too: `ctx.require(Scheduler).schedule(agent, delay=..., key=...)`.
