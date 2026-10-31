"""Triggers: what starts an agent besides a person - an HTTP call from another system, or the clock.

In an agent file:

    triggers:
      - webhook:
          path: /hooks/orders                  # POST here
          secret_env: ORDERS_SECRET            # Authorization: Bearer <secret>; or signature: {...}; or public: true
          thread: {label: $.order.id}          # one thread per order (a new thread per call without it)
          message: {text: $.summary}           # the input's text; the whole payload comes along as data
          thread_metadata: {customer: $.customer.id}
          idempotency: {key: $.event_id}       # the same event twice: one run, the first answer again
          busy: send                           # the thread's run is busy: send it there (send), skip, or reject
      - schedule:
          cron: "0 7 * * MON-FRI"              # or every: 15m, or at: 2026-12-01T09:00:00Z (once)
          timezone: Europe/Brussels
          input: {text: Daily digest}
          thread: {label: digest}              # the same thread every time (a new one without it)
          busy: skip

    base.py       payload paths, busy policies, thread labels
    webhook.py    WebhookTrigger
    schedule.py   ScheduleTrigger, and when a schedule fires next
    cron.py       cron expressions and intervals

A trigger's message comes from `Sender.trigger(name)`. Schedules are kept by the store and fired by the
workers' sweeper - once, whichever process sees them due first; a fire missed while nothing ran happens once
when one runs again. Agents schedule runs themselves with the Scheduler capability.
"""

from .base import BUSY, LABEL_KEY, resolve
from .cron import Cron, CronError, parse_every
from .schedule import ScheduleTrigger, message_of, next_time
from .webhook import WebhookTrigger

__all__ = ["WebhookTrigger", "ScheduleTrigger", "Cron", "CronError", "parse_every", "next_time", "message_of",
           "resolve", "BUSY", "LABEL_KEY"]
