"""External subscribers: parties outside llamphouse that hear about runs without holding a stream open.

    base.py       Subscription - a URL, a filter, a format; what a delivery carries
    delivery.py   delivering the outbox: POST, again with backoff, dead after the attempts

Subscriptions are made over the API (`POST /v2/subscriptions`) or by an A2A client (push notification
configs); deliveries are written to the outbox with the change they report, and the workers send them.
"""

from .base import A2A_STATES, FORMATS, Subscription
from .delivery import backoff, deliver_due

__all__ = ["Subscription", "deliver_due", "backoff", "FORMATS", "A2A_STATES"]
