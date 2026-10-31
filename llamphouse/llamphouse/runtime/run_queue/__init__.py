"""The run queue: the doorbell that tells workers there is work (the store says what the work is).

    base.py     JobQueue
    memory.py   MemoryQueue - in this process
    redis.py    RedisQueue - a Redis list

The runtime file names one: `run_queue: {redis: {url: ...}}`, or your own `{my_pkg:MyQueue: {...}}`.
"""

from .base import JobQueue
from .memory import MemoryQueue
from .redis import RedisQueue

__all__ = ["JobQueue", "MemoryQueue", "RedisQueue"]
