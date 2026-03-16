class QueueError(Exception):
    """Base class for queue-related errors."""

class QueueRateLimitError(QueueError):
    """Raised when enqueue exceeds the rate limit (e.g. > max_per_minute)."""
    def __init__(self, key: str, limit: int, window_seconds: int):
        super().__init__(f"rate limit exceeded for {key}: {limit}/{window_seconds}s")
        self.key = key
        self.limit = limit
        self.window_seconds = window_seconds

class QueueRetryExceeded(QueueError):
    """Raised when a message exceeds max retry attempts."""
    def __init__(self, run_id: str | None = None, attempts: int | None = None, max_attempts: int | None = None):
        super().__init__("max exceed retry")
        self.run_id = run_id
        self.attempts = attempts
        self.max_attempts = max_attempts