from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

@dataclass(frozen=True)
class RetentionPolicy:
    """Retention policy based on created_at cutoff."""
    ttl_days: int
    interval_seconds: int = 24 * 60 * 60 # once a day
    dry_run: bool = False
    enabled: bool = True
    batch_size: Optional[int] = None
    now_fn: Optional[Callable[[], datetime]] = None
    log_fn: Optional[Callable[[str], None]] = None

    def batch_limit(self) -> Optional[int]:
        if self.batch_size is None or self.batch_size <= 0:
            return None
        return self.batch_size

    def cutoff(self) -> datetime:
        now = self.now_fn() if self.now_fn else datetime.now(timezone.utc)
        return now - timedelta(days=self.ttl_days)
    
    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)
    
@dataclass
class PurgeStats:
    threads: int = 0
    messages: int = 0
    runs: int = 0
    run_steps: int = 0