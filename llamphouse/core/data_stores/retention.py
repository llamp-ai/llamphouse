from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

@dataclass(frozen=True)
class RetentionPolicy:
    """Retention policy based on created_at cutoff."""
    ttl_days: int
    run_hour: int = 2
    run_minute: int = 0
    tz = timezone.utc
    dry_run: bool = False
    enabled: bool = True
    batch_size: Optional[int] = None
    now_fn: Optional[Callable[[], datetime]] = None
    log_fn: Optional[Callable[[str], None]] = None

    def batch_limit(self) -> Optional[int]:
        if self.batch_size is None or self.batch_size <= 0:
            return None
        return self.batch_size
    
    def now(self) -> datetime:
        return self.now_fn() if self.now_fn else datetime.now(self.tz)

    def cutoff(self) -> datetime:
        now = self.now_fn() if self.now_fn else datetime.now(timezone.utc)
        return now - timedelta(days=self.ttl_days)
    
    def next_run_at(self) -> datetime:
        now = self.now()
        run_at = now.replace(hour=self.run_hour, minute=self.run_minute, second=0, microsecond=0)
        if now >= run_at:
            run_at = run_at + timedelta(days=1)
        return run_at

    def sleep_seconds(self) -> float:
        now = self.now()
        return max(0.0, (self.next_run_at() - now).total_seconds())
    
    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)
    
@dataclass
class PurgeStats:
    threads: int = 0
    messages: int = 0
    runs: int = 0
    run_steps: int = 0