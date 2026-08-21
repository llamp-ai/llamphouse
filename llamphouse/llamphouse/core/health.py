from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class HealthCheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class HealthCheckResult:
    name: str
    module: str
    status: HealthCheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_(cls, name: str, module: str, message: str, **details: Any) -> "HealthCheckResult":
        return cls(name, module, HealthCheckStatus.PASS, message, details)

    @classmethod
    def warn(cls, name: str, module: str, message: str, **details: Any) -> "HealthCheckResult":
        return cls(name, module, HealthCheckStatus.WARN, message, details)

    @classmethod
    def fail(cls, name: str, module: str, message: str, **details: Any) -> "HealthCheckResult":
        return cls(name, module, HealthCheckStatus.FAIL, message, details)


class HealthCheckable(Protocol):
    async def health_check(self) -> HealthCheckResult:
        ...
