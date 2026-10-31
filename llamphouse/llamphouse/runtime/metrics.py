"""Numbers for an autoscaler and for dashboards, in the Prometheus text format (`GET /metrics`).

    llamphouse_jobs_queued{lane}               jobs waiting for a worker slot - shared by the deployment
    llamphouse_sessions_revising               sessions with a revision in flight - shared
    llamphouse_sessions_waiting_for_worker     ... of which no worker has picked it up yet - shared
    llamphouse_worker_slots{takes}             this process's slots: "runs+revisions", "runs" or "revisions"
    llamphouse_jobs_running{lane}              jobs this process works on now
    llamphouse_jobs_done_total{lane}           jobs this process finished

Scale workers on `llamphouse_jobs_queued` (KEDA, a Kubernetes HPA on an external metric, or your own): work
piling up in a lane means its slots are all busy. Revisions are their own lane, so `revision_workers` can be
scaled apart from runs. API processes serve /metrics next to the API; worker processes with
`--metrics-port` (and /health there too).
"""

from __future__ import annotations

import json
from typing import Any

from .run_queue.base import LANES


def _line(name: str, value: float, labels: dict[str, str] | None = None) -> str:
    if labels:
        inner = ",".join(f'{k}="{v}"' for k, v in labels.items())
        return f"{name}{{{inner}}} {value}"
    return f"{name} {value}"


async def render(engine: Any) -> str:
    out: list[str] = []

    def metric(name: str, kind: str, help_text: str, samples: list[tuple[float, dict[str, str] | None]]) -> None:
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {kind}")
        out.extend(_line(name, v, labels) for v, labels in samples)

    import llamphouse

    metric("llamphouse_info", "gauge", "This process.",
           [(1, {"engine": engine.id, "version": getattr(llamphouse, "__version__", "?")})])
    try:
        depth = await engine.queue.depth()
    except Exception:  # noqa: BLE001 - a queue that can't tell: no sample
        depth = {}
    if depth:
        metric("llamphouse_jobs_queued", "gauge", "Jobs waiting for a worker slot, per lane (the deployment's).",
               [(depth.get(lane, 0), {"lane": lane}) for lane in LANES])
    revising = waiting = None
    try:
        from .codec import decode

        found = await engine.store.list_sessions(revising=True, limit=1000)
        revising = len(found)
        waiting = sum(1 for row in found if getattr(decode(json.loads(row["text"])), "revision_owner", None) is None)
    except Exception:  # noqa: BLE001
        pass
    if revising is not None:
        metric("llamphouse_sessions_revising", "gauge", "Sessions with a revision in flight (the deployment's).",
               [(revising, None)])
        metric("llamphouse_sessions_waiting_for_worker", "gauge",
               "Sessions whose revision no worker has picked up yet (the deployment's).", [(waiting or 0, None)])
    slots = []
    if engine.revision_workers:
        slots = [(engine.n_workers, {"takes": "runs"}), (engine.revision_workers, {"takes": "revisions"})]
    else:
        slots = [(engine.n_workers, {"takes": "runs+revisions"})]
    metric("llamphouse_worker_slots", "gauge", "This process's worker slots, by what they take.", slots)
    metric("llamphouse_jobs_running", "gauge", "Jobs this process works on now, per lane.",
           [(engine.busy.get(lane, 0), {"lane": lane}) for lane in LANES])
    metric("llamphouse_jobs_done_total", "counter", "Jobs this process finished, per lane.",
           [(engine.done_jobs.get(lane, 0), {"lane": lane}) for lane in LANES])
    return "\n".join(out) + "\n"
