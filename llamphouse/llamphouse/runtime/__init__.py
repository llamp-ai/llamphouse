"""The llamphouse 2.0 runtime: stores, the event log, runs, workers.

Built next to the 1.x core (`llamphouse.core`), which goes away once each
piece is ported. Modules here import only the standard library and
`llamphouse.sdk`, unless they are a backend behind an extra.
"""

from .engine import Engine, RunChange, StoreUnavailable  # noqa: E402


def serve(*args, **kwargs):  # noqa: ANN002, ANN003 - see llamphouse.runtime.server.serve
    """Serve agents over HTTP, with Compass (`llamphouse.runtime.server.serve`)."""
    from .server import serve as _serve

    return _serve(*args, **kwargs)


__all__ = ["Engine", "RunChange", "StoreUnavailable", "serve"]
