"""Compass: see, debug and learn what your agents do - a web app at /compass, on the API at /v2/compass.

    frontend/   the app's source (Vue 3, Vite): npm install && npm run build -> static/
    static/     the built app, shipped in the package
    explain.py  what Compass says about a run: why it is where it is, its story, steps, trees
    routes.py   the API the app reads (behind the deployment's auth and access)

The app needs nothing else: no CDN, no build at runtime - it works air-gapped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STATIC = Path(__file__).resolve().parent / "static"
FRESH = {"Cache-Control": "no-cache"}  # file names are stable across versions: revalidate (ETag) on every load


def mount(app: Any, gw: Any) -> None:
    """The Compass API on /v2/compass, and the app on /compass (with / sending you there)."""
    from fastapi.responses import FileResponse, HTMLResponse, Response

    from .routes import mount_api

    mount_api(app, gw)

    def index() -> HTMLResponse:
        page = STATIC / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>Compass is not built</h1><p>cd llamphouse/runtime/compass/frontend && "
                                "npm install && npm run build</p>", status_code=503)
        return HTMLResponse(page.read_text(), headers=FRESH)

    @app.get("/compass", include_in_schema=False)
    async def compass_root() -> Any:
        return Response(status_code=307, headers={"Location": "/compass/"})

    @app.get("/compass/{path:path}", include_in_schema=False)
    async def compass_files(path: str) -> Any:
        file = (STATIC / path).resolve()
        if path and file.is_file() and STATIC in file.parents:
            return FileResponse(file, headers=FRESH)
        return index()  # the app's own routes

    @app.get("/", include_in_schema=False)
    async def root() -> Response:
        return Response(status_code=307, headers={"Location": "/compass/"})


__all__ = ["mount", "STATIC"]
