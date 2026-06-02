"""Stdlib HTTP server for the Contractor explorer UI.

A dependency-free read-only browser over the project's agent prompts, task
templates, workflow pipelines, and skills. Serves a self-contained static SPA
plus a small JSON API; everything is read live from the package tree on each
request so editing a prompt and refreshing is enough to see the change.

Run it with ``analytics-ui`` (console script) or ``python -m analytics_ui``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import socket
import threading
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from analytics_ui import comments, evals, reader, registry, tools_introspect

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


# ───────────────────────── API ─────────────────────────


def _overview() -> dict[str, Any]:
    agents = reader.list_agents()
    tasks = reader.list_tasks()
    skills = reader.list_skills()
    workflows = registry.list_workflows()
    eval_runs = evals.list_eval_runs()
    return {
        "counts": {
            "agents": len(agents),
            "tasks": len(tasks),
            "skills": len(skills),
            "workflows": len(workflows),
            "evals": len(eval_runs),
        },
        "agents": [a.__dict__ for a in agents],
        "tasks": [t.__dict__ for t in tasks],
        "skills": [s.__dict__ for s in skills],
        "workflows": workflows,
        "evals": eval_runs,
    }


def _route_api(parts: list[str]) -> Any | None:
    """Resolve an /api/* path (already split + unquoted) to a JSON payload.

    Returns ``None`` for an unknown route (→ 404); raises ``KeyError`` via the
    caller's None-checks for missing entities.
    """
    if parts == ["overview"]:
        return _overview()
    if parts == ["crossrefs"]:
        return registry.crossrefs()

    if parts and parts[0] == "agents":
        if len(parts) == 1:
            return [a.__dict__ for a in reader.list_agents()]
        if len(parts) == 2:
            info = reader.get_agent(parts[1])
            if info is None:
                return None
            try:
                info["tools"] = tools_introspect.agent_tools(parts[1])
            except Exception:  # pragma: no cover - never let introspection break the view
                logger.exception("tool introspection failed for %s", parts[1])
                info["tools"] = None
            return info
        if len(parts) == 3:
            return reader.get_agent_version(parts[1], parts[2])

    if parts and parts[0] == "tasks":
        if len(parts) == 1:
            return [t.__dict__ for t in reader.list_tasks()]
        if len(parts) == 2:
            return reader.get_task(parts[1])
        if len(parts) == 3:
            return reader.get_task(parts[1], parts[2])

    if parts and parts[0] == "skills":
        if len(parts) == 1:
            return [s.__dict__ for s in reader.list_skills()]
        if len(parts) == 2:
            return reader.get_skill(parts[1])
        if len(parts) == 4 and parts[2] == "ref":
            return reader.get_skill_reference(parts[1], parts[3])

    if parts and parts[0] == "workflows":
        if len(parts) == 1:
            return registry.list_workflows()
        if len(parts) == 2:
            return registry.get_workflow(parts[1])

    if parts and parts[0] == "evals":
        if len(parts) == 1:
            return evals.list_eval_runs()
        if len(parts) == 2:
            return evals.get_eval_run(parts[1])

    return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "ContractorUI/1.0"

    # quieter, prettier logs
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/comments":
            self._list_comments(parse_qs(parsed.query))
            return
        if path.startswith("/api/"):
            self._handle_api(path)
            return
        self._handle_static(path)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/comments":
            self._send_json({"error": "not found"}, status=404)
            return
        self._create_comment()

    def do_PUT(self) -> None:  # noqa: N802
        cid = self._comment_id_from_path()
        if cid is None:
            self._send_json({"error": "not found"}, status=404)
            return
        self._update_comment(cid)

    def do_DELETE(self) -> None:  # noqa: N802
        cid = self._comment_id_from_path()
        if cid is None:
            self._send_json({"error": "not found"}, status=404)
            return
        if comments.delete_comment(cid):
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, status=404)

    # -- comments --

    def _comment_id_from_path(self) -> int | None:
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "comments":
            try:
                return int(parts[2])
            except ValueError:
                return None
        return None

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def _list_comments(self, q: dict[str, list[str]]) -> None:
        def first(k: str) -> str | None:
            v = q.get(k)
            return v[0] if v else None

        rows = comments.list_comments(
            kind=first("kind"), target_id=first("id"), version=first("version")
        )
        self._send_json(rows)

    def _create_comment(self) -> None:
        d = self._read_json()
        try:
            row = comments.add_comment(
                kind=str(d.get("kind", "")),
                target_id=str(d.get("id", "")),
                version=str(d.get("version", "")),
                line_start=d.get("line_start"),
                line_end=d.get("line_end", d.get("line_start")),
                body=str(d.get("body", "")),
            )
        except comments.CommentError as e:
            self._send_json({"error": str(e)}, status=400)
            return
        self._send_json(row, status=201)

    def _update_comment(self, cid: int) -> None:
        d = self._read_json()
        try:
            row = comments.update_comment(cid, str(d.get("body", "")))
        except comments.CommentError as e:
            self._send_json({"error": str(e)}, status=400)
            return
        if row is None:
            self._send_json({"error": "not found"}, status=404)
            return
        self._send_json(row)

    # -- API (read-only resources) --

    def _handle_api(self, path: str) -> None:
        raw = path[len("/api/") :].strip("/")
        parts = [unquote(p) for p in raw.split("/") if p]
        try:
            payload = _route_api(parts)
        except Exception:  # pragma: no cover - defensive
            logger.exception("API error for %s", path)
            self._send_json({"error": "internal error"}, status=500)
            return
        if payload is None:
            self._send_json({"error": "not found"}, status=404)
            return
        self._send_json(payload)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- static --

    def _handle_static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        target = (STATIC_DIR / rel).resolve()
        # Confine to STATIC_DIR; unknown routes fall back to the SPA shell.
        if not str(target).startswith(str(STATIC_DIR)) or not target.is_file():
            target = STATIC_DIR / "index.html"
        if not target.is_file():
            self._send_json({"error": "static assets missing"}, status=500)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            _CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ───────────────────────── entrypoint ─────────────────────────


def _find_open_port(host: str, preferred: int) -> int:
    """Return ``preferred`` if free, else an OS-assigned open port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    on_ready: Callable[[str], None] | None = None,
) -> ThreadingHTTPServer:
    port = _find_open_port(host, port)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    if on_ready:
        on_ready(url)
    if open_browser:
        threading.Timer(0.4, lambda: _try_open(url)).start()
    return httpd


def _try_open(url: str) -> None:
    with contextlib.suppress(Exception):  # pragma: no cover
        webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analytics-ui",
        description="Browse Contractor agent prompts, tasks, pipelines and skills.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser", action="store_true", help="do not auto-open a browser tab"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    httpd = serve(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        on_ready=lambda url: print(
            f"\n  Contractor explorer → {url}\n  (Ctrl-C to stop)\n", flush=True
        ),
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
