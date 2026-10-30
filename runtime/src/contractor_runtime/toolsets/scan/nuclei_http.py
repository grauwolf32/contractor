"""Admit the normal HTTP subset before Nuclei can evaluate template expressions.

Native selection reads inert metadata copies. Selected original bytes retain their
ordinary manifest identity; private execution copies disable native cookie state.
No original variables, DSL or protocol requests reach the engine before admission.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import yaml

from contractor_runtime.toolsets.scan.nuclei_package import PackageError, PackageSnapshot
from contractor_runtime.toolsets.scan.nuclei_rulepack import parse_template

SUPPORTED_ENGINE_VERSION = "v3.11.1"
_ROOT_FIELDS = {"id", "info", "http", "variables", "constants", "stop-at-first-match"}
_HTTP_FIELDS = {
    "id",
    "method",
    "path",
    "headers",
    "body",
    "matchers",
    "matchers-condition",
    "extractors",
    "stop-at-first-match",
    "redirects",
    "host-redirects",
    "max-redirects",
    "disable-cookie",
    "cookie-reuse",
    "req-condition",
    "iterate-all",
    "payloads",
    "attack",
}
_PURE_HELPERS = {
    "base64",
    "base64_decode",
    "concat",
    "contains",
    "contains_all",
    "contains_any",
    "ends_with",
    "hex_decode",
    "hex_encode",
    "index_of",
    "len",
    "md5",
    "mmh3",
    "rand_base",
    "rand_int",
    "rand_text_alpha",
    "rand_text_alphanumeric",
    "rand_text_numeric",
    "regex",
    "regex_all",
    "regex_any",
    "replace",
    "replace_regex",
    "sha1",
    "sha256",
    "starts_with",
    "substr",
    "substring",
    "to_lower",
    "to_number",
    "to_string",
    "to_upper",
    "trim",
    "trim_left",
    "trim_prefix",
    "trim_right",
    "trim_space",
    "trim_suffix",
    "url_decode",
    "url_encode",
}
_EXPRESSIONS = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
_CALLS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise PackageError("scan_timeout")


def _invalid() -> PackageError:
    return PackageError("nuclei_selection_invalid")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as stream:
        os.fchmod(stream.fileno(), 0o400)
        yaml.safe_dump(value, stream, sort_keys=False)


def _read(snapshot: PackageSnapshot, path: str, deadline: float) -> dict:
    try:
        return parse_template((snapshot.root / path).read_bytes(), deadline)
    except PackageError as error:
        if str(error) == "scan_timeout":
            raise
        raise _invalid() from None
    except (OSError, UnicodeError):
        raise _invalid() from None


def selection_package(snapshot: PackageSnapshot, destination: Path, deadline: float) -> Path:
    """Native filtering sees ID/info/type but cannot evaluate original variables."""
    destination.mkdir(mode=0o700)
    count = 0
    for record in snapshot.files:
        _deadline(deadline)
        if not record["path"].endswith((".yaml", ".yml")):
            continue
        value = _read(snapshot, record["path"], deadline)
        if "http" not in value:
            continue
        if not isinstance(value.get("id"), str) or not isinstance(value.get("info"), dict):
            raise _invalid()
        metadata = {
            "id": value["id"],
            "info": value["info"],
            "http": [
                {
                    "method": "GET",
                    "path": ["{{BaseURL}}/"],
                    "matchers": [{"type": "status", "status": [200]}],
                }
            ],
        }
        _write(destination / record["path"], metadata)
        count += 1
    if not count:
        raise _invalid()
    return destination


def source_selection(
    snapshot: PackageSnapshot, selection: Path, output: bytes, deadline: float
) -> tuple[bytes, list[str]]:
    """Map native metadata selection back to exact original package paths."""
    index = {record["path"] for record in snapshot.files}
    paths, seen = [], set()
    try:
        lines = output.decode("utf-8", errors="strict").splitlines()
    except UnicodeError:
        raise _invalid() from None
    for line in lines:
        _deadline(deadline)
        line = line.strip()
        if not line or line.startswith("Listing available  nuclei templates for "):
            continue
        path = Path(line)
        try:
            relative = path.relative_to(selection).as_posix() if path.is_absolute() else line
        except ValueError:
            raise _invalid() from None
        if relative not in index or relative in seen:
            raise _invalid()
        paths.append(relative)
        seen.add(relative)
    if not paths:
        raise _invalid()
    return ("\n".join(str(snapshot.root / path) for path in paths) + "\n").encode(), paths


def _expressions(value, *, dsl=False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _expressions(child, dsl=key == "dsl")
    elif isinstance(value, list):
        for child in value:
            _expressions(child, dsl=dsl)
    elif isinstance(value, str):
        expressions = [value] if dsl else _EXPRESSIONS.findall(value)
        for expression in expressions:
            if any(name not in _PURE_HELPERS for name in _CALLS.findall(expression)):
                raise _invalid()
        if "interactsh" in value.lower():
            raise _invalid()


def execution_package(
    snapshot: PackageSnapshot,
    selected: list[str],
    destination: Path,
    deadline: float,
    credential_id: str | None,
) -> Path:
    """Admit normal HTTP requests; every other execution feature fails closed."""
    destination.mkdir(mode=0o700)
    for path in selected:
        _deadline(deadline)
        value = _read(snapshot, path, deadline)
        if (
            set(value) - _ROOT_FIELDS
            or not isinstance(value.get("http"), list)
            or not value["http"]
        ):
            raise _invalid()
        _expressions({key: item for key, item in value.items() if key != "info"})
        for request in value["http"]:
            if not isinstance(request, dict) or set(request) - _HTTP_FIELDS:
                raise _invalid()
            if request.get("method", "GET") not in {
                "GET",
                "HEAD",
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
                "OPTIONS",
                "TRACE",
            }:
                raise _invalid()
            paths = request.get("path")
            if (
                not isinstance(paths, list)
                or not paths
                or any(not isinstance(p, str) for p in paths)
            ):
                raise _invalid()
            headers = request.get("headers", {})
            if not isinstance(headers, dict):
                raise _invalid()
            for name, content in headers.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(content, str)
                    or "{{" in name
                    or name.lower() in {"upgrade", "proxy-authorization", "proxy-connection"}
                    or (name.lower() == "connection" and "upgrade" in content.lower())
                    or name.lower() == "cookie"
                    or (credential_id is not None and name.lower() == "authorization")
                ):
                    raise _invalid()
            payloads = request.get("payloads", {})
            if not isinstance(payloads, dict) or any(
                not isinstance(items, list)
                or not items
                or any(
                    not isinstance(item, (str, int, float)) or isinstance(item, bool)
                    for item in items
                )
                for items in payloads.values()
            ):
                raise _invalid()
            # Shared Runtime identity jars own learned cookies and origin isolation.
            request["disable-cookie"] = True
            request["redirects"] = False
            request["host-redirects"] = False
        _write(destination / path, value)
    return destination
