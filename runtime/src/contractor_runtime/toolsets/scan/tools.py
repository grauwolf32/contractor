"""Typed wrappers for optional nuclei, sqlmap and naabu executables."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding
from contractor_runtime.toolsets.common.artifacts import ArtifactClientFactory, _unconfigured_client
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.common.metrics import ToolMetrics
from contractor_runtime.toolsets.scan.http_request import (
    MAX_REQUEST_ARTIFACT_BYTES,
    parse_http_request,
)
from contractor_runtime.toolsets.scan.process import ProcessResult, run_process
from contractor_runtime.workspace import AllocationWorkspace

PROBE_TIMEOUT_SECONDS = 2.0
MAX_RESULTS = 100
MAX_RESULTS_BYTES = 128 * 1024
PrepareInvocation = Callable[[Path], Awaitable[list[str]]]


class ScanInputError(ToolInputError):
    """Runtime-authored validation diagnostics, without supplied argument values."""


class ScannerUnavailable(Exception):
    """An adapter-owned prerequisite is unavailable; the message is a fixed error code."""


class _ScanSession:
    def __init__(self, workspace, executables, templates, *, proxy_configured, artifact_client):
        self.workspace = workspace
        self.executables = executables
        self.templates = templates
        self.proxy_configured = proxy_configured
        self.artifact_client = artifact_client
        self._closed = False
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def execute(
        self,
        tool: ScanTool,
        arguments: list[str],
        timeout: int,
        *,
        prepare: PrepareInvocation | None = None,
        observation: Callable[[ProcessResult], dict] | None = None,
    ) -> dict:
        async with self._lock:
            if self._closed:
                result = ProcessResult(None, error_code="scan_closed")
            elif self.proxy_configured:
                # The generic env-based launcher does not guarantee scanner routing.
                # Never silently bypass the allocation's configured proxy.
                result = ProcessResult(None, error_code="scan_proxy_unsupported")
            elif not self.executables[tool.name]:
                result = ProcessResult(None, error_code="scanner_unavailable")
            else:
                self._task = asyncio.create_task(self._execute(tool, arguments, timeout, prepare))
                try:
                    result = await self._task
                finally:
                    self._task = None
        return (observation or tool.observation)(result)

    async def _execute(
        self,
        tool: ScanTool,
        arguments: list[str],
        timeout: int,
        prepare: PrepareInvocation | None,
    ) -> ProcessResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"{tool.name}-", dir=self.workspace) as root:
            directory = Path(root)
            try:
                prepared = tool.prepare(directory, self.templates)
                if prepare is not None:
                    # Artifact retrieval and materialization share the scan deadline.
                    async with asyncio.timeout(timeout):
                        prepared += await prepare(directory)
            except TimeoutError:
                return ProcessResult(None, error_code="scan_timeout")
            except ScannerUnavailable as error:
                return ProcessResult(None, error_code=str(error))
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                return ProcessResult(None, error_code="scan_timeout")
            command = [self.executables[tool.name], *arguments, *prepared]
            return await run_process(command, directory, remaining)

    async def close(self) -> None:
        self._closed = True
        task = self._task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # Join any caller holding the lock before allocation scratch deletion.
        async with self._lock:
            pass


class ScanTool:
    """Trusted scanner adapter; registration never comes from invocation input."""

    name: str
    binary: str
    version_arguments: tuple[str, ...]
    description: str

    def __init__(self, session: _ScanSession, metrics: ToolMetrics) -> None:
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        await self._session.close()

    def prepare(self, directory: Path, templates: Path) -> list[str]:
        """Create private files and return additional argv before process launch."""
        return []

    def observation(self, result: ProcessResult) -> dict:
        return {**result.observation(), "scanner": self.binary}

    async def _call(
        self,
        timeout: int,
        arguments: Callable[[], list[str]],
        *,
        prepare: PrepareInvocation | None = None,
        observation: Callable[[ProcessResult], dict] | None = None,
    ) -> dict:
        started = time.monotonic()
        result = None
        error = None
        try:
            _integer(timeout, "timeout_seconds", 1, 3600)
            result = await self._session.execute(
                self, arguments(), timeout, prepare=prepare, observation=observation
            )
            return result
        except asyncio.CancelledError:
            error = RuntimeError("scan cancelled")
            raise
        except Exception as caught:
            error = caught
            raise
        finally:
            # Targets, credentials, payloads and scanner output never enter metrics.
            metric_result = (
                {key: result[key] for key in ("status", "exitCode", "errorCode")}
                if result is not None
                else None
            )
            if result is not None and result["errorCode"]:
                error = RuntimeError(result["errorCode"])
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                result=metric_result,
                error=error,
                duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            )


class JSONLinesScanTool(ScanTool):
    def observation(self, result: ProcessResult) -> dict:
        response = super().observation(result)
        items, truncated, invalid = _json_lines(result.stdout)
        response.update(results=items, resultsTruncated=truncated, invalidResultLines=invalid)
        if invalid and response["errorCode"] is None:
            response.update(status="failed", errorCode="invalid_scanner_output")
        return response


class NucleiTool(JSONLinesScanTool):
    name = "scan_nuclei"
    binary = "nuclei"
    version_arguments = ("-version",)
    description = """Scan one HTTP(S) URL with installed nuclei HTTP templates.

    Uses local templates, disables updates and external OAST callbacks. Calls are
    serialized within the allocation. A configured subprocess proxy is unsupported.
    Findings are scanner evidence and are not automatically published.

    Args:
        url: Absolute HTTP(S) target URL without embedded credentials.
        template_ids: Comma-separated template IDs or wildcard IDs; empty selects all.
        tags: Comma-separated template tags; empty applies no tag filter.
        severity: Comma-separated info, low, medium, high, critical, unknown values.
        rate_limit: Maximum requests per second, 1 through 1000; defaults to 10.
        timeout_seconds: Total scan deadline, 1 through 3600 seconds; defaults to 300.

    Returns:
        status, exitCode, errorCode, bounded stdout/stderr previews and truncation
        flags, durationMs, up to 100 JSON results, resultsTruncated and invalidResultLines.
        Timeout, overflow and execution errors are failures, not clean scans.
    """

    def prepare(self, directory: Path, templates: Path) -> list[str]:
        if not templates.is_dir():
            raise ScannerUnavailable("nuclei_templates_unavailable")
        # Prevent a first-run template install even on older nuclei versions.
        (directory / "nuclei-templates").mkdir()
        return ["-t", str(templates)]

    async def __call__(
        self,
        url: str,
        template_ids: str = "",
        tags: str = "",
        severity: str = "",
        rate_limit: int = 10,
        timeout_seconds: int = 300,
    ) -> dict:
        def arguments():
            _url(url)
            _integer(rate_limit, "rate_limit", 1, 1000)
            command = [
                "-u",
                url,
                "-jsonl",
                "-silent",
                "-no-color",
                "-disable-update-check",
                "-no-interactsh",
                "-type",
                "http",
                "-omit-raw",
                "-omit-template",
                "-disable-redirects",
                "-rate-limit",
                str(rate_limit),
                "-concurrency",
                "5",
                "-timeout",
                "10",
                "-retries",
                "1",
            ]
            for flag, value in (("-id", template_ids), ("-tags", tags), ("-severity", severity)):
                _tokens(value)
                if value:
                    command += [flag, value]
            if severity and set(severity.split(",")) - {
                "info",
                "low",
                "medium",
                "high",
                "critical",
                "unknown",
            }:
                raise ScanInputError("severity contains an unsupported value")
            return command

        return await self._call(timeout_seconds, arguments)


class SQLMapTool(ScanTool):
    name = "scan_sqlmap"
    binary = "sqlmap"
    version_arguments = ("--version",)
    description = """Check one prepared HTTP request or URL for SQL injection with sqlmap.

    Each call uses a fresh session. No database dumping or shell operations are
    requested. A configured subprocess proxy is unsupported. Review the output
    to distinguish detected injection, a negative check, and scanner diagnostics.

    Args:
        url: HTTP(S) URL for legacy URL mode; omit when using request_ref.
        parameter: Optional comma-separated parameter names to test.
        data: Optional POST body, at most 16384 UTF-8 bytes; empty sends GET.
        cookie: Optional Cookie header, at most 4096 UTF-8 bytes.
        level: sqlmap test coverage level, 1 through 5; defaults to 1.
        risk: sqlmap test risk level, 1 through 3; defaults to 1.
        timeout_seconds: Total scan deadline, 1 through 3600 seconds; defaults to 300.
        request_ref: Exact artifact reference with namespace, name and revision.
            Contains one schemaVersion 1 HTTP request (method, url, headers, body,
            testParameters). Mutually exclusive with url, parameter, data and cookie.
            Only supported UTF-8 text requests are accepted; see the scan request contract.

    Returns:
        status, exitCode, errorCode, bounded stdout/stderr previews, truncation
        flags and durationMs. completed means the process exited successfully,
        not that the target is free of SQL injection. Request mode suppresses raw
        diagnostics and returns exact requestArtifact and bounded injection evidence.
        Direct calls do not publish evidence; tool@1 publishes a report artifact.
    """

    def prepare(self, directory: Path, templates: Path) -> list[str]:
        return [f"--output-dir={directory / 'output'}", f"--tmp-dir={directory}"]

    async def __call__(
        self,
        url: str = "",
        parameter: str = "",
        data: str = "",
        cookie: str = "",
        level: int = 1,
        risk: int = 1,
        timeout_seconds: int = 300,
        request_ref: dict[str, str] | None = None,
    ) -> dict:
        # All preparation/projection state belongs to this invocation. Concurrent
        # callers never overwrite an input on the shared ToolInstance.
        exact_ref: ArtifactRef | None = None

        def arguments():
            nonlocal exact_ref
            if request_ref is None:
                _url(url)
                _tokens(parameter)
                _text(data, "data", 16384)
                _text(cookie, "cookie", 4096)
            else:
                if any(value != "" for value in (url, parameter, data, cookie)):
                    raise ScanInputError("request_ref cannot be combined with URL-mode arguments")
                try:
                    exact_ref = ArtifactRef.model_validate(request_ref).require_exact()
                    require_model_visible_binding(exact_ref.namespace, exact_ref.name)
                except (ValueError, TypeError):
                    raise ScanInputError(
                        "request_ref must be an accessible exact artifact ref"
                    ) from None
            _integer(level, "level", 1, 5)
            _integer(risk, "risk", 1, 3)
            command = [
                "--batch",
                "--disable-coloring",
                "--ignore-stdin",
                "--ignore-redirects",
                "--threads=1",
                "--timeout=10",
                "--retries=1",
                f"--level={level}",
                f"--risk={risk}",
            ]
            if request_ref is None:
                command.insert(0, f"--url={url}")
            if parameter:
                command += ["-p", parameter]
            if data:
                command.append(f"--data={data}")
            if cookie:
                command.append(f"--cookie={cookie}")
            return command

        async def prepare_request(directory: Path) -> list[str]:
            assert exact_ref is not None
            try:
                value = await self._session.artifact_client.read_artifact(
                    exact_ref, max_bytes=MAX_REQUEST_ARTIFACT_BYTES
                )
            except Exception:
                raise ScanInputError("request artifact is inaccessible or invalid") from None
            if value.media_type not in {
                "application/json",
                "application/vnd.contractor.http-request+json",
            }:
                raise ScanInputError("request artifact must contain HTTP request JSON")
            request = parse_http_request(value.data)
            path = directory / "request.http"
            try:
                # Exclusive creation and mode independent of the service umask.
                with path.open("xb") as output:
                    os.fchmod(output.fileno(), 0o600)
                    output.write(request.raw)
            except OSError:
                raise ScannerUnavailable("scan_request_file_unavailable") from None
            return [
                "-r",
                str(path),
                f"--method={request.method}",
                "--encoding=utf-8",
                "--drop-set-cookie",
                # The automatic WAF probe injects an unselected query parameter.
                "--skip-waf",
                "-p",
                ",".join(request.test_parameters),
            ]

        def request_observation(result: ProcessResult) -> dict:
            assert exact_ref is not None
            response = self.observation(result)
            # Scanner diagnostics can echo or transform credentials and payloads.
            # A finite vocabulary preserves evidence without substring-redaction
            # promises for arbitrary request bodies or vulnerable target responses.
            techniques = _sqlmap_techniques(result.stdout)
            response.update(
                stdout="",
                stderr="",
                diagnosticsRedacted=True,
                requestArtifact=exact_ref.model_dump(by_alias=True),
                injectionOutcome="reported" if techniques else "unknown",
                injectionTechniques=techniques,
            )
            return response

        return await self._call(
            timeout_seconds,
            arguments,
            prepare=prepare_request if request_ref is not None else None,
            observation=request_observation if request_ref is not None else None,
        )


class NaabuTool(JSONLinesScanTool):
    name = "scan_naabu"
    binary = "naabu"
    version_arguments = ("-version",)
    description = """Discover open TCP ports on one host with naabu CONNECT scanning.

    Does not require raw-socket privileges. A configured subprocess HTTP proxy is
    unsupported. Calls are serialized within the allocation.

    Args:
        host: One DNS hostname or IPv4/IPv6 address, without scheme, port or CIDR.
        ports: Comma-separated TCP ports or inclusive ranges, up to 4096 ports;
            defaults to 80,443. Each port must be from 1 through 65535.
        rate: Maximum probes per second, 1 through 1000; defaults to 100.
        timeout_seconds: Total scan deadline, 1 through 3600 seconds; defaults to 60.

    Returns:
        status, exitCode, errorCode, bounded stdout/stderr previews and truncation
        flags, durationMs, up to 100 JSON results, resultsTruncated and invalidResultLines.
        Failed or truncated results do not establish that remaining ports are closed.
    """

    async def __call__(
        self,
        host: str,
        ports: str = "80,443",
        rate: int = 100,
        timeout_seconds: int = 60,
    ) -> dict:
        def arguments():
            _host(host)
            _ports(ports)
            _integer(rate, "rate", 1, 1000)
            return [
                "-host",
                host,
                "-p",
                ports,
                "-scan-type",
                "c",
                "-json",
                "-silent",
                "-no-color",
                "-disable-update-check",
                "-rate",
                str(rate),
                "-c",
                "10",
                "-retries",
                "1",
                "-timeout",
                "1000",
            ]

        return await self._call(timeout_seconds, arguments)


SCANNERS = (NucleiTool, SQLMapTool, NaabuTool)


class ScanToolsetFactory:
    ref = "scan@1"

    def __init__(
        self,
        artifact_client_factory: ArtifactClientFactory | None = None,
        *,
        templates_directory: Path | None = None,
        scanners: Sequence[type[ScanTool]] = SCANNERS,
    ) -> None:
        self._clients = artifact_client_factory or _unconfigured_client
        self._scanners = {scanner.name: scanner for scanner in scanners}
        if len(self._scanners) != len(scanners):
            raise ValueError("duplicate scan tool names")
        self.exported_tools = frozenset(self._scanners)
        self.infrastructure_channels = MappingProxyType(
            {name: frozenset({"runtime-subprocess-launcher"}) for name in self._scanners}
        )
        self._templates = (
            templates_directory
            or Path(os.environ.get("NUCLEI_TEMPLATES_DIR", Path.home() / "nuclei-templates"))
        ).resolve()
        self._executables = {
            scanner.name: (
                str(Path(executable).absolute())
                if (executable := shutil.which(scanner.binary))
                else None
            )
            for scanner in scanners
        }

    async def probe(self) -> frozenset[str]:
        async def available(name: str) -> bool:
            executable = self._executables[name]
            if executable is None:
                return False
            try:
                with tempfile.TemporaryDirectory(prefix="contractor-scan-probe-") as root:
                    result = await run_process(
                        [executable, *self._scanners[name].version_arguments],
                        Path(root),
                        PROBE_TIMEOUT_SECONDS,
                    )
                return result.error_code is None
            except OSError:
                return False

        # Independent deadlines: one hung or broken binary cannot hide another.
        results = await asyncio.gather(*(available(name) for name in self._scanners))
        return frozenset(
            name for name, present in zip(self._scanners, results, strict=True) if present
        )

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, namespace, project_workspace
        if set(selected) - self.exported_tools:
            raise ValueError("unknown selected scan tools")
        if not selected:
            return {}
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("scan@1 requires State.metrics")
        session = _ScanSession(
            workspace.path,
            self._executables,
            self._templates,
            proxy_configured=adapter_handles.tool_subprocess is not None,
            artifact_client=self._clients(allocation_id, runtime_settings),
        )
        return {name: self._scanners[name](session, metrics) for name in selected}


def _integer(value: int, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ScanInputError(f"{name} must be an integer from {low} through {high}")


def _text(value: str, name: str, limit: int) -> None:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > limit
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ScanInputError(f"{name} exceeds its text limit or contains control characters")


def _tokens(value: str) -> None:
    _text(value, "filter", 2048)
    if value and not re.fullmatch(
        r"[A-Za-z0-9_*][A-Za-z0-9_.*-]*(,[A-Za-z0-9_*][A-Za-z0-9_.*-]*)*", value
    ):
        raise ScanInputError("filters must be comma-separated names without paths or spaces")


def _url(value: str) -> None:
    _text(value, "url", 8192)
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and not any(character.isspace() for character in value)
        )
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            valid = False
        if not valid:
            raise ValueError
        _host(parsed.hostname)
    except ValueError:
        raise ScanInputError(
            "url must be an HTTP(S) URL without credentials or a fragment"
        ) from None


def _host(value: str) -> None:
    _text(value, "host", 253)
    try:
        ipaddress.ip_address(value)
        return
    except ValueError:
        pass
    if not value or any(
        not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
        for label in value.removesuffix(".").split(".")
    ):
        raise ScanInputError("host must be one DNS hostname or IP address")


def _ports(value: str) -> None:
    _text(value, "ports", 4096)
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
        raise ScanInputError("ports must be comma-separated TCP ports or ranges")
    count = 0
    for item in value.split(","):
        first, _, last = item.partition("-")
        low, high = int(first), int(last or first)
        if not 1 <= low <= high <= 65535:
            raise ScanInputError("port ranges must be ordered and within 1 through 65535")
        count += high - low + 1
    if count > 4096:
        raise ScanInputError("at most 4096 ports may be selected")


def _json_lines(output: bytes) -> tuple[list[dict], bool, int]:
    items: list[dict] = []
    size, invalid = 0, 0
    truncated = False
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
        except (ValueError, RecursionError):
            invalid += 1
            continue
        if len(items) >= MAX_RESULTS or size + len(line) > MAX_RESULTS_BYTES:
            truncated = True
            continue
        items.append(value)
        size += len(line)
    return items, truncated, invalid


def _sqlmap_techniques(output: bytes) -> list[str]:
    """Expose only sqlmap's fixed technique labels, never parameters or payloads."""
    allowed = {
        b"boolean-based blind",
        b"error-based",
        b"inline query",
        b"stacked queries",
        b"time-based blind",
        b"union query",
    }
    return sorted(
        {
            value.decode("ascii")
            for match in re.finditer(rb"(?m)^[ \t]*Type:[ \t]*([^\r\n]+)", output)
            if (value := match[1].strip().lower()) in allowed
        }
    )
