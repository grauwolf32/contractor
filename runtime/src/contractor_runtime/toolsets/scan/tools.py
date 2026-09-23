"""Typed adapters for independently available scanner and discovery executables."""

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
from contractor_runtime.artifacts import ArtifactAPIError
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding
from contractor_runtime.toolsets.common.artifacts import ArtifactClientFactory, _unconfigured_client
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.common.metrics import ToolMetrics
from contractor_runtime.toolsets.common.target_policy import (
    TargetDenied,
    TargetPolicy,
    TargetPolicyConfig,
    TargetUnresolved,
    url_endpoint,
)
from contractor_runtime.toolsets.scan.ffuf import (
    ffuf_observation,
    validate_ffuf_filter,
    validate_ffuf_url,
)
from contractor_runtime.toolsets.scan.http_request import (
    MAX_REQUEST_ARTIFACT_BYTES,
    parse_http_request,
)
from contractor_runtime.toolsets.scan.katana import (
    MAX_RESPONSE_BYTES,
    MAX_TARGET_BYTES,
    TARGET_LIST_MEDIA_TYPE,
    canonical_url,
    katana_observation,
    scope_regex,
)
from contractor_runtime.toolsets.scan.process import ProcessResult, run_process
from contractor_runtime.toolsets.scan.wordlist import MAX_WORDLIST_ARTIFACT_BYTES, parse_wordlist
from contractor_runtime.workspace import AllocationWorkspace

PROBE_TIMEOUT_SECONDS = 2.0
MAX_RESULTS = 100
MAX_RESULTS_BYTES = 128 * 1024
PrepareInvocation = Callable[[Path], Awaitable[list[str]]]
Destination = tuple[str, tuple[int, ...]]


class ScanInputError(ToolInputError):
    """Runtime-authored validation diagnostics, without supplied argument values."""


class ScannerUnavailable(Exception):
    """An adapter-owned prerequisite is unavailable; the message is a fixed error code."""


class ScanTargetRefused(Exception):
    """The destination failed the target policy; the message is a fixed error code."""


class _ScanSession:
    def __init__(
        self,
        workspace,
        executables,
        templates,
        *,
        proxy_configured,
        artifact_client,
        namespace,
        target_policy: TargetPolicy,
    ):
        self.workspace = workspace
        self.executables = executables
        self.templates = templates
        self.proxy_configured = proxy_configured
        self.artifact_client = artifact_client
        self.namespace = namespace
        self.target_policy = target_policy
        self._closed = False
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def execute(
        self,
        tool: ScanTool,
        arguments: list[str],
        timeout: int,
        *,
        destination: Destination | None = None,
        prepare: PrepareInvocation | None = None,
        observation: Callable[[ProcessResult], dict] | None = None,
        finalize: Callable[[dict], Awaitable[dict]] | None = None,
    ) -> dict:
        async def invocation():
            if self._closed:
                result = ProcessResult(None, error_code="scan_closed")
            elif self.proxy_configured:
                # The generic env-based launcher does not guarantee scanner routing.
                # Never silently bypass the allocation's configured proxy.
                result = ProcessResult(None, error_code="scan_proxy_unsupported")
            elif not self.executables[tool.name]:
                result = ProcessResult(None, error_code="scanner_unavailable")
            else:
                result = await self._execute(tool, arguments, timeout, destination, prepare)
            value = (observation or tool.observation)(result)
            return await finalize(value) if finalize is not None else value

        async with self._lock:
            # Artifact publication is part of the invocation: keep it serialized
            # and cancellable until close has joined its cleanup as well.
            self._task = asyncio.create_task(invocation())
            try:
                return await self._task
            finally:
                self._task = None

    async def _execute(
        self,
        tool: ScanTool,
        arguments: list[str],
        timeout: int,
        destination: Destination | None,
        prepare: PrepareInvocation | None,
    ) -> ProcessResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"{tool.name}-", dir=self.workspace) as root:
            directory = Path(root)
            try:
                prepared = tool.prepare(directory, self.templates)
                # Destination checks, artifact retrieval and materialization
                # share the scan deadline.
                async with asyncio.timeout(timeout):
                    if destination is not None:
                        await self.require_destination(*destination)
                    if prepare is not None:
                        prepared += await prepare(directory)
            except TimeoutError:
                return ProcessResult(None, error_code="scan_timeout")
            except (ScannerUnavailable, ScanTargetRefused) as error:
                return ProcessResult(None, error_code=str(error))
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                return ProcessResult(None, error_code="scan_timeout")
            command = [self.executables[tool.name], *arguments, *prepared]
            return await run_process(command, directory, remaining)

    async def require_destination(self, host: str, ports: tuple[int, ...]) -> None:
        """Check every address the host resolves to now, before the scanner starts.

        The scanner resolves the name again itself. This cannot pin its
        connections, so a name that changes answers after this check (DNS
        rebinding) is a residual risk; deployments needing a closed boundary
        restrict the Runtime's network namespace.
        """

        try:
            await self.target_policy.require(host, ports)
        except TargetDenied:
            raise ScanTargetRefused("scan_target_denied") from None
        except TargetUnresolved:
            raise ScanTargetRefused("scan_target_unresolved") from None

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

    @staticmethod
    def accepts_probe(result: ProcessResult) -> bool:
        return result.error_code is None

    async def _call(
        self,
        timeout: int,
        arguments: Callable[[], list[str]],
        *,
        destination: Callable[[], Destination] | None = None,
        prepare: PrepareInvocation | None = None,
        observation: Callable[[ProcessResult], dict] | None = None,
        finalize: Callable[[dict], Awaitable[dict]] | None = None,
    ) -> dict:
        started = time.monotonic()
        result = None
        error = None
        try:
            _integer(timeout, "timeout_seconds", 1, 3600)
            command = arguments()
            result = await self._session.execute(
                self,
                command,
                timeout,
                # Evaluated only after arguments() validated the target text.
                destination=destination() if destination is not None else None,
                prepare=prepare,
                observation=observation,
                finalize=finalize,
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
    serialized within the allocation. A configured tool proxy route is unsupported.
    Loopback, private, metadata and Runtime service destinations fail with
    scan_target_denied unless the operator allows the network.
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

        return await self._call(
            timeout_seconds, arguments, destination=lambda: _url_destination(url)
        )


class SQLMapTool(ScanTool):
    name = "scan_sqlmap"
    binary = "sqlmap"
    version_arguments = ("--version",)
    description = """Check one prepared HTTP request or URL for SQL injection with sqlmap.

    Each call uses a fresh session. No database dumping or shell operations are
    requested. A configured tool proxy route is unsupported. Review the output
    to distinguish detected injection, a negative check, and scanner diagnostics.
    Loopback, private, metadata and Runtime service destinations fail with
    scan_target_denied unless the operator allows the network.

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
            host, port = url_endpoint(request.url)
            await self._session.require_destination(host, (port,))
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
            # Request mode checks the artifact's URL once it has been read.
            destination=(lambda: _url_destination(url)) if request_ref is None else None,
            prepare=prepare_request if request_ref is not None else None,
            observation=request_observation if request_ref is not None else None,
        )


class NaabuTool(JSONLinesScanTool):
    name = "scan_naabu"
    binary = "naabu"
    version_arguments = ("-version",)
    description = """Discover open TCP ports on one host with naabu CONNECT scanning.

    Does not require raw-socket privileges. A configured tool proxy route is
    unsupported. Calls are serialized within the allocation.
    Loopback, private, metadata and Runtime service destinations fail with
    scan_target_denied unless the operator allows the network.

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

        return await self._call(
            timeout_seconds, arguments, destination=lambda: (host, _port_values(ports))
        )


class FFUFTool(ScanTool):
    name = "scan_ffuf"
    binary = "ffuf"
    version_arguments = ("-V",)
    description = """Fuzz one HTTP(S) URL using an exact uploaded wordlist artifact.

    Replace FUZZ in the path or query with each UTF-8 payload, preserving duplicates,
    spaces and empty entries. Uses GET and one thread, without following redirects,
    recursion, auto-calibration or external input commands. Configured proxy unsupported.
    Loopback, private, metadata and Runtime service destinations fail with
    scan_target_denied unless the operator allows the network.

    Args:
        url: HTTP(S) URL containing FUZZ in its path or query, never the authority.
        wordlist_ref: Exact artifact ref (namespace, name, revision) for text/plain or
            text/vnd.contractor.wordlist; max 1 MiB, 10000 entries, 4096 bytes per entry.
        rate: Configured payloads per second, 1 through 1000; defaults to 10.
            ffuf may retry a failed HTTP request once outside this rate limiter.
        match_status: HTTP status numbers/ranges to match, or all (default).
        filter_status: Optional comma-separated HTTP status numbers/ranges to exclude.
        filter_size: Optional comma-separated response byte sizes/ranges to exclude.
        filter_words: Optional comma-separated response word counts/ranges to exclude.
        filter_lines: Optional comma-separated response line counts/ranges to exclude.
        timeout_seconds: Total deadline including artifact retrieval, 1 through 3600
            seconds; defaults to 300. Each HTTP request has a 10-second timeout.

    Returns:
        Process status, exact wordlistArtifact, wordlistEntries, bounded decoded results,
        truncation flags, payloadsAttempted, requestErrors and scanComplete. Missing
        progress or request errors fail even if ffuf exits zero. Empty matches do not
        establish a clean target. Raw diagnostics are suppressed; tool@1 stores a report.
    """

    async def __call__(
        self,
        url: str,
        wordlist_ref: dict[str, str],
        rate: int = 10,
        match_status: str = "all",
        filter_status: str = "",
        filter_size: str = "",
        filter_words: str = "",
        filter_lines: str = "",
        timeout_seconds: int = 300,
    ) -> dict:
        exact_ref: ArtifactRef | None = None
        entries = 0

        def arguments():
            nonlocal exact_ref
            validate_ffuf_url(url)
            _url(url)
            _integer(rate, "rate", 1, 1000)
            try:
                exact_ref = ArtifactRef.model_validate(wordlist_ref).require_exact()
                require_model_visible_binding(exact_ref.namespace, exact_ref.name)
            except (ValueError, TypeError):
                raise ScanInputError(
                    "wordlist_ref must be an accessible exact artifact ref"
                ) from None
            validate_ffuf_filter(match_status, "match_status", status=True, all_=True)
            command = [
                "-u",
                url,
                "-json",
                "-noninteractive",
                "-t",
                "1",
                "-rate",
                str(rate),
                "-timeout",
                "10",
                "-mc",
                match_status,
            ]
            for name, flag, value in (
                ("filter_status", "-fc", filter_status),
                ("filter_size", "-fs", filter_size),
                ("filter_words", "-fw", filter_words),
                ("filter_lines", "-fl", filter_lines),
            ):
                validate_ffuf_filter(value, name, status=name == "filter_status")
                if value:
                    command.extend((flag, value))
            return command

        async def prepare_wordlist(directory: Path) -> list[str]:
            nonlocal entries
            assert exact_ref is not None
            try:
                value = await self._session.artifact_client.read_artifact(
                    exact_ref, max_bytes=MAX_WORDLIST_ARTIFACT_BYTES
                )
            except Exception:
                raise ScanInputError("wordlist artifact is inaccessible or invalid") from None
            if value.media_type not in {"text/plain", "text/vnd.contractor.wordlist"}:
                raise ScanInputError("wordlist artifact must have a supported text media type")
            wordlist = parse_wordlist(value.data)
            entries = wordlist.line_count
            path = directory / "wordlist.txt"
            try:
                with path.open("xb") as output:
                    os.fchmod(output.fileno(), 0o600)
                    output.write(wordlist.raw)
            except OSError:
                raise ScannerUnavailable("scan_wordlist_file_unavailable") from None
            return ["-w", f"{path}:FUZZ"]

        def observation(result: ProcessResult) -> dict:
            assert exact_ref is not None
            return {
                **ffuf_observation(result, entries),
                "wordlistArtifact": exact_ref.model_dump(by_alias=True),
            }

        return await self._call(
            timeout_seconds,
            arguments,
            destination=lambda: _url_destination(url),
            prepare=prepare_wordlist,
            observation=observation,
        )


class KatanaTool(ScanTool):
    name = "scan_katana"
    binary = "katana"
    version_arguments = ("-version",)

    @staticmethod
    def accepts_probe(result: ProcessResult) -> bool:
        # The page-budget/scope contract is verified against the 1.7 series.
        return (
            result.error_code is None
            and re.search(
                rb"\bCurrent version: v1\.7\.[0-9]+(?:\s|$)", result.stdout + result.stderr
            )
            is not None
        )

    description = """Discover bounded same-origin HTTP targets using installed Katana 1.7.

    Uses standard crawling with redirects and retries disabled, no browser, form
    filling, JavaScript crawling or external lookups. Publishes a create-only
    targets Artifact in the Worker's namespace and returns its exact reference.
    Only observed same-origin GET responses become targets; no scan is dispatched.
    A configured tool proxy route is unsupported.
    Loopback, private, metadata and Runtime service destinations fail with
    scan_target_denied unless the operator allows the network.

    Args:
        url: One HTTP(S) seed URL without credentials or a fragment.
        max_depth: Maximum crawl depth, 1 through 5; defaults to 2.
        max_pages: Katana per-domain page budget, 1 through 1000; defaults to 100.
        rate_limit: Maximum configured requests per second, 1 through 1000; default 10.
        timeout_seconds: Total deadline including artifact access, 1 through 3600;
            defaults to 60. Individual request and queue idle timeouts are at
            most 10 seconds and shorten with the total deadline.

    Returns:
        Process status, exact targetsArtifact, targetsDigest, source, limits and
        bounded per-target provenance. discoveryComplete is always false: bounded
        discovery cannot certify exhaustion. Coverage records skipped/failed and
        truncated observations; raw headers, bodies and diagnostics are suppressed.
    """

    async def __call__(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 100,
        rate_limit: int = 10,
        timeout_seconds: int = 60,
    ) -> dict:
        seed = ""
        data = b""
        started = time.monotonic()
        target = ArtifactRef(namespace=self._session.namespace, name="targets")

        def arguments():
            nonlocal seed
            seed = canonical_url(url)
            _integer(max_depth, "max_depth", 1, 5)
            _integer(max_pages, "max_pages", 1, 1000)
            _integer(rate_limit, "rate_limit", 1, 1000)
            return [
                "-j",
                "-or",
                "-ob",
                "-eof",
                "headers",
                "-silent",
                "-nc",
                "-duc",
                "-dr",
                "-retry",
                "0",
                "-c",
                "1",
                "-p",
                "1",
                "-duf",
                "-fs",
                "fqdn",
                "-cs",
                scope_regex(seed),
                "-d",
                str(max_depth),
                "-mdp",
                str(max_pages),
                "-rl",
                str(rate_limit),
                "-ct",
                f"{max(1, timeout_seconds - 2)}s",
                "-timeout",
                str(min(10, max(1, timeout_seconds // 3))),
                "-mrs",
                str(MAX_RESPONSE_BYTES),
            ]

        async def prepare(directory: Path) -> list[str]:
            try:
                await self._session.artifact_client.read_artifact(
                    target, max_bytes=MAX_TARGET_BYTES
                )
            except ArtifactAPIError as error:
                if error.status_code != 404:
                    raise ScannerUnavailable("scan_artifact_unavailable") from None
            except Exception:
                raise ScannerUnavailable("scan_artifact_unavailable") from None
            else:
                raise ScannerUnavailable("scan_output_exists")
            # Katana's -u string-slice flag splits commas. A file preserves one
            # exact seed and prevents query text from introducing another target.
            for name, content in (("seed.txt", (seed + "\n").encode()), ("config.yaml", b"{}\n")):
                with (directory / name).open("xb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(content)
            return ["-list", str(directory / "seed.txt"), "-config", str(directory / "config.yaml")]

        def observation(result: ProcessResult) -> dict:
            nonlocal data
            value, data = katana_observation(
                result,
                seed=seed,
                max_depth=max_depth,
                max_pages=max_pages,
                rate_limit=rate_limit,
                timeout_seconds=timeout_seconds,
            )
            return value

        async def finalize(value: dict) -> dict:
            if data:
                try:
                    remaining = timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError
                    async with asyncio.timeout(remaining):
                        written = await self._session.artifact_client.write_artifact(
                            target,
                            data=data,
                            media_type=TARGET_LIST_MEDIA_TYPE,
                            expected_revision=None,
                        )
                    exact = written.artifact.require_exact()
                    if exact.namespace != target.namespace or exact.name != target.name:
                        raise ValueError
                    value["targetsArtifact"] = exact.model_dump(by_alias=True)
                    value["artifacts"] = {"targets": value["targetsArtifact"]}
                except Exception:
                    value.update(
                        status="failed",
                        errorCode=value["errorCode"] or "scan_artifact_failed",
                        artifactErrorCode="scan_artifact_failed",
                    )
            return value

        return await self._call(
            timeout_seconds,
            arguments,
            destination=lambda: _url_destination(seed),
            prepare=prepare,
            observation=observation,
            finalize=finalize,
        )


SCANNERS = (NucleiTool, SQLMapTool, NaabuTool, FFUFTool, KatanaTool)


class ScanToolsetFactory:
    ref = "scan@1"

    def __init__(
        self,
        artifact_client_factory: ArtifactClientFactory | None = None,
        *,
        templates_directory: Path | None = None,
        scanners: Sequence[type[ScanTool]] = SCANNERS,
        target_policy: TargetPolicyConfig | None = None,
    ) -> None:
        self._clients = artifact_client_factory or _unconfigured_client
        self._target_policy = target_policy or TargetPolicyConfig()
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
                return self._scanners[name].accepts_probe(result)
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
        del run_id, project_workspace
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
            proxy_configured=(
                adapter_handles.tool_subprocess is not None or _proxy_routes_tools(runtime_settings)
            ),
            artifact_client=self._clients(allocation_id, runtime_settings),
            namespace=namespace,
            target_policy=await self._target_policy.build(runtime_settings),
        )
        return {name: self._scanners[name](session, metrics) for name in selected}


def _proxy_routes_tools(settings: RuntimeSettings) -> bool:
    # A tool-http route is the deployment's egress boundary for model-selected
    # targets. Scanners cannot use it, so either tool route fails closed.
    proxy = settings.http_proxy
    return proxy is not None and bool({"tool-http", "tool-subprocess"} & set(proxy.targets))


def _url_destination(value: str) -> Destination:
    host, port = url_endpoint(value)
    return host, (port,)


def _port_values(value: str) -> tuple[int, ...]:
    result: set[int] = set()
    for item in value.split(","):
        first, _, last = item.partition("-")
        result.update(range(int(first), int(last or first) + 1))
    return tuple(sorted(result))


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
