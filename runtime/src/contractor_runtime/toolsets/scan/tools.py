"""Typed adapters for independently available scanner and discovery executables."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime import threads
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy_bridge import HTTPProxyBridge
from contractor_runtime.adapters.http_transport import HTTPTransport, HTTPTransportError
from contractor_runtime.artifacts import ArtifactAPIError
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.process import (
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_command,
)
from contractor_runtime.core.target_policy import (
    TargetDenied,
    TargetPolicy,
    TargetPolicyConfig,
    TargetUnresolved,
    url_endpoint,
    validate_host_syntax,
)
from contractor_runtime.core.tool_metrics import ToolCallCancelled, ToolMetrics, elapsed_ms
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding
from contractor_runtime.toolsets.common.artifacts import ArtifactClientFactory, _unconfigured_client
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.resources import ToolResources
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
    TARGET_LIST_MEDIA_TYPE,
    canonical_url,
    katana_observation,
    scope_regex,
)
from contractor_runtime.toolsets.scan.nuclei_http import (
    SUPPORTED_ENGINE_VERSION,
    execution_package,
    selection_package,
    source_selection,
)
from contractor_runtime.toolsets.scan.nuclei_inputs import validate_nuclei_arguments
from contractor_runtime.toolsets.scan.nuclei_package import (
    MANIFEST_MEDIA_TYPE,
    MAX_MANIFEST_BYTES,
    PackageError,
    selected_manifest,
    snapshot_package,
)
from contractor_runtime.toolsets.scan.nuclei_rulepack import (
    MAX_FILE_BYTES,
    MAX_ZIP_BYTES,
    RULEPACK_MEDIA_TYPE,
    materialize_rulepack,
)
from contractor_runtime.toolsets.scan.process import ProcessResult, child_environment, run_process
from contractor_runtime.toolsets.scan.wordlist import MAX_WORDLIST_ARTIFACT_BYTES, parse_wordlist
from contractor_runtime.workspace import AllocationWorkspace

PROBE_TIMEOUT_SECONDS = 2.0
MAX_RESULTS = 100
MAX_RESULTS_BYTES = 128 * 1024
PrepareInvocation = Callable[[Path, float], Awaitable[list[str]]]
Destination = tuple[str, tuple[int, ...]]
FinalizeInvocation = Callable[[dict, float], Awaitable[dict]]


class ScanInputError(ToolInputError):
    """Runtime-authored validation diagnostics, without supplied argument values."""


class ScannerUnavailable(Exception):
    """An adapter-owned prerequisite is unavailable; the message is a fixed error code."""


class ScanTargetRefused(Exception):
    """The destination failed the target policy; the message is a fixed error code."""


class _Scratch:
    """A private directory created and removed in worker threads, off the event loop.

    Every thread is joined, also on cancellation (``to_thread_until_done``).
    Creation records the directory before its thread returns, so a cancelled
    entry still removes a directory that the thread created after the cancel.
    """

    def __init__(self, parent: Path | None, prefix: str) -> None:
        self._parent = parent
        self._prefix = prefix
        self._directory: tempfile.TemporaryDirectory[str] | None = None

    def _create(self) -> Path:
        self._directory = tempfile.TemporaryDirectory(prefix=self._prefix, dir=self._parent)
        return Path(self._directory.name)

    async def _remove(self) -> None:
        if self._directory is not None:
            await threads.to_thread_until_done(self._directory.cleanup, name="scan-scratch-remove")

    async def __aenter__(self) -> Path:
        try:
            return await threads.to_thread_until_done(self._create, name="scan-scratch-create")
        except BaseException:
            await self._remove()
            raise

    async def __aexit__(self, *_exception: object) -> None:
        await self._remove()


def _write_private(path: Path, data: bytes) -> None:
    """Create one scanner input exclusively, with a mode independent of the umask.

    Blocking file I/O: callers run it in a worker thread.
    """

    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)


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
        target_policy: TargetPolicy | None,
        controlled_http: HTTPTransport | None = None,
    ):
        self.workspace = workspace
        self.executables = executables
        self.templates = templates
        self.proxy_configured = proxy_configured
        self.artifact_client = artifact_client
        self.namespace = namespace
        self.target_policy = target_policy
        self.controlled_http = controlled_http
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
        finalize: FinalizeInvocation | None = None,
        credential_id: str | None = None,
    ) -> dict:
        async def invocation():
            # The scan lock serializes calls; waiting for it is outside this
            # invocation's preparation, process and publication budget.
            deadline = time.monotonic() + timeout
            if self._closed:
                result = ProcessResult(None, error_code="scan_closed")
            elif self.proxy_configured:
                # The generic env-based launcher does not guarantee scanner routing.
                # Never silently bypass the allocation's configured proxy.
                result = ProcessResult(None, error_code="scan_proxy_unsupported")
            elif not self.executables[tool.name]:
                result = ProcessResult(None, error_code="scanner_unavailable")
            else:
                result = await self._execute(
                    tool, arguments, deadline, destination, prepare, credential_id
                )
            # Output parsing is CPU work over up to 1 MiB; publication stays on the loop.
            value = await threads.to_thread_until_done(
                observation or tool.observation, result, name="scan-observation"
            )
            return await finalize(value, deadline) if finalize is not None else value

        async def owned_invocation():
            owned = self.controlled_http.own_invocation() if self.controlled_http else False
            try:
                return await invocation()
            finally:
                if self.controlled_http is not None:
                    self.controlled_http.finish_invocation(owned)

        async with self._lock:
            # Artifact publication is part of the invocation: keep it serialized
            # and cancellable until close has joined its cleanup as well. Worker
            # threads are joined before the invocation ends, so the lock is
            # held until parsing, publication and scratch removal finish.
            self._task = asyncio.create_task(owned_invocation())
            try:
                return await self._task
            finally:
                self._task = None

    async def _execute(
        self,
        tool: ScanTool,
        arguments: list[str],
        deadline: float,
        destination: Destination | None,
        prepare: PrepareInvocation | None,
        credential_id: str | None,
    ) -> ProcessResult:
        async with _Scratch(self.workspace, f"{tool.name}-") as directory:
            try:
                # Destination checks, artifact retrieval and materialization
                # share the scan deadline.
                async with asyncio.timeout_at(deadline):
                    prepared = await threads.to_thread_until_done(
                        tool.prepare, directory, self.templates, name="scan-prepare"
                    )
                    if destination is not None and self.controlled_http is None:
                        await self.require_destination(*destination)
                    if prepare is not None:
                        prepared += await prepare(directory, deadline)
            except TimeoutError:
                return ProcessResult(None, error_code="scan_timeout")
            except (ScannerUnavailable, ScanTargetRefused, PackageError) as error:
                return ProcessResult(None, error_code=str(error))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ProcessResult(None, error_code="scan_timeout")
            command = [self.executables[tool.name], *arguments, *prepared]
            if self.controlled_http is None:
                return await run_process(command, directory, remaining)
            try:
                async with asyncio.timeout_at(deadline):
                    async with HTTPProxyBridge(
                        self.controlled_http,
                        credential_id=credential_id,
                        deadline=deadline,
                    ) as bridge:
                        configure_proxy = tool.http_proxy_arguments
                        if configure_proxy is None:
                            raise RuntimeError("scanner HTTP proxy configuration is missing")
                        proxy_arguments = await configure_proxy(bridge, directory)
                        result = await run_process(
                            [*command, *proxy_arguments],
                            directory,
                            max(0.001, deadline - time.monotonic()),
                        )
                        result = await threads.to_thread_until_done(
                            lambda: replace(
                                result,
                                stdout=bridge.redact(result.stdout),
                                stderr=bridge.redact(result.stderr),
                            ),
                            name="scanner-private-output-redaction",
                        )
                    # Cleanup joins every bridge task. An early engine exit can
                    # otherwise race a still-pending exchange or its failure.
                    return replace(
                        result,
                        error_code=result.error_code
                        or bridge.error_code
                        or ("scan_request_failed" if bridge.requests == 0 else None),
                    )
            except TimeoutError:
                return ProcessResult(None, error_code="scan_timeout")
            except (HTTPTransportError, OSError, ValueError):
                return ProcessResult(None, error_code="scan_request_failed")

    async def require_destination(self, host: str, ports: tuple[int, ...]) -> None:
        """Check every address the host resolves to now, before the scanner starts.

        The scanner resolves the name again itself. This cannot pin its
        connections, so a name that changes answers after this check (DNS
        rebinding) is a residual risk; deployments needing a closed boundary
        restrict the Runtime's network namespace.
        """

        if self.target_policy is None:
            raise RuntimeError("standalone scanner destination policy is missing")
        try:
            await self.target_policy.require(host, ports)
        except TargetDenied:
            raise ScanTargetRefused("scan_target_denied") from None
        except TargetUnresolved:
            raise ScanTargetRefused("scan_target_unresolved") from None

    async def require_output_name_free(self, ref: ArtifactRef) -> None:
        """Admit a create-only output only while its binding does not exist.

        A metadata read (ART-R116) answers this without downloading an
        existing payload. Only 404 admits the output; any other failure is
        unavailable, never a guess that the name is free.
        """

        try:
            await self.artifact_client.read_artifact_metadata(ref)
        except ArtifactAPIError as error:
            if error.status_code != 404:
                raise ScannerUnavailable("scan_artifact_unavailable") from None
        except Exception:
            raise ScannerUnavailable("scan_artifact_unavailable") from None
        else:
            raise ScannerUnavailable("scan_output_exists")

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
    http_proxy_arguments: Callable[[HTTPProxyBridge, Path], Awaitable[list[str]]] | None = None

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
        finalize: FinalizeInvocation | None = None,
        credential_id: str | None = None,
    ) -> dict:
        started = time.perf_counter_ns()
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
                credential_id=credential_id,
            )
            return result
        except asyncio.CancelledError:
            error = ToolCallCancelled()
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
                duration_ms=elapsed_ms(started),
            )


class JSONLinesScanTool(ScanTool):
    def observation(self, result: ProcessResult) -> dict:
        response = super().observation(result)
        items, truncated, invalid = _json_lines(result.stdout)
        response.update(results=items, resultsTruncated=truncated, invalidResultLines=invalid)
        if invalid and response["errorCode"] is None:
            response.update(status="failed", errorCode="invalid_scanner_output")
        return response


@dataclass(slots=True)
class _NucleiPreparationResult:
    """Prepared target argv and the facts confirmed before success or failure.

    Source fields are set after argument admission, digests after materialization,
    engine version after its probe, and manifest fields only after publication.
    A failed later step retains earlier facts for the existing observation.
    """

    argv: list[str] = field(default_factory=list)
    source: str | None = None
    rulepack_artifact: dict[str, str] | None = None
    expected_package_digest: str | None = None
    actual_package_digest: str | None = None
    source_content_digest: str | None = None
    engine_version: str | None = None
    template_manifest: dict[str, str] | None = None
    selected_template_count: int | None = None

    def provenance(self) -> dict[str, Any]:
        values = {
            "source": self.source,
            "rulepackArtifact": self.rulepack_artifact,
            "expectedPackageDigest": self.expected_package_digest,
            "actualPackageDigest": self.actual_package_digest,
            "sourceContentDigest": self.source_content_digest,
            "engineVersion": self.engine_version,
            "templateManifest": self.template_manifest,
            "selectedTemplateCount": self.selected_template_count,
        }
        return {name: value for name, value in values.items() if value is not None}


@dataclass(slots=True)
class _NucleiPreparation:
    """One invocation's source admission, selection and manifest publication.

    Exact source verification precedes native selection; create-only manifest
    publication precedes target scanning. Any failure escapes to _ScanSession,
    which never launches a target scan after failed preparation. That session
    keeps the lock, shared deadline and scratch ownership; all blocking helpers
    keep their joined worker-thread boundaries. No state survives this call.
    """

    session: _ScanSession
    url: str
    template_ids: str
    tags: str
    severity: str
    rate_limit: int
    timeout_seconds: int
    templates_digest: str
    rulepack_ref: dict[str, str] | None
    credential_id: str | None = None
    result: _NucleiPreparationResult = field(default_factory=_NucleiPreparationResult, init=False)
    manifest_ref: ArtifactRef = field(init=False)

    def __post_init__(self) -> None:
        name = "template-manifest"
        if self.session.controlled_http is not None:
            name += "." + secrets.token_hex(12)
        self.manifest_ref = ArtifactRef(namespace=self.session.namespace, name=name)

    @property
    def publishes_manifest(self) -> bool:
        return bool(
            self.session.controlled_http or self.templates_digest or self.rulepack_ref is not None
        )

    def arguments(self) -> list[str]:
        values = dict(
            url=self.url,
            template_ids=self.template_ids,
            tags=self.tags,
            severity=self.severity,
            rate_limit=self.rate_limit,
            timeout_seconds=self.timeout_seconds,
            templates_digest=self.templates_digest,
        )
        if self.rulepack_ref is not None:
            values["rulepack_ref"] = self.rulepack_ref
        if self.credential_id is not None:
            values["credential_id"] = self.credential_id
            if self.session.controlled_http is None:
                raise ScanInputError("credential_id requires an engagement HTTP assignment")
        if self.session.controlled_http is not None:
            try:
                catalog = self.session.controlled_http.credentials()
                if self.credential_id is not None and not any(
                    item["credential_id"] == self.credential_id for item in catalog
                ):
                    raise ValueError
            except (ValueError, HTTPTransportError):
                raise ScanInputError("credential is not assigned to this allocation") from None
        try:
            validate_nuclei_arguments(values)
        except ToolInputError as error:
            raise ScanInputError(str(error)) from None
        if self.rulepack_ref is not None:
            self.result.source = "rulepack"
            self.result.rulepack_artifact = self.rulepack_ref
        elif self.templates_digest or self.session.controlled_http is not None:
            self.result.source = "installed"
            self.result.expected_package_digest = self.templates_digest or None
        command = [
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
            str(self.rate_limit if self.session.controlled_http is None else 0),
            "-concurrency",
            "5",
            "-timeout",
            "10",
            "-retries",
            "1" if self.session.controlled_http is None else "0",
        ]
        if self.session.controlled_http is not None:
            command += ["-no-httpx", "-payload-concurrency", "1"]
        for flag, value in (
            ("-id", self.template_ids),
            ("-tags", self.tags),
            ("-severity", self.severity),
        ):
            if value:
                command += [flag, value]
        return command

    async def prepare(self, directory: Path, deadline: float) -> _NucleiPreparationResult:
        templates = self.session.templates
        identifiers = None
        source = None
        if self.rulepack_ref is not None:
            exact_ref = ArtifactRef.model_validate(self.rulepack_ref).require_exact()
            try:
                metadata = await self.session.artifact_client.read_artifact_metadata(exact_ref)
                if metadata.media_type not in {"application/yaml", RULEPACK_MEDIA_TYPE}:
                    raise PackageError("nuclei_rulepack_invalid")
                maximum = (
                    MAX_FILE_BYTES if metadata.media_type == "application/yaml" else MAX_ZIP_BYTES
                )
                if metadata.size > maximum:
                    raise PackageError("nuclei_rulepack_invalid")
                artifact = await self.session.artifact_client.read_artifact(
                    exact_ref, max_bytes=maximum
                )
                if (
                    artifact.media_type != metadata.media_type
                    or len(artifact.data) != metadata.size
                ):
                    raise PackageError("nuclei_rulepack_invalid")
            except PackageError:
                raise
            except Exception:
                raise ScannerUnavailable("scan_artifact_unavailable") from None
            snapshot, identifiers, digest = await threads.to_thread_until_done(
                materialize_rulepack,
                artifact.data,
                artifact.media_type,
                directory / "nuclei-templates",
                deadline,
                name="nuclei-rulepack-materialize",
            )
            self.result.actual_package_digest = snapshot.digest
            self.result.source_content_digest = digest
            source = {
                "kind": "rulepack",
                "artifact": exact_ref.model_dump(by_alias=True),
                "contentDigest": digest,
            }
        else:
            if not await threads.to_thread_until_done(
                templates.is_dir, name="nuclei-templates-check"
            ):
                raise ScannerUnavailable("nuclei_templates_unavailable")
            if self.templates_digest or self.session.controlled_http is not None:
                snapshot = await threads.to_thread_until_done(
                    snapshot_package,
                    templates,
                    directory / "nuclei-templates",
                    deadline,
                    name="nuclei-package-snapshot",
                )
                self.result.actual_package_digest = snapshot.digest
                if self.templates_digest and snapshot.digest != self.templates_digest:
                    raise PackageError("nuclei_package_mismatch")
        if self.publishes_manifest:
            templates = snapshot.root
            await self.session.require_output_name_free(self.manifest_ref)
            executable = self.session.executables["scan_nuclei"]
            try:
                version = await run_command(
                    [executable, "-version"],
                    cwd=directory,
                    env=child_environment(directory),
                    timeout=min(2.0, max(0.001, deadline - time.monotonic())),
                    max_output_bytes=8192,
                )
                match = re.search(
                    rb"(?:Nuclei Engine Version|Current [Vv]ersion):\s*"
                    rb"(v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?)",
                    version.stdout + version.stderr,
                )
                if version.returncode or match is None:
                    raise PackageError("nuclei_selection_invalid")
                engine_version = match[1].decode("ascii")
                self.result.engine_version = engine_version
                if (
                    self.session.controlled_http is not None
                    and engine_version != SUPPORTED_ENGINE_VERSION
                ):
                    raise PackageError("nuclei_selection_invalid")
                if self.rulepack_ref is not None and self.session.controlled_http is None:
                    validation = await run_command(
                        [
                            executable,
                            "-validate",
                            "-t",
                            str(templates),
                            "-disable-update-check",
                            "-no-interactsh",
                            "-no-color",
                            "-silent",
                        ],
                        cwd=directory,
                        env=child_environment(directory),
                        timeout=max(0.001, deadline - time.monotonic()),
                        max_output_bytes=MAX_MANIFEST_BYTES,
                    )
                    if validation.returncode:
                        raise PackageError("nuclei_rulepack_invalid")
                selection = templates
                if self.session.controlled_http is not None:
                    selection = await threads.to_thread_until_done(
                        selection_package,
                        snapshot,
                        directory / "selection",
                        deadline,
                        name="nuclei-inert-selection",
                    )
                listing = await run_command(
                    [executable, *self.arguments(), "-t", str(selection), "-tl"],
                    cwd=directory,
                    env=child_environment(directory),
                    timeout=max(0.001, deadline - time.monotonic()),
                    max_output_bytes=MAX_MANIFEST_BYTES,
                )
                if listing.returncode:
                    raise PackageError("nuclei_selection_invalid")
                selected_output = listing.stdout
                selected_paths = []
                if self.session.controlled_http is not None:
                    selected_output, selected_paths = await threads.to_thread_until_done(
                        source_selection,
                        snapshot,
                        selection,
                        listing.stdout,
                        deadline,
                        name="nuclei-original-selection",
                    )
                data, count = await threads.to_thread_until_done(
                    selected_manifest,
                    snapshot,
                    selected_output,
                    engine_version,
                    deadline,
                    identifiers,
                    source,
                    name="nuclei-selected-manifest",
                )
                if self.session.controlled_http is not None:
                    templates = await threads.to_thread_until_done(
                        execution_package,
                        snapshot,
                        selected_paths,
                        directory / "http-templates",
                        deadline,
                        self.credential_id,
                        name="nuclei-http-admission",
                    )
                    validation = await run_command(
                        [
                            executable,
                            "-validate",
                            "-t",
                            str(templates),
                            "-disable-update-check",
                            "-no-interactsh",
                            "-no-color",
                            "-silent",
                        ],
                        cwd=directory,
                        env=child_environment(directory),
                        timeout=max(0.001, deadline - time.monotonic()),
                        max_output_bytes=MAX_MANIFEST_BYTES,
                    )
                    if validation.returncode:
                        raise PackageError("nuclei_selection_invalid")
            except ProcessTimeoutError:
                raise PackageError("scan_timeout") from None
            except (ProcessOutputLimitError, OSError):
                raise PackageError("nuclei_selection_invalid") from None
            try:
                written = await self.session.artifact_client.write_artifact(
                    self.manifest_ref,
                    data=data,
                    media_type=MANIFEST_MEDIA_TYPE,
                    expected_revision=None,
                )
                exact = written.artifact.require_exact()
                if (
                    exact.namespace != self.manifest_ref.namespace
                    or exact.name != self.manifest_ref.name
                ):
                    raise ValueError
            except Exception:
                raise ScannerUnavailable("scan_artifact_failed") from None
            self.result.template_manifest = exact.model_dump(by_alias=True)
            self.result.selected_template_count = count
        # A one-line file preserves commas in the supplied URL.
        target = directory / "target.txt"
        await threads.to_thread_until_done(
            _write_private,
            target,
            (self.url + "\n").encode("utf-8"),
            name="nuclei-target-materialize",
        )
        self.result.argv = ["-l", str(target), "-t", str(templates)]
        return self.result


class NucleiTool(JSONLinesScanTool):
    name = "scan_nuclei"
    binary = "nuclei"
    version_arguments = ("-version",)
    description = """Scan one HTTP(S) URL with installed or exact custom nuclei HTTP templates.

    Uses local templates, disables updates and external OAST callbacks. Calls are
    serialized within the allocation. A configured tool proxy route is unsupported.
    The supplied target passes target policy before launch. Custom pack-controlled
    destinations follow the trusted templates. With an engagement, normal HTTP
    templates use the allocation HTTP transport and its scope, TLS, identity jars
    and optional limits. Scanner retries are disabled and Runtime owns cookies;
    unsupported protocols, raw/unsafe HTTP and impure expressions fail before traffic.
    Findings are scanner evidence and are not automatically published.

    Args:
        url: Absolute HTTP(S) target URL without embedded credentials.
        template_ids: Comma-separated template IDs or wildcard IDs; empty selects all.
        tags: Comma-separated template tags; empty applies no tag filter.
        severity: Comma-separated info, low, medium, high, critical, unknown values.
        rate_limit: Standalone scanner rate, 1 through 1000; defaults to 10.
            Engagement scans use only the allocation's HTTP rate limit instead.
        timeout_seconds: Total scan deadline, 1 through 3600 seconds; defaults to 300.
        templates_digest: Optional sha256 package pin. Verify a private installed-package
            snapshot, list the selected rules and publish their exact template_manifest
            artifact before scanning. A mismatch fails without scanning.
        credential_id: Optional exact credential ID from the engagement assignment;
            omitted uses the anonymous session. Unavailable without an engagement.
        rulepack_ref: Optional exact RunScope artifact ref for application/yaml or
            application/vnd.contractor.nuclei-rulepack+zip. Custom mode validates
            every HTTP template and exclusively scans the pack, without installed
            templates. A bad supplied pack fails; it never selects installed rules.

    Returns:
        status, exitCode, errorCode, bounded stdout/stderr previews and truncation
        flags, durationMs, up to 100 JSON results, resultsTruncated and invalidResultLines.
        Timeout, overflow and execution errors are failures, not clean scans.
    """

    async def http_proxy_arguments(self, bridge: HTTPProxyBridge, directory: Path) -> list[str]:
        config = directory / "nuclei-proxy.yaml"
        await threads.to_thread_until_done(
            _write_private,
            config,
            ("proxy: [" + json.dumps(bridge.proxy_url) + "]\nproxy-internal: true\n").encode(),
            name="nuclei-private-proxy-config",
        )
        return ["-config", str(config)]

    def prepare(self, directory: Path, templates: Path) -> list[str]:
        # Prevent a first-run template install even on older nuclei versions.
        (directory / "nuclei-templates").mkdir()
        return []

    async def __call__(
        self,
        url: str,
        template_ids: str = "",
        tags: str = "",
        severity: str = "",
        rate_limit: int = 10,
        timeout_seconds: int = 300,
        templates_digest: str = "",
        rulepack_ref: dict[str, str] | None = None,
        credential_id: str | None = None,
    ) -> dict:
        preparation = _NucleiPreparation(
            session=self._session,
            url=url,
            template_ids=template_ids,
            tags=tags,
            severity=severity,
            rate_limit=rate_limit,
            timeout_seconds=timeout_seconds,
            templates_digest=templates_digest,
            rulepack_ref=rulepack_ref,
            credential_id=credential_id,
        )

        async def prepare_target(directory: Path, deadline: float) -> list[str]:
            prepared = await preparation.prepare(directory, deadline)
            return prepared.argv

        def observation(result: ProcessResult) -> dict:
            value = self.observation(result)
            if preparation.publishes_manifest:
                prepared = preparation.result
                value.update(
                    **prepared.provenance(),
                    filters={"templateIds": template_ids, "tags": tags, "severity": severity},
                )
                if self._session.controlled_http is None:
                    value["rateLimit"] = rate_limit
                else:
                    value["httpLimits"] = self._session.controlled_http.configured_limits()
            return value

        return await self._call(
            timeout_seconds,
            preparation.arguments,
            destination=lambda: _url_destination(url),
            prepare=prepare_target,
            observation=observation,
            credential_id=credential_id,
        )


class SQLMapTool(ScanTool):
    name = "scan_sqlmap"
    binary = "sqlmap"
    version_arguments = ("--version",)
    description = """Check one prepared HTTP request or URL for SQL injection with sqlmap.

    Each call uses a fresh session. No database dumping or shell operations are
    requested. A configured tool proxy route is unsupported. Review the output
    to distinguish detected injection, a negative check, and scanner diagnostics.
    Metadata and Runtime service destinations fail with scan_target_denied, as do
    loopback and link-local ones unless the operator allows the network.

    Args:
        url: HTTP(S) target URL; omit when using request_ref.
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

        async def prepare_request(directory: Path, _deadline: float) -> list[str]:
            if not (exact_ref is not None):
                raise RuntimeError("Expected exact_ref is not None")
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
            request = await threads.to_thread_until_done(
                parse_http_request, value.data, name="sqlmap-request-parse"
            )
            host, port = url_endpoint(request.url)
            await self._session.require_destination(host, (port,))
            path = directory / "request.http"
            try:
                await threads.to_thread_until_done(
                    _write_private, path, request.raw, name="sqlmap-request-materialize"
                )
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
            if not (exact_ref is not None):
                raise RuntimeError("Expected exact_ref is not None")
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
    Metadata and Runtime service destinations fail with scan_target_denied, as do
    loopback and link-local ones unless the operator allows the network.

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
    Metadata and Runtime service destinations fail with scan_target_denied, as do
    loopback and link-local ones unless the operator allows the network.

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

        async def prepare_wordlist(directory: Path, _deadline: float) -> list[str]:
            nonlocal entries
            if not (exact_ref is not None):
                raise RuntimeError("Expected exact_ref is not None")
            try:
                value = await self._session.artifact_client.read_artifact(
                    exact_ref, max_bytes=MAX_WORDLIST_ARTIFACT_BYTES
                )
            except Exception:
                raise ScanInputError("wordlist artifact is inaccessible or invalid") from None
            if value.media_type not in {"text/plain", "text/vnd.contractor.wordlist"}:
                raise ScanInputError("wordlist artifact must have a supported text media type")
            path = directory / "wordlist.txt"

            def materialize() -> int:
                # Validation finishes before the private list is created.
                wordlist = parse_wordlist(value.data)
                try:
                    _write_private(path, wordlist.raw)
                except OSError:
                    raise ScannerUnavailable("scan_wordlist_file_unavailable") from None
                return wordlist.line_count

            entries = await threads.to_thread_until_done(
                materialize, name="ffuf-wordlist-materialize"
            )
            return ["-w", f"{path}:FUZZ"]

        def observation(result: ProcessResult) -> dict:
            if not (exact_ref is not None):
                raise RuntimeError("Expected exact_ref is not None")
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
        # The page-budget/scope contract is verified against these release families.
        return (
            result.error_code is None
            and re.search(
                rb"\bCurrent version: v1\.(?:7|8)\.[0-9]+(?:\s|$)", result.stdout + result.stderr
            )
            is not None
        )

    description = """Discover bounded same-origin HTTP targets using installed Katana 1.7 or 1.8.

    Uses standard crawling with redirects and retries disabled, no browser, form
    filling, JavaScript crawling or external lookups. Publishes a create-only
    targets Artifact in the Worker's namespace and returns its exact reference.
    Only observed same-origin GET responses become targets; no scan is dispatched.
    A configured tool proxy route is unsupported.
    Metadata and Runtime service destinations fail with scan_target_denied, as do
    loopback and link-local ones unless the operator allows the network.

    Args:
        url: One HTTP(S) seed URL without credentials or a fragment.
        max_depth: Maximum crawl depth, 1 through 5; defaults to 2.
        max_pages: Katana per-domain page budget, 1 through 1000; defaults to 100.
        rate_limit: Maximum configured requests per second, 1 through 1000; default 10.
        timeout_seconds: Deadline after this allocation's scan lock is acquired,
            including artifact access, 1 through 3600; defaults to 60. Individual
            request and queue idle timeouts are at most 10 seconds and shorten
            with the total deadline.

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

        async def prepare(directory: Path, _deadline: float) -> list[str]:
            await self._session.require_output_name_free(target)
            # Katana's -u string-slice flag splits commas. A file preserves one
            # exact seed and prevents query text from introducing another target.
            seed_file, config = directory / "seed.txt", directory / "config.yaml"

            def materialize() -> None:
                _write_private(seed_file, (seed + "\n").encode())
                _write_private(config, b"{}\n")

            await threads.to_thread_until_done(materialize, name="katana-seed-materialize")
            return ["-list", str(seed_file), "-config", str(config)]

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

        async def finalize(value: dict, deadline: float) -> dict:
            if data:
                try:
                    remaining = deadline - time.monotonic()
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
            {
                name: frozenset(
                    {"runtime-subprocess-launcher"}
                    | (
                        {"runtime-http-client"}
                        if scanner.http_proxy_arguments is not None
                        else set()
                    )
                )
                for name, scanner in self._scanners.items()
            }
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
                async with _Scratch(None, "contractor-scan-probe-") as root:
                    result = await run_process(
                        [executable, *self._scanners[name].version_arguments],
                        root,
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
        resources: ToolResources,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
        controlled_http: HTTPTransport | None = None,
    ) -> Mapping[str, Any]:
        del run_id, project_workspace
        require_selected_tools(
            selected, self.exported_tools, description="unknown selected scan tools"
        )
        if not selected:
            return {}
        if controlled_http is not None and any(
            self._scanners[name].http_proxy_arguments is None for name in selected
        ):
            raise ValueError("selected scanner cannot use the allocation HTTP transport")
        metrics = require_metrics(resources, "scan@1")
        session = _ScanSession(
            workspace.path,
            self._executables,
            self._templates,
            proxy_configured=(
                adapter_handles.tool_subprocess is not None or _proxy_routes_tools(runtime_settings)
            ),
            artifact_client=self._clients(allocation_id, runtime_settings),
            namespace=namespace,
            target_policy=(await self._target_policy.build(runtime_settings))
            if controlled_http is None
            else None,
            controlled_http=controlled_http,
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
    try:
        validate_host_syntax(value)
    except ValueError:
        raise ScanInputError("host must be one DNS hostname or IP address") from None


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
