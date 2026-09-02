"""Allocation-local owner for one bounded Trailmark graph child."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import signal
import struct
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from contractor_runtime.projectfs.paths import ProjectPathError, normalize_project_path
from contractor_runtime.projectfs.storage import WorkspaceSnapshot, WorkspaceTextFile
from contractor_runtime.toolsets import code_analysis_languages as language_support
from contractor_runtime.toolsets.code_analysis_ids import (
    decode_symbol_id,
    encode_symbol_key,
)
from contractor_runtime.toolsets.trailmark_child import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SCHEMA_VERSION,
)

MAX_GRAPH_FILES = 20_000
MAX_GRAPH_BYTES = 128 * 1024 * 1024
MAX_GRAPH_FILE_BYTES = 4 * 1024 * 1024
DEFAULT_BUILD_TIMEOUT_SECONDS = 120.0
DEFAULT_QUERY_TIMEOUT_SECONDS = 10.0
DEFAULT_STOP_TIMEOUT_SECONDS = 2.0
MIRROR_PREFIX = "code-analysis-mirror-"

_ENTRYPOINT_BOUNDARY = ".trailmark/entrypoints.toml"
_SAFE_CHILD_ENVIRONMENT = {
    "LC_ALL": "C.UTF-8",
    "NO_COLOR": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


class TrailmarkHostError(RuntimeError):
    """Stable child-host failure with no source or physical-path detail."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"Code analysis child failed ({code})")


class _ChildResponseError(TrailmarkHostError):
    pass


@dataclass(frozen=True, slots=True)
class GraphCoverage:
    analyzed_files: int
    analyzed_bytes: int
    binary_files: int
    unsupported_source_files: int
    oversized_files: int
    parse_errors: int
    reasons: tuple[str, ...]

    @property
    def incomplete(self) -> bool:
        return bool(self.reasons)

    def wire(self) -> dict[str, Any]:
        return {
            "analyzedFiles": self.analyzed_files,
            "analyzedBytes": self.analyzed_bytes,
            "binaryFiles": self.binary_files,
            "unsupportedSourceFiles": self.unsupported_source_files,
            "oversizedFiles": self.oversized_files,
            "parseErrors": self.parse_errors,
            "incomplete": self.incomplete,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class GraphBuildResult:
    snapshot_digest: str
    coverage: GraphCoverage
    languages: tuple[str, ...]
    node_count: int
    edge_count: int
    call_edge_count: int
    entrypoint_count: int
    dependency_count: int
    rss_kib: int


@dataclass(frozen=True, slots=True)
class GraphSymbolProjection:
    symbol_id: str
    name: str
    kind: str
    path: str
    line: int
    end_line: int
    column: int


@dataclass(frozen=True, slots=True)
class GraphSymbolPage:
    snapshot_digest: str
    items: tuple[GraphSymbolProjection, ...]
    observed_total: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class GraphRelationshipProjection:
    symbol: GraphSymbolProjection
    confidence: str


@dataclass(frozen=True, slots=True)
class GraphRelationshipPage:
    snapshot_digest: str
    items: tuple[GraphRelationshipProjection, ...]
    observed_total: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class _PreparedMirror:
    path: Path
    snapshot_digest: str
    coverage: GraphCoverage


class TrailmarkChildHost:
    """Serialize access to one graph child and its source-bearing mirror."""

    def __init__(
        self,
        scratch_root: Path,
        *,
        child_command: Sequence[str] | None = None,
        build_timeout_seconds: float = DEFAULT_BUILD_TIMEOUT_SECONDS,
        query_timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
        stop_timeout_seconds: float = DEFAULT_STOP_TIMEOUT_SECONDS,
    ) -> None:
        if min(build_timeout_seconds, query_timeout_seconds, stop_timeout_seconds) <= 0:
            raise ValueError("Trailmark child timeouts must be positive")
        root = scratch_root.expanduser().resolve()
        if root == Path(root.anchor):
            raise ValueError("Trailmark scratch root must not be a filesystem root")
        self._scratch_root = root
        self._child_command = (
            tuple(child_command) if child_command is not None else _child_command()
        )
        if not self._child_command or any(
            not isinstance(value, str) or not value for value in self._child_command
        ):
            raise ValueError("Trailmark child command must be non-empty strings")
        self._build_timeout_seconds = build_timeout_seconds
        self._query_timeout_seconds = query_timeout_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._mirror: _PreparedMirror | None = None
        self._request_number = 0
        self._symbol_key = bytearray(secrets.token_bytes(32))
        self._closed = False
        self._closing = False
        self._stderr_observed = False

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.returncode is None else None

    @property
    def mirror_exists(self) -> bool:
        return self._mirror is not None and self._mirror.path.exists()

    @property
    def stderr_observed(self) -> bool:
        return self._stderr_observed

    async def build(self, snapshot: WorkspaceSnapshot) -> GraphBuildResult:
        async with self._lock:
            self._require_open()
            if self._process is not None or self._mirror is not None:
                await self._stop_locked(remove_mirror=True)
            deadline = asyncio.get_running_loop().time() + self._build_timeout_seconds
            try:
                mirror = await _materialize_snapshot_cancellation_safe(snapshot, self._scratch_root)
                self._mirror = mirror
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TrailmarkHostError("code_analysis_build_timeout", retryable=True)
                await self._start_locked(mirror.path)
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TrailmarkHostError("code_analysis_build_timeout", retryable=True)
                result = await self._request_locked(
                    "build",
                    {
                        "snapshotDigest": mirror.snapshot_digest,
                        "coverage": mirror.coverage.wire(),
                        "symbolKey": encode_symbol_key(bytes(self._symbol_key)),
                    },
                    timeout=remaining,
                    timeout_code="code_analysis_build_timeout",
                )
                return _build_result(result, mirror)
            except asyncio.CancelledError:
                await self._cleanup_after_failure_locked()
                raise
            except TrailmarkHostError:
                await self._cleanup_after_failure_locked()
                raise
            except Exception:
                await self._cleanup_after_failure_locked()
                raise TrailmarkHostError("code_analysis_engine_failed", retryable=True) from None

    async def summary(self) -> GraphBuildResult:
        async with self._lock:
            self._require_running()
            assert self._mirror is not None
            result = await self._request_locked(
                "summary",
                {},
                timeout=self._query_timeout_seconds,
                timeout_code="code_analysis_query_timeout",
            )
            try:
                return _build_result(result, self._mirror)
            except TrailmarkHostError:
                await self._stop_locked(remove_mirror=True)
                raise

    async def symbols(self, *, limit: int = 200) -> GraphSymbolPage:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise TrailmarkHostError("code_analysis_input_invalid")
        async with self._lock:
            self._require_running()
            result = await self._request_locked(
                "symbols",
                {"limit": limit},
                timeout=self._query_timeout_seconds,
                timeout_code="code_analysis_query_timeout",
            )
            try:
                assert self._mirror is not None
                return _symbol_page(
                    result,
                    self._mirror.snapshot_digest,
                    bytes(self._symbol_key),
                    expected_offset=0,
                    expected_limit=limit,
                )
            except TrailmarkHostError:
                await self._stop_locked(remove_mirror=True)
                raise

    async def find_symbols(self, query: str, *, offset: int, limit: int) -> GraphSymbolPage:
        async with self._lock:
            self._require_running()
            assert self._mirror is not None
            result = await self._request_locked(
                "find_symbol",
                {"query": query, "offset": offset, "limit": limit},
                timeout=self._query_timeout_seconds,
                timeout_code="code_analysis_query_timeout",
            )
            try:
                return _symbol_page(
                    result,
                    self._mirror.snapshot_digest,
                    bytes(self._symbol_key),
                    expected_offset=offset,
                    expected_limit=limit,
                )
            except TrailmarkHostError:
                await self._stop_locked(remove_mirror=True)
                raise

    async def relationships(
        self,
        operation: str,
        symbol_id: str,
        *,
        offset: int,
        limit: int,
    ) -> GraphRelationshipPage:
        if operation not in {"find_callers", "find_callees"}:
            raise TrailmarkHostError("code_analysis_input_invalid")
        async with self._lock:
            self._require_running()
            assert self._mirror is not None
            self.validate_symbol_id(symbol_id, self._mirror.snapshot_digest)
            result = await self._request_locked(
                operation,
                {"symbolId": symbol_id, "offset": offset, "limit": limit},
                timeout=self._query_timeout_seconds,
                timeout_code="code_analysis_query_timeout",
            )
            try:
                return _relationship_page(
                    result,
                    self._mirror.snapshot_digest,
                    bytes(self._symbol_key),
                    expected_offset=offset,
                    expected_limit=limit,
                )
            except TrailmarkHostError:
                await self._stop_locked(remove_mirror=True)
                raise

    def validate_symbol_id(self, value: str, snapshot_digest: str) -> None:
        try:
            decoded = decode_symbol_id(bytes(self._symbol_key), value)
        except ValueError:
            raise TrailmarkHostError("code_analysis_symbol_not_found") from None
        if decoded.snapshot_digest != snapshot_digest:
            raise TrailmarkHostError("code_analysis_stale_symbol")

    async def invalidate(self) -> None:
        async with self._lock:
            self._require_open()
            await self._stop_locked(remove_mirror=True)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closing = True
            try:
                await self._stop_locked(remove_mirror=True)
            except BaseException:
                # A caller must keep the allocation fenced when process reaping
                # or source-bearing cleanup cannot be confirmed.
                raise
            else:
                self._symbol_key[:] = b"\x00" * len(self._symbol_key)
                self._closed = True

    def _require_open(self) -> None:
        if self._closed or self._closing:
            raise TrailmarkHostError("code_analysis_closing", retryable=True)

    def _require_running(self) -> None:
        self._require_open()
        if self._process is None or self._mirror is None or self._process.returncode is not None:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)

    async def _start_locked(self, mirror: Path) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *self._child_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=mirror,
                env=dict(_SAFE_CHILD_ENVIRONMENT),
                start_new_session=True,
            )
        except Exception:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True) from None
        self._process = process
        assert process.stderr is not None
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(process.stderr),
            name="trailmark-child-stderr",
        )

    async def _request_locked(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        timeout: float,
        timeout_code: str,
    ) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        self._request_number += 1
        request_id = f"r{self._request_number:x}-{uuid.uuid4().hex}"
        request = {
            "schemaVersion": SCHEMA_VERSION,
            "requestId": request_id,
            "operation": operation,
            "arguments": dict(arguments),
        }
        payload = _encode_request(request)
        try:
            async with asyncio.timeout(timeout):
                process.stdin.write(struct.pack(">I", len(payload)) + payload)
                await process.stdin.drain()
                response = await _read_response(process.stdout)
        except TimeoutError:
            await self._stop_locked(remove_mirror=True)
            raise TrailmarkHostError(timeout_code, retryable=True) from None
        except asyncio.CancelledError:
            await self._stop_locked(remove_mirror=True)
            raise
        except TrailmarkHostError:
            await self._stop_locked(remove_mirror=True)
            raise
        except (BrokenPipeError, ConnectionError, asyncio.IncompleteReadError):
            await self._stop_locked(remove_mirror=True)
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True) from None

        try:
            result = _validate_response(response, request_id)
        except _ChildResponseError:
            raise
        except TrailmarkHostError:
            await self._stop_locked(remove_mirror=True)
            raise
        if result is None:
            await self._stop_locked(remove_mirror=True)
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        return result

    async def _stop_locked(self, *, remove_mirror: bool) -> None:
        process = self._process
        if process is not None:
            reaped = await self._terminate_process(process)
            if not reaped:
                raise TrailmarkHostError("code_analysis_engine_failed")
            self._process = None

        stderr_task = self._stderr_task
        if stderr_task is not None:
            if not stderr_task.done():
                stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
            except Exception:
                self._stderr_observed = True
            self._stderr_task = None

        if remove_mirror and self._mirror is not None:
            mirror = self._mirror
            try:
                await asyncio.to_thread(_remove_mirror, mirror.path, self._scratch_root)
            except Exception:
                raise TrailmarkHostError("code_analysis_engine_failed") from None
            self._mirror = None

    async def _cleanup_after_failure_locked(self) -> None:
        if self._process is None and self._mirror is None:
            return
        try:
            await self._stop_locked(remove_mirror=True)
        except TrailmarkHostError:
            raise

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> bool:
        if process.returncode is not None:
            await process.wait()
            return True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        try:
            await asyncio.wait_for(process.wait(), timeout=self._stop_timeout_seconds)
            return True
        except TimeoutError:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        try:
            await asyncio.wait_for(process.wait(), timeout=self._stop_timeout_seconds)
            return True
        except TimeoutError:
            return False

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            self._stderr_observed = True


async def probe_trailmark_child(scratch_root: Path, *, timeout_seconds: float = 5.0) -> bool:
    """Run one finite offline build and always tear down its child/mirror."""

    if timeout_seconds <= 0:
        raise ValueError("Trailmark probe timeout must be positive")
    source = "def probe_leaf():\n    return 1\n\ndef main():\n    return probe_leaf()\n"
    size = len(source.encode("utf-8"))
    snapshot = WorkspaceSnapshot(
        directories=(),
        files=(WorkspaceTextFile(path="probe.py", text=source, size=size),),
        binary_paths=(),
        digest="sha256:" + "0" * 64,
    )
    host = TrailmarkChildHost(
        scratch_root,
        build_timeout_seconds=timeout_seconds,
        query_timeout_seconds=timeout_seconds,
        stop_timeout_seconds=min(1.0, timeout_seconds),
    )
    try:
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await host.build(snapshot)
                return result.node_count >= 2 and result.languages == ("python",)
        except (TimeoutError, TrailmarkHostError, OSError):
            return False
    finally:
        await host.close()


def _child_command() -> tuple[str, ...]:
    executable = str(Path(sys.executable))
    if not Path(executable).is_absolute():
        raise RuntimeError("Python executable must be absolute")
    return (
        executable,
        "-I",
        "-m",
        "contractor_runtime.toolsets.trailmark_child",
    )


def _materialize_snapshot(snapshot: WorkspaceSnapshot, scratch_root: Path) -> _PreparedMirror:
    scratch_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = scratch_root.resolve()
    mirror = root / f"{MIRROR_PREFIX}{uuid.uuid4().hex}"
    if mirror.parent != root:
        raise ValueError("generated graph mirror escaped scratch root")
    mirror.mkdir(mode=0o700)
    try:
        candidates, coverage = _admit_snapshot(snapshot)
        for item in candidates:
            relative = _safe_relative_path(item.path)
            destination = mirror.joinpath(*PurePosixPath(relative).parts)
            if destination.parent != mirror:
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            encoded = item.text.encode("utf-8")
            if len(encoded) != item.size or b"\x00" in encoded:
                raise ValueError("invalid managed text snapshot")
            with destination.open("xb") as stream:
                stream.write(encoded)
            destination.chmod(0o600)

        # Trailmark's entrypoint detector otherwise walks parent directories
        # looking for repository metadata.  An empty valid override anchors its
        # public detector at the private mirror without changing the graph.
        boundary = mirror / _ENTRYPOINT_BOUNDARY
        boundary.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with boundary.open("xb") as stream:
            stream.write(b"")
        boundary.chmod(0o600)
        return _PreparedMirror(mirror, snapshot.digest, coverage)
    except BaseException:
        shutil.rmtree(mirror, ignore_errors=True)
        raise


async def _materialize_snapshot_cancellation_safe(
    snapshot: WorkspaceSnapshot, scratch_root: Path
) -> _PreparedMirror:
    task = asyncio.create_task(
        asyncio.to_thread(_materialize_snapshot, snapshot, scratch_root),
        name="trailmark-mirror-materialize",
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            mirror = await task
            await asyncio.to_thread(_remove_mirror, mirror.path, scratch_root)
        except Exception:
            raise TrailmarkHostError("code_analysis_engine_failed") from None
        raise


def _admit_snapshot(
    snapshot: WorkspaceSnapshot,
) -> tuple[tuple[WorkspaceTextFile, ...], GraphCoverage]:
    supported: list[WorkspaceTextFile] = []
    unsupported = 0
    oversized = 0
    for item in sorted(snapshot.files, key=lambda candidate: candidate.path):
        suffix = PurePosixPath(item.path).suffix
        if suffix in language_support.GRAPH_EXTENSION_LANGUAGES:
            if item.size > MAX_GRAPH_FILE_BYTES:
                oversized += 1
            else:
                supported.append(item)
        elif language_support.detect_language(item.path) is not None:
            unsupported += 1

    reasons: set[str] = set()
    if len(supported) > MAX_GRAPH_FILES:
        reasons.add("file_limit")
        supported = supported[:MAX_GRAPH_FILES]

    admitted: list[WorkspaceTextFile] = []
    admitted_bytes = 0
    for item in supported:
        if admitted_bytes + item.size > MAX_GRAPH_BYTES:
            reasons.add("byte_limit")
            break
        admitted.append(item)
        admitted_bytes += item.size
    coverage = GraphCoverage(
        analyzed_files=len(admitted),
        analyzed_bytes=admitted_bytes,
        binary_files=len(snapshot.binary_paths),
        unsupported_source_files=unsupported,
        oversized_files=oversized,
        parse_errors=0,
        reasons=tuple(sorted(reasons)),
    )
    return tuple(admitted), coverage


def _safe_relative_path(value: str) -> str:
    try:
        return normalize_project_path(value, allow_root=False)
    except ProjectPathError:
        raise ValueError("invalid graph source path") from None


def _remove_mirror(path: Path, scratch_root: Path) -> None:
    root = scratch_root.resolve()
    if (
        path.parent != root
        or not path.name.startswith(MIRROR_PREFIX)
        or path == root
        or path.is_symlink()
    ):
        raise ValueError("refusing to remove an unowned graph mirror")
    if not path.exists():
        return
    shutil.rmtree(path)


def _encode_request(document: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise TrailmarkHostError("code_analysis_engine_failed") from None
    if not 0 < len(payload) <= MAX_REQUEST_BYTES:
        raise TrailmarkHostError("code_analysis_capacity_exceeded")
    return payload


async def _read_response(stream: asyncio.StreamReader) -> object:
    header = await stream.readexactly(4)
    length = struct.unpack(">I", header)[0]
    if not 0 < length <= MAX_RESPONSE_BYTES:
        raise TrailmarkHostError("code_analysis_capacity_exceeded")
    payload = await stream.readexactly(length)
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, ValueError):
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True) from None


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _validate_response(value: object, request_id: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    if value.get("schemaVersion") != SCHEMA_VERSION or value.get("requestId") != request_id:
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    if value.get("ok") is True:
        if set(value) != {"schemaVersion", "requestId", "ok", "result"}:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        result = value["result"]
        if not isinstance(result, dict):
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        return result
    if value.get("ok") is False:
        if set(value) != {"schemaVersion", "requestId", "ok", "code", "retryable"}:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        code = value.get("code")
        retryable = value.get("retryable")
        if not isinstance(code, str) or not isinstance(retryable, bool):
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        if code not in {
            "code_analysis_capacity_exceeded",
            "code_analysis_input_invalid",
            "code_analysis_stale_symbol",
            "code_analysis_symbol_not_found",
        }:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        raise _ChildResponseError(code, retryable=retryable)
    raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)


def _build_result(value: Mapping[str, Any], mirror: _PreparedMirror) -> GraphBuildResult:
    keys = {
        "snapshotDigest",
        "coverage",
        "languages",
        "nodeCount",
        "edgeCount",
        "callEdgeCount",
        "entrypointCount",
        "dependencyCount",
        "rssKiB",
    }
    if set(value) != keys or value.get("snapshotDigest") != mirror.snapshot_digest:
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    if value.get("coverage") != mirror.coverage.wire():
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    languages = value.get("languages")
    counts = [
        value.get("nodeCount"),
        value.get("edgeCount"),
        value.get("callEdgeCount"),
        value.get("entrypointCount"),
        value.get("dependencyCount"),
        value.get("rssKiB"),
    ]
    if (
        not isinstance(languages, list)
        or any(not isinstance(language, str) or not language for language in languages)
        or languages != sorted(set(languages))
        or not set(languages) <= set(language_support.GRAPH_EXTENSION_LANGUAGES.values())
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= 9_223_372_036_854_775_807
            for count in counts
        )
        or counts[2] > counts[1]
        or counts[3] > counts[0]
    ):
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    return GraphBuildResult(
        snapshot_digest=mirror.snapshot_digest,
        coverage=mirror.coverage,
        languages=tuple(languages),
        node_count=counts[0],
        edge_count=counts[1],
        call_edge_count=counts[2],
        entrypoint_count=counts[3],
        dependency_count=counts[4],
        rss_kib=counts[5],
    )


def _symbol_page(
    value: Mapping[str, Any],
    expected_digest: str,
    symbol_key: bytes,
    *,
    expected_offset: int,
    expected_limit: int,
) -> GraphSymbolPage:
    if set(value) != {"snapshotDigest", "items", "observedTotal", "truncated"}:
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    digest = value.get("snapshotDigest")
    rows = value.get("items")
    observed_total = value.get("observedTotal")
    truncated = value.get("truncated")
    if (
        digest != expected_digest
        or not isinstance(rows, list)
        or len(rows) > 200
        or not isinstance(observed_total, int)
        or isinstance(observed_total, bool)
        or not 0 <= observed_total <= 2_147_483_647
        or not 0 <= expected_offset <= observed_total
        or len(rows) != min(expected_limit, observed_total - expected_offset)
        or not isinstance(truncated, bool)
        or truncated != (expected_offset + len(rows) < observed_total)
    ):
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    projected: list[GraphSymbolProjection] = []
    for row in rows:
        projected.append(_symbol_projection(row, expected_digest, symbol_key))
    return GraphSymbolPage(expected_digest, tuple(projected), observed_total, truncated)


def _relationship_page(
    value: Mapping[str, Any],
    expected_digest: str,
    symbol_key: bytes,
    *,
    expected_offset: int,
    expected_limit: int,
) -> GraphRelationshipPage:
    if set(value) != {"snapshotDigest", "items", "observedTotal", "truncated"}:
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    if value.get("snapshotDigest") != expected_digest:
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    rows = value.get("items")
    observed_total = value.get("observedTotal")
    truncated = value.get("truncated")
    if (
        not isinstance(rows, list)
        or len(rows) > 200
        or not isinstance(observed_total, int)
        or isinstance(observed_total, bool)
        or not 0 <= observed_total <= 2_147_483_647
        or not 0 <= expected_offset <= observed_total
        or len(rows) != min(expected_limit, observed_total - expected_offset)
        or not isinstance(truncated, bool)
        or truncated != (expected_offset + len(rows) < observed_total)
    ):
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    projected: list[GraphRelationshipProjection] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "symbolId",
            "name",
            "kind",
            "path",
            "line",
            "endLine",
            "column",
            "confidence",
        }:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        confidence = row["confidence"]
        if confidence not in {"certain", "inferred", "uncertain"}:
            raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
        symbol = _symbol_projection(
            {key: row[key] for key in row if key != "confidence"},
            expected_digest,
            symbol_key,
        )
        projected.append(GraphRelationshipProjection(symbol, confidence))
    return GraphRelationshipPage(expected_digest, tuple(projected), observed_total, truncated)


def _symbol_projection(
    row: object, expected_digest: str, symbol_key: bytes
) -> GraphSymbolProjection:
    if not isinstance(row, dict) or set(row) != {
        "symbolId",
        "name",
        "kind",
        "path",
        "line",
        "endLine",
        "column",
    }:
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    symbol_id = row["symbolId"]
    try:
        decoded = decode_symbol_id(symbol_key, symbol_id)
    except (TypeError, ValueError):
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True) from None
    if decoded.snapshot_digest != expected_digest:
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    if (
        not _bounded_protocol_text(row["name"], 256, allow_empty=True)
        or not _bounded_protocol_text(row["kind"], 64, allow_empty=False)
        or not _valid_projection_path(row["path"])
        or any(
            not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0
            for key in ("line", "endLine", "column")
        )
        or row["line"] < 1
        or row["endLine"] < 1
    ):
        raise TrailmarkHostError("code_analysis_engine_failed", retryable=True)
    return GraphSymbolProjection(
        symbol_id=symbol_id,
        name=row["name"],
        kind=row["kind"],
        path=row["path"],
        line=row["line"],
        end_line=row["endLine"],
        column=row["column"],
    )


def _bounded_protocol_text(value: object, limit: int, *, allow_empty: bool) -> bool:
    if not isinstance(value, str) or (not allow_empty and not value) or len(value) > limit:
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return not any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _valid_projection_path(value: object) -> bool:
    if not isinstance(value, str) or value.startswith("/"):
        return False
    try:
        if not 1 <= len(value.encode("utf-8")) <= 4096:
            return False
        if normalize_project_path(value, allow_root=False) != value:
            return False
    except (ProjectPathError, UnicodeError):
        return False
    return True
