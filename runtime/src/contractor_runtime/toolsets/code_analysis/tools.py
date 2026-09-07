"""Bounded structural code analysis over one immutable workspace snapshot."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jcs
from tree_sitter import Parser

import contractor_runtime.toolsets.code_analysis.languages as language_support
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.projectfs.paths import ProjectPathError, normalize_project_path
from contractor_runtime.projectfs.storage import (
    WorkspaceReader,
    WorkspaceSnapshot,
    WorkspaceStorageError,
    WorkspaceTextFile,
)
from contractor_runtime.toolsets.code_analysis.ids import MAX_SYMBOL_ID_BYTES
from contractor_runtime.toolsets.code_analysis.languages import Language, SymbolRecord
from contractor_runtime.toolsets.code_analysis.trailmark_host import (
    GraphBuildResult,
    GraphComplexityProjection,
    GraphCoverage,
    GraphEntrypointProjection,
    GraphPathPage,
    GraphRelationshipProjection,
    GraphSymbolProjection,
    TrailmarkChildHost,
    TrailmarkHostError,
    probe_trailmark_child,
)
from contractor_runtime.toolsets.common.metrics import ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

CODE_ANALYSIS_REF = "code-analysis@1"

SHALLOW_TOOLS = frozenset({"list_symbols", "search_def"})
CORE_GRAPH_TOOLS = frozenset({"find_callees", "find_callers", "find_symbol", "graph_summary"})
ADVANCED_GRAPH_TOOLS = frozenset(
    {
        "attack_surface",
        "complexity_hotspots",
        "entrypoint_paths_to",
        "functions_that_raise",
        "paths_between",
    }
)
GRAPH_TOOLS = CORE_GRAPH_TOOLS | ADVANCED_GRAPH_TOOLS
EXPORTED_TOOLS = SHALLOW_TOOLS | GRAPH_TOOLS

PINNED_DEPENDENCIES = MappingProxyType(
    {
        "trailmark": "0.5.0",
        "tree-sitter": "0.25.2",
        "tree-sitter-language-pack": "1.14.3",
    }
)
SHALLOW_PINNED_DEPENDENCIES = MappingProxyType(
    {name: PINNED_DEPENDENCIES[name] for name in ("tree-sitter", "tree-sitter-language-pack")}
)

MAX_SOURCE_FILES = 20_000
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 4 * 1024 * 1024
MAX_COMPACT_SYMBOLS = 100_000
MAX_COMPACT_CACHE_FILES = 20_000
MAX_PAGE_ITEMS = 200
MAX_RESULT_BYTES = 256 * 1024
MAX_QUERY_CHARS = 256
MAX_SCAN_SECONDS = 10.0
MAX_CURSOR_BYTES = 2048
MAX_PREVIEW_LINES = 12
MAX_PREVIEW_BYTES = 4096

_NODE_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")


def dependency_versions_match(
    required: Mapping[str, str] = PINNED_DEPENDENCIES,
) -> bool:
    """Return whether the requested reviewed distributions are installed exactly."""

    try:
        return all(version(name) == expected for name, expected in required.items())
    except PackageNotFoundError:
        return False


class CodeAnalysisError(RuntimeError):
    """Stable model-facing failure without source, query, or physical path detail."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"Code analysis operation failed ({code})")


class CodeAnalysisToolsetFactory:
    ref = CODE_ANALYSIS_REF
    exported_tools = EXPORTED_TOOLS
    infrastructure_channels = MappingProxyType({})
    requires_workspace = True
    workspace_access = "read"

    def __init__(
        self,
        *,
        workspace_storage: str | None = None,
        graph_probe_root: Path | None = None,
    ) -> None:
        if workspace_storage not in {None, "local", "memory"}:
            raise ValueError("unsupported code-analysis workspace storage")
        if workspace_storage == "local" and graph_probe_root is None:
            raise ValueError("local graph probing requires private scratch")
        self._workspace_storage = workspace_storage
        self._graph_probe_root = graph_probe_root
        self._graph_probe_succeeded = False
        self._available_tools = SHALLOW_TOOLS

    @property
    def graph_probe_succeeded(self) -> bool:
        return self._graph_probe_succeeded

    async def probe(self) -> frozenset[str]:
        self._graph_probe_succeeded = False
        self._available_tools = frozenset()
        if not dependency_versions_match(SHALLOW_PINNED_DEPENDENCIES):
            return frozenset()
        available = await asyncio.to_thread(language_support.probe_all_parsers)
        if not available:
            return frozenset()
        self._available_tools = SHALLOW_TOOLS
        if (
            self._workspace_storage == "local"
            and self._graph_probe_root is not None
            and dependency_versions_match()
        ):
            self._graph_probe_succeeded = await probe_trailmark_child(self._graph_probe_root)
        if self._graph_probe_succeeded:
            self._available_tools |= GRAPH_TOOLS
        return self._available_tools

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
        project_workspace: WorkspaceReader | None = None,
    ) -> Mapping[str, Any]:
        del allocation_id, run_id, namespace, runtime_settings, adapter_handles
        unavailable = sorted(set(selected) - self._available_tools)
        if unavailable:
            raise ValueError(f"unavailable selected code-analysis tools: {', '.join(unavailable)}")
        if project_workspace is None:
            raise CodeAnalysisError("workspace_required")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("code-analysis@1 requires State.metrics")
        graph_selected = bool(set(selected) & GRAPH_TOOLS)
        session = _CodeAnalysisSession(
            project_workspace,
            graph_scratch=workspace.path if graph_selected else None,
        )
        builders = {
            "attack_surface": lambda: AttackSurfaceTool(session, metrics),
            "complexity_hotspots": lambda: ComplexityHotspotsTool(session, metrics),
            "entrypoint_paths_to": lambda: EntrypointPathsToTool(session, metrics),
            "find_callees": lambda: FindCalleesTool(session, metrics),
            "find_callers": lambda: FindCallersTool(session, metrics),
            "find_symbol": lambda: FindSymbolTool(session, metrics),
            "functions_that_raise": lambda: FunctionsThatRaiseTool(session, metrics),
            "graph_summary": lambda: GraphSummaryTool(session, metrics),
            "list_symbols": lambda: ListSymbolsTool(session, metrics),
            "paths_between": lambda: PathsBetweenTool(session, metrics),
            "search_def": lambda: SearchDefinitionTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


@dataclass(slots=True)
class _Coverage:
    analyzed_files: int = 0
    analyzed_bytes: int = 0
    binary_files: int = 0
    unsupported_source_files: int = 0
    oversized_files: int = 0
    parse_errors: int = 0
    reasons: set[str] = field(default_factory=set)

    def wire(self) -> dict[str, Any]:
        return {
            "analyzedFiles": self.analyzed_files,
            "analyzedBytes": self.analyzed_bytes,
            "binaryFiles": self.binary_files,
            "unsupportedSourceFiles": self.unsupported_source_files,
            "oversizedFiles": self.oversized_files,
            "parseErrors": self.parse_errors,
            "incomplete": bool(self.reasons),
            "reasons": sorted(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class _Cursor:
    snapshot: str
    operation: str
    query: str
    offset: int


@dataclass(frozen=True, slots=True)
class _CachedFile:
    symbols: tuple[SymbolRecord, ...]
    parse_error: bool


@dataclass(frozen=True, slots=True)
class _ScanStats:
    cache_hits: int
    cache_misses: int
    cache_invalidations: int


@dataclass(frozen=True, slots=True)
class _OperationResult:
    value: dict[str, Any]
    metric: dict[str, Any]


class _CodeAnalysisSession:
    def __init__(self, reader: WorkspaceReader, *, graph_scratch: Path | None = None) -> None:
        self._reader: WorkspaceReader | None = reader
        self._lock = asyncio.Lock()
        self._cursor_key = bytearray(secrets.token_bytes(32))
        self._digest: str | None = None
        self._file_cache: dict[str, _CachedFile] = {}
        self._cached_symbols = 0
        self._parsers: dict[Language, Parser] = {}
        self._graph_host = TrailmarkChildHost(graph_scratch) if graph_scratch is not None else None
        self._graph_result: GraphBuildResult | None = None
        self._graph_builds = 0
        self._graph_rebuilds = 0
        self._closed = False
        self._closing = False

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closing = True
            if self._graph_host is not None:
                try:
                    await self._graph_host.close()
                except TrailmarkHostError as error:
                    raise CodeAnalysisError(error.code, retryable=error.retryable) from None
                self._graph_host = None
            self._clear_derived_state()
            self._digest = None
            self._reader = None
            self._cursor_key[:] = b"\x00" * len(self._cursor_key)
            self._closed = True

    async def search_def(
        self,
        symbol: str,
        path: str,
        language: str,
        cursor: str,
        limit: int,
    ) -> _OperationResult:
        normalized_symbol = _query_string(symbol)
        normalized_path = _path(path)
        selected_language = _language(language)
        resolved_limit = _limit(limit)
        query = _query_digest(
            {
                "operation": "search_def",
                "symbol": normalized_symbol,
                "path": normalized_path,
                "language": selected_language.value if selected_language else "",
            }
        )
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            offset = (
                self._cursor_offset(cursor, snapshot.digest, "search_def", query) if cursor else 0
            )
            symbols, coverage, stats = await self._scan(
                snapshot,
                normalized_path,
                selected_language,
                search_symbol=normalized_symbol,
                cache_invalidations=invalidations,
            )
            files = {item.path: item for item in snapshot.files}
            matches = [item for item in symbols if _symbol_matches(item.name, normalized_symbol)]
            matches.sort(key=_symbol_sort_key)
            rows = await _to_thread_cancellation_safe(
                _definition_rows,
                tuple(matches),
                files,
            )
            value = self._page(
                rows,
                offset=offset,
                limit=resolved_limit,
                snapshot=snapshot.digest,
                operation="search_def",
                query=query,
                coverage=coverage,
            )
            return _OperationResult(value, _metric(value, coverage, stats))

    async def list_symbols(
        self,
        path: str,
        language: str,
        node_type: str,
        cursor: str,
        limit: int,
    ) -> _OperationResult:
        normalized_path = _path(path)
        selected_language = _language(language)
        normalized_node_type = _node_type(node_type)
        resolved_limit = _limit(limit)
        query = _query_digest(
            {
                "operation": "list_symbols",
                "path": normalized_path,
                "language": selected_language.value if selected_language else "",
                "nodeType": normalized_node_type,
            }
        )
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            offset = (
                self._cursor_offset(cursor, snapshot.digest, "list_symbols", query) if cursor else 0
            )
            symbols, coverage, stats = await self._scan(
                snapshot,
                normalized_path,
                selected_language,
                search_symbol=None,
                cache_invalidations=invalidations,
            )
            if normalized_node_type:
                symbols = tuple(item for item in symbols if item.node_type == normalized_node_type)
            rows = [_symbol_row(item) for item in sorted(symbols, key=_symbol_sort_key)]
            value = self._page(
                rows,
                offset=offset,
                limit=resolved_limit,
                snapshot=snapshot.digest,
                operation="list_symbols",
                query=query,
                coverage=coverage,
            )
            return _OperationResult(value, _metric(value, coverage, stats))

    async def graph_summary(self) -> _OperationResult:
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            graph = await self._ensure_graph(snapshot)
            value = {
                "nodeCount": graph.node_count,
                "callEdgeCount": graph.call_edge_count,
                "entrypointCount": graph.entrypoint_count,
                "languages": list(graph.languages),
                "coverage": graph.coverage.wire(),
            }
            return _OperationResult(
                value,
                self._graph_metric(
                    count=graph.node_count,
                    truncated=False,
                    graph=graph,
                    invalidations=invalidations,
                ),
            )

    async def find_symbol(self, query: str, cursor: str, limit: int) -> _OperationResult:
        normalized_query = _query_string(query)
        resolved_limit = _limit(limit)
        query_digest = _query_digest(
            {"operation": "find_symbol", "query": normalized_query.casefold()}
        )
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            offset = (
                self._cursor_offset(
                    cursor,
                    snapshot.digest,
                    "find_symbol",
                    query_digest,
                )
                if cursor
                else 0
            )
            graph = await self._ensure_graph(snapshot)
            host = self._require_graph_host()
            try:
                page = await host.find_symbols(
                    normalized_query,
                    offset=offset,
                    limit=resolved_limit,
                )
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            rows = [_graph_symbol_row(item) for item in page.items]
            value = self._graph_page(
                rows,
                observed_total=page.observed_total,
                offset=offset,
                limit=resolved_limit,
                snapshot=snapshot.digest,
                operation="find_symbol",
                query=query_digest,
                coverage=graph.coverage,
            )
            return _OperationResult(
                value,
                self._graph_metric(
                    count=len(value["items"]),
                    truncated=bool(value["truncated"]),
                    graph=graph,
                    invalidations=invalidations,
                ),
            )

    async def find_relationships(
        self,
        operation: str,
        symbol_id: str,
        cursor: str,
        limit: int,
    ) -> _OperationResult:
        if operation not in {"find_callers", "find_callees"}:
            raise CodeAnalysisError("code_analysis_input_invalid")
        if (
            not isinstance(symbol_id, str)
            or not symbol_id
            or not symbol_id.isascii()
            or len(symbol_id) > MAX_SYMBOL_ID_BYTES
        ):
            raise CodeAnalysisError("code_analysis_symbol_not_found")
        resolved_limit = _limit(limit)
        query_digest = _query_digest({"operation": operation, "symbolId": symbol_id})
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            host = self._require_graph_host()
            try:
                host.validate_symbol_id(symbol_id, snapshot.digest)
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            offset = (
                self._cursor_offset(
                    cursor,
                    snapshot.digest,
                    operation,
                    query_digest,
                )
                if cursor
                else 0
            )
            graph = await self._ensure_graph(snapshot)
            try:
                page = await host.relationships(
                    operation,
                    symbol_id,
                    offset=offset,
                    limit=resolved_limit,
                )
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            rows = [_graph_relationship_row(item) for item in page.items]
            value = self._graph_page(
                rows,
                observed_total=page.observed_total,
                offset=offset,
                limit=resolved_limit,
                snapshot=snapshot.digest,
                operation=operation,
                query=query_digest,
                coverage=graph.coverage,
            )
            return _OperationResult(
                value,
                self._graph_metric(
                    count=len(value["items"]),
                    truncated=bool(value["truncated"]),
                    graph=graph,
                    invalidations=invalidations,
                ),
            )

    async def paths_between(
        self,
        source_id: str,
        target_id: str,
        max_depth: int,
        limit: int,
    ) -> _OperationResult:
        normalized_source = _symbol_id(source_id)
        normalized_target = _symbol_id(target_id)
        resolved_depth = _path_depth(max_depth)
        resolved_limit = _path_limit(limit)
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            host = self._require_graph_host()
            self._validate_symbol_id(host, normalized_source, snapshot.digest)
            self._validate_symbol_id(host, normalized_target, snapshot.digest)
            graph = await self._ensure_graph(snapshot)
            try:
                page = await host.paths_between(
                    normalized_source,
                    normalized_target,
                    max_depth=resolved_depth,
                    limit=resolved_limit,
                )
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            return self._path_operation_result(
                page,
                graph,
                invalidations=invalidations,
            )

    async def entrypoint_paths_to(
        self,
        symbol_id: str,
        max_depth: int,
        limit: int,
    ) -> _OperationResult:
        normalized_symbol = _symbol_id(symbol_id)
        resolved_depth = _path_depth(max_depth)
        resolved_limit = _path_limit(limit)
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            host = self._require_graph_host()
            self._validate_symbol_id(host, normalized_symbol, snapshot.digest)
            graph = await self._ensure_graph(snapshot)
            try:
                page = await host.entrypoint_paths_to(
                    normalized_symbol,
                    max_depth=resolved_depth,
                    limit=resolved_limit,
                )
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            return self._path_operation_result(
                page,
                graph,
                invalidations=invalidations,
            )

    async def attack_surface(self, cursor: str, limit: int) -> _OperationResult:
        resolved_limit = _limit(limit)
        query_digest = _query_digest({"operation": "attack_surface"})
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            offset = (
                self._cursor_offset(cursor, snapshot.digest, "attack_surface", query_digest)
                if cursor
                else 0
            )
            graph = await self._ensure_graph(snapshot)
            host = self._require_graph_host()
            try:
                page = await host.attack_surface(offset=offset, limit=resolved_limit)
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            rows = [_graph_entrypoint_row(item) for item in page.items]
            value = self._graph_page(
                rows,
                observed_total=page.observed_total,
                offset=offset,
                limit=resolved_limit,
                snapshot=snapshot.digest,
                operation="attack_surface",
                query=query_digest,
                coverage=graph.coverage,
            )
            return _OperationResult(
                value,
                self._graph_metric(
                    count=len(value["items"]),
                    truncated=bool(value["truncated"]),
                    graph=graph,
                    invalidations=invalidations,
                ),
            )

    async def complexity_hotspots(
        self,
        threshold: int,
        cursor: str,
        limit: int,
    ) -> _OperationResult:
        resolved_threshold = _complexity_threshold(threshold)
        resolved_limit = _limit(limit)
        query_digest = _query_digest(
            {"operation": "complexity_hotspots", "threshold": resolved_threshold}
        )
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            offset = (
                self._cursor_offset(
                    cursor,
                    snapshot.digest,
                    "complexity_hotspots",
                    query_digest,
                )
                if cursor
                else 0
            )
            graph = await self._ensure_graph(snapshot)
            host = self._require_graph_host()
            try:
                page = await host.complexity_hotspots(
                    resolved_threshold,
                    offset=offset,
                    limit=resolved_limit,
                )
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            rows = [_graph_complexity_row(item) for item in page.items]
            value = self._graph_page(
                rows,
                observed_total=page.observed_total,
                offset=offset,
                limit=resolved_limit,
                snapshot=snapshot.digest,
                operation="complexity_hotspots",
                query=query_digest,
                coverage=graph.coverage,
            )
            return _OperationResult(
                value,
                self._graph_metric(
                    count=len(value["items"]),
                    truncated=bool(value["truncated"]),
                    graph=graph,
                    invalidations=invalidations,
                ),
            )

    async def functions_that_raise(
        self,
        exception: str,
        cursor: str,
        limit: int,
    ) -> _OperationResult:
        normalized_exception = _query_string(exception)
        resolved_limit = _limit(limit)
        query_digest = _query_digest(
            {"operation": "functions_that_raise", "exception": normalized_exception}
        )
        async with self._lock:
            snapshot, invalidations = await self._begin_call()
            offset = (
                self._cursor_offset(
                    cursor,
                    snapshot.digest,
                    "functions_that_raise",
                    query_digest,
                )
                if cursor
                else 0
            )
            graph = await self._ensure_graph(snapshot)
            host = self._require_graph_host()
            try:
                page = await host.functions_that_raise(
                    normalized_exception,
                    offset=offset,
                    limit=resolved_limit,
                )
            except TrailmarkHostError as error:
                raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            rows = [_graph_symbol_row(item) for item in page.items]
            value = self._graph_page(
                rows,
                observed_total=page.observed_total,
                offset=offset,
                limit=resolved_limit,
                snapshot=snapshot.digest,
                operation="functions_that_raise",
                query=query_digest,
                coverage=graph.coverage,
            )
            return _OperationResult(
                value,
                self._graph_metric(
                    count=len(value["items"]),
                    truncated=bool(value["truncated"]),
                    graph=graph,
                    invalidations=invalidations,
                ),
            )

    @staticmethod
    def _validate_symbol_id(
        host: TrailmarkChildHost,
        symbol_id: str,
        snapshot_digest: str,
    ) -> None:
        try:
            host.validate_symbol_id(symbol_id, snapshot_digest)
        except TrailmarkHostError as error:
            raise CodeAnalysisError(error.code, retryable=error.retryable) from None

    def _path_operation_result(
        self,
        page: GraphPathPage,
        graph: GraphBuildResult,
        *,
        invalidations: int,
    ) -> _OperationResult:
        paths = [[_graph_symbol_row(item) for item in path] for path in page.items]
        truncated = page.truncated
        while True:
            value = {
                "items": paths,
                "truncated": truncated,
                "coverage": graph.coverage.wire(),
            }
            if len(jcs.canonicalize(value)) <= MAX_RESULT_BYTES:
                metric = self._graph_metric(
                    count=len(paths),
                    truncated=truncated,
                    graph=graph,
                    invalidations=invalidations,
                )
                metric["traversal_steps"] = page.traversal_steps
                metric["path_nodes"] = sum(len(path) for path in paths)
                return _OperationResult(value, metric)
            if not paths:
                raise CodeAnalysisError("code_analysis_capacity_exceeded")
            paths = paths[:-1]
            truncated = True

    async def _ensure_graph(self, snapshot: WorkspaceSnapshot) -> GraphBuildResult:
        host = self._require_graph_host()
        if (
            self._graph_result is not None
            and self._graph_result.snapshot_digest == snapshot.digest
            and host.pid is not None
        ):
            return self._graph_result
        had_graph = self._graph_builds > 0
        try:
            result = await host.build(snapshot)
        except TrailmarkHostError as error:
            self._graph_result = None
            raise CodeAnalysisError(error.code, retryable=error.retryable) from None
        self._graph_builds += 1
        self._graph_rebuilds += int(had_graph)
        self._graph_result = result
        return result

    def _require_graph_host(self) -> TrailmarkChildHost:
        if self._graph_host is None:
            raise CodeAnalysisError("code_analysis_engine_failed")
        return self._graph_host

    def _graph_page(
        self,
        rows: list[dict[str, Any]],
        *,
        observed_total: int,
        offset: int,
        limit: int,
        snapshot: str,
        operation: str,
        query: str,
        coverage: GraphCoverage,
    ) -> dict[str, Any]:
        if offset > observed_total or len(rows) > limit or offset + len(rows) > observed_total:
            raise CodeAnalysisError("code_analysis_engine_failed", retryable=True)
        page = rows
        while True:
            next_offset = offset + len(page)
            next_cursor = (
                self._encode_cursor(snapshot, operation, query, next_offset)
                if next_offset < observed_total
                else None
            )
            result = {
                "items": page,
                "nextCursor": next_cursor,
                "truncated": next_cursor is not None,
                "observedTotal": observed_total,
                "coverage": coverage.wire(),
            }
            if len(jcs.canonicalize(result)) <= MAX_RESULT_BYTES:
                return result
            if not page:
                raise CodeAnalysisError("code_analysis_capacity_exceeded")
            page = page[:-1]

    def _graph_metric(
        self,
        *,
        count: int,
        truncated: bool,
        graph: GraphBuildResult,
        invalidations: int,
    ) -> dict[str, Any]:
        return {
            "engine": "graph",
            "count": count,
            "truncated": truncated,
            "analyzed_files": graph.coverage.analyzed_files,
            "analyzed_bytes": graph.coverage.analyzed_bytes,
            "nodes": graph.node_count,
            "call_edges": graph.call_edge_count,
            "entrypoints": graph.entrypoint_count,
            "binary_files": graph.coverage.binary_files,
            "unsupported_source_files": graph.coverage.unsupported_source_files,
            "oversized_files": graph.coverage.oversized_files,
            "parse_errors": graph.coverage.parse_errors,
            "graph_builds": self._graph_builds,
            "graph_rebuilds": self._graph_rebuilds,
            "cache_invalidations": invalidations,
        }

    async def _begin_call(self) -> tuple[WorkspaceSnapshot, int]:
        if self._closed or self._closing:
            raise CodeAnalysisError("code_analysis_closing", retryable=True)
        reader = self._reader
        if reader is None:
            raise CodeAnalysisError("code_analysis_closing", retryable=True)
        try:
            snapshot = await reader.snapshot()
        except WorkspaceStorageError:
            raise CodeAnalysisError("code_analysis_engine_failed", retryable=True) from None
        except Exception:
            raise CodeAnalysisError("code_analysis_engine_failed", retryable=True) from None
        invalidations = int(self._digest is not None and self._digest != snapshot.digest)
        if self._digest != snapshot.digest:
            if self._digest is not None and self._graph_host is not None:
                try:
                    await self._graph_host.invalidate()
                except TrailmarkHostError as error:
                    raise CodeAnalysisError(error.code, retryable=error.retryable) from None
            self._clear_derived_state()
            self._digest = snapshot.digest
        return snapshot, invalidations

    async def _scan(
        self,
        snapshot: WorkspaceSnapshot,
        path: str,
        selected_language: Language | None,
        *,
        search_symbol: str | None,
        cache_invalidations: int,
    ) -> tuple[tuple[SymbolRecord, ...], _Coverage, _ScanStats]:
        if path and not _snapshot_has_path(snapshot, path):
            raise CodeAnalysisError("code_analysis_input_invalid")
        coverage = _Coverage(
            binary_files=sum(1 for item in snapshot.binary_paths if _under_path(item, path))
        )
        candidates: list[tuple[WorkspaceTextFile, Language]] = []
        for item in sorted(snapshot.files, key=lambda value: value.path):
            if not _under_path(item.path, path):
                continue
            language = language_support.detect_language(item.path)
            if language is None:
                if language_support.graph_only_source(item.path):
                    coverage.unsupported_source_files += 1
                continue
            if selected_language is not None and language is not selected_language:
                continue
            candidates.append((item, language))
        if len(candidates) > MAX_SOURCE_FILES:
            candidates = candidates[:MAX_SOURCE_FILES]
            coverage.reasons.add("file_limit")

        deadline = time.monotonic() + MAX_SCAN_SECONDS
        needle = _bare_name(search_symbol).casefold() if search_symbol is not None else None
        symbols: list[SymbolRecord] = []
        seen_symbols = 0
        cache_hits = 0
        cache_misses = 0
        for item, language in candidates:
            if time.monotonic() >= deadline:
                coverage.reasons.add("deadline")
                break
            if item.size > MAX_SOURCE_FILE_BYTES:
                coverage.oversized_files += 1
                continue
            if coverage.analyzed_bytes + item.size > MAX_SOURCE_BYTES:
                coverage.reasons.add("byte_limit")
                break
            coverage.analyzed_files += 1
            coverage.analyzed_bytes += item.size
            if needle is not None:
                contains = await _to_thread_cancellation_safe(
                    _contains_casefold,
                    item.text,
                    needle,
                )
                if not contains:
                    continue
            if time.monotonic() >= deadline:
                coverage.reasons.add("deadline")
                break

            remaining = MAX_COMPACT_SYMBOLS - seen_symbols
            parsed, cache_hit = await self._symbols_for_file(
                item,
                language,
                remaining + 1,
            )
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            coverage.parse_errors += int(parsed.parse_error)
            if parsed.parse_error:
                coverage.reasons.add("parse_errors")
            admitted = parsed.symbols[:remaining]
            symbols.extend(admitted)
            seen_symbols += len(admitted)
            if parsed.symbol_limit_reached or len(parsed.symbols) > remaining:
                coverage.reasons.add("symbol_limit")
                break

        return (
            tuple(symbols),
            coverage,
            _ScanStats(cache_hits, cache_misses, cache_invalidations),
        )

    async def _symbols_for_file(
        self,
        item: WorkspaceTextFile,
        language: Language,
        parse_limit: int,
    ) -> tuple[language_support.ParseResult, bool]:
        cached = self._file_cache.get(item.path)
        if cached is not None:
            return (
                language_support.ParseResult(cached.symbols, cached.parse_error, False),
                True,
            )
        try:
            parser = self._parsers.get(language)
            if parser is None:
                parser = await _to_thread_cancellation_safe(
                    language_support.load_parser,
                    language,
                )
                self._parsers[language] = parser
            parsed = await _to_thread_cancellation_safe(
                _parse_symbols_text,
                parser,
                item.text,
                item.path,
                language,
                parse_limit,
            )
        except Exception:
            parsed = language_support.ParseResult((), True, False)
        if (
            not parsed.symbol_limit_reached
            and len(self._file_cache) < MAX_COMPACT_CACHE_FILES
            and self._cached_symbols + len(parsed.symbols) <= MAX_COMPACT_SYMBOLS
        ):
            self._file_cache[item.path] = _CachedFile(parsed.symbols, parsed.parse_error)
            self._cached_symbols += len(parsed.symbols)
        return parsed, False

    def _page(
        self,
        rows: list[dict[str, Any]],
        *,
        offset: int,
        limit: int,
        snapshot: str,
        operation: str,
        query: str,
        coverage: _Coverage,
    ) -> dict[str, Any]:
        if offset > len(rows):
            raise CodeAnalysisError("code_analysis_cursor_invalid")
        page = rows[offset : offset + limit]
        while True:
            next_offset = offset + len(page)
            next_cursor = (
                self._encode_cursor(snapshot, operation, query, next_offset)
                if next_offset < len(rows)
                else None
            )
            result = {
                "items": page,
                "nextCursor": next_cursor,
                "truncated": next_cursor is not None,
                "observedTotal": len(rows),
                "coverage": coverage.wire(),
            }
            if len(jcs.canonicalize(result)) <= MAX_RESULT_BYTES:
                return result
            if not page:
                raise CodeAnalysisError("code_analysis_capacity_exceeded")
            page = page[:-1]

    def _cursor_offset(
        self,
        value: str,
        snapshot: str,
        operation: str,
        query: str,
    ) -> int:
        cursor = self._decode_cursor(value)
        if cursor.operation != operation or cursor.query != query:
            raise CodeAnalysisError("code_analysis_cursor_invalid")
        if cursor.snapshot != snapshot:
            raise CodeAnalysisError("code_analysis_workspace_changed", retryable=True)
        return cursor.offset

    def _encode_cursor(self, snapshot: str, operation: str, query: str, offset: int) -> str:
        body = jcs.canonicalize(
            {"snapshot": snapshot, "operation": operation, "query": query, "offset": offset}
        )
        signature = hmac.digest(bytes(self._cursor_key), body, "sha256")
        return f"{_b64(body)}.{_b64(signature)}"

    def _decode_cursor(self, value: str) -> _Cursor:
        if not isinstance(value, str) or not value or len(value) > MAX_CURSOR_BYTES:
            raise CodeAnalysisError("code_analysis_cursor_invalid")
        try:
            encoded_body, encoded_signature = value.split(".", 1)
            body = _unb64(encoded_body)
            signature = _unb64(encoded_signature)
            expected = hmac.digest(bytes(self._cursor_key), body, "sha256")
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            document = json.loads(body)
            if jcs.canonicalize(document) != body or set(document) != {
                "snapshot",
                "operation",
                "query",
                "offset",
            }:
                raise ValueError
            if (
                not isinstance(document["snapshot"], str)
                or not isinstance(document["operation"], str)
                or not isinstance(document["query"], str)
                or not isinstance(document["offset"], int)
                or isinstance(document["offset"], bool)
                or document["offset"] < 0
            ):
                raise ValueError
            return _Cursor(**document)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise CodeAnalysisError("code_analysis_cursor_invalid") from None

    def _clear_derived_state(self) -> None:
        self._file_cache.clear()
        self._cached_symbols = 0
        self._parsers.clear()
        self._graph_result = None


class _BaseCodeAnalysisTool:
    name: str
    description: str

    def __init__(self, session: _CodeAnalysisSession, metrics: ToolMetrics) -> None:
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        await self._session.close()

    def _success(self, started_ns: int, result: _OperationResult) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments={},
            result=result.metric,
            duration_ms=_elapsed_ms(started_ns),
        )

    def _failure(self, started_ns: int, error: Exception) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments={},
            error=error,
            duration_ms=_elapsed_ms(started_ns),
        )

    def _cancelled(self, started_ns: int) -> None:
        self._failure(
            started_ns,
            CodeAnalysisError("code_analysis_cancelled", retryable=True),
        )


class SearchDefinitionTool(_BaseCodeAnalysisTool):
    name = "search_def"
    description = """Find structural symbol definitions in the current project workspace.

    Args:
        symbol: Symbol name to match.
        path: Project-relative file or subtree; empty searches the whole workspace.
        language: Supported language name, such as "python"; empty includes all.
        cursor: Opaque nextCursor from the same query; empty starts a new search.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum definitions per page, from 1 to 200; defaults to 50.

    Returns:
        Definition items with source locations, observedTotal, nextCursor,
        truncated and coverage. Only structurally parsed definitions are returned.
    """

    async def __call__(
        self,
        symbol: str,
        path: str = "",
        language: str = "",
        cursor: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.search_def(symbol, path, language, cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class ListSymbolsTool(_BaseCodeAnalysisTool):
    name = "list_symbols"
    description = """List structural symbol definitions in the current project workspace.

    Args:
        path: Project-relative file or subtree; empty scans the whole workspace.
        language: Supported language name, such as "python"; empty includes all.
        node_type: Symbol kind, such as "function" or "class"; empty includes all.
        cursor: Opaque nextCursor from the same query; empty starts a new listing.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum symbols per page, from 1 to 200; defaults to 100.

    Returns:
        Symbol items and locations, observedTotal, nextCursor, truncated and coverage.
    """

    async def __call__(
        self,
        path: str = "",
        language: str = "",
        node_type: str = "",
        cursor: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.list_symbols(path, language, node_type, cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class GraphSummaryTool(_BaseCodeAnalysisTool):
    name = "graph_summary"
    description = """Report the size and coverage of the current workspace code graph.

    Use this before graph queries to understand which source was analyzed.

    Returns:
        nodeCount, callEdgeCount, entrypointCount, languages and coverage.
        Missing edges or entrypoints may reflect analysis coverage limits.
    """

    async def __call__(self) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.graph_summary()
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class FindSymbolTool(_BaseCodeAnalysisTool):
    name = "find_symbol"
    description = """Resolve a symbol query to matching exact graph symbol IDs.

    Use the returned IDs in caller, callee and path queries.

    Args:
        query: Symbol name or exact graph symbol ID to look up.
        cursor: Opaque nextCursor from the same query; empty starts a new search.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum symbols per page, from 1 to 200; defaults to 50.

    Returns:
        Matching symbol items and locations, nextCursor, truncated and coverage.
    """

    async def __call__(
        self,
        query: str,
        cursor: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.find_symbol(query, cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class FindCallersTool(_BaseCodeAnalysisTool):
    name = "find_callers"
    description = """List direct callers of an exact graph symbol.

    Args:
        symbol_id: Exact ID returned by find_symbol; resolve names before calling.
        cursor: Opaque nextCursor from the same query; empty starts a new listing.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum callers per page, from 1 to 200; defaults to 100.

    Returns:
        Caller items with edge metadata, nextCursor, truncated and coverage.
        An empty result is limited to the analyzed graph.
    """

    async def __call__(
        self,
        symbol_id: str,
        cursor: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.find_relationships(self.name, symbol_id, cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class FindCalleesTool(_BaseCodeAnalysisTool):
    name = "find_callees"
    description = """List direct callees of an exact graph symbol.

    Args:
        symbol_id: Exact ID returned by find_symbol; resolve names before calling.
        cursor: Opaque nextCursor from the same query; empty starts a new listing.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum callees per page, from 1 to 200; defaults to 100.

    Returns:
        Callee items with edge metadata, nextCursor, truncated and coverage.
        An empty result is limited to the analyzed graph.
    """

    async def __call__(
        self,
        symbol_id: str,
        cursor: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.find_relationships(self.name, symbol_id, cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class PathsBetweenTool(_BaseCodeAnalysisTool):
    name = "paths_between"
    description = """Find bounded direct-call paths between two exact graph symbols.

    Args:
        source_id: Exact starting caller ID returned by find_symbol.
        target_id: Exact destination callee ID returned by find_symbol.
        max_depth: Maximum call-path exploration depth, from 1 to 20; defaults to 20.
        limit: Maximum paths to return, from 1 to 50; defaults to 20.

    Returns:
        Path items, truncated and coverage. An empty result means no path was
        found within the graph coverage and traversal limits.
    """

    async def __call__(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 20,
        limit: int = 20,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.paths_between(
                source_id,
                target_id,
                max_depth,
                limit,
            )
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class EntrypointPathsToTool(_BaseCodeAnalysisTool):
    name = "entrypoint_paths_to"
    description = """Find bounded call paths from detected entrypoints to an exact graph symbol.

    Use this to investigate reachability from framework-detected handlers.

    Args:
        symbol_id: Exact target ID returned by find_symbol.
        max_depth: Maximum call-path exploration depth, from 1 to 20; defaults to 20.
        limit: Maximum paths to return, from 1 to 50; defaults to 20.

    Returns:
        Path items, truncated and coverage. An empty result does not establish
        unreachability beyond the analyzed graph and traversal limits.
    """

    async def __call__(
        self,
        symbol_id: str,
        max_depth: int = 20,
        limit: int = 20,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.entrypoint_paths_to(symbol_id, max_depth, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class AttackSurfaceTool(_BaseCodeAnalysisTool):
    name = "attack_surface"
    description = """List framework-detected entrypoints in the current workspace graph.

    Args:
        cursor: Opaque nextCursor from this listing; empty starts a new page set.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum entrypoints per page, from 1 to 200; defaults to 100.

    Returns:
        Entrypoint items with reviewed trust metadata, nextCursor, truncated
        and coverage. Detection is limited to supported frameworks and source.
    """

    async def __call__(self, cursor: str = "", limit: int = 100) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.attack_surface(cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class ComplexityHotspotsTool(_BaseCodeAnalysisTool):
    name = "complexity_hotspots"
    description = """Find graph symbols at or above a cyclomatic-complexity threshold.

    Args:
        threshold: Minimum complexity to include, from 1 to 10000; defaults to 10.
        cursor: Opaque nextCursor from the same query; empty starts a new listing.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum symbols per page, from 1 to 200; defaults to 100.

    Returns:
        Matching symbols with locations and complexity, nextCursor, truncated
        and coverage.
    """

    async def __call__(
        self,
        threshold: int = 10,
        cursor: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.complexity_hotspots(threshold, cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


class FunctionsThatRaiseTool(_BaseCodeAnalysisTool):
    name = "functions_that_raise"
    description = """Find graph symbols with a parser-detected exception matching an exact name.

    Args:
        exception: Exception name to match, such as "ValueError".
        cursor: Opaque nextCursor from the same query; empty starts a new listing.
            Restart after workspace changes invalidate the cursor.
        limit: Maximum symbols per page, from 1 to 200; defaults to 100.

    Returns:
        Matching symbol items and locations, nextCursor, truncated and coverage.
    """

    async def __call__(
        self,
        exception: str,
        cursor: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.functions_that_raise(exception, cursor, limit)
            self._success(started_ns, result)
            return result.value
        except asyncio.CancelledError:
            self._cancelled(started_ns)
            raise
        except Exception as error:
            self._failure(started_ns, error)
            raise


def _metric(value: Mapping[str, Any], coverage: _Coverage, stats: _ScanStats) -> dict[str, Any]:
    return {
        "engine": "shallow",
        "count": len(value["items"]),
        "truncated": bool(value["truncated"]),
        "analyzed_files": coverage.analyzed_files,
        "analyzed_bytes": coverage.analyzed_bytes,
        "symbols": int(value["observedTotal"]),
        "binary_files": coverage.binary_files,
        "unsupported_source_files": coverage.unsupported_source_files,
        "oversized_files": coverage.oversized_files,
        "parse_errors": coverage.parse_errors,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "cache_invalidations": stats.cache_invalidations,
    }


def _graph_symbol_row(item: GraphSymbolProjection) -> dict[str, Any]:
    return {
        "symbolId": item.symbol_id,
        "name": item.name,
        "kind": item.kind,
        "path": item.path,
        "line": item.line,
        "endLine": item.end_line,
        "column": item.column,
    }


def _graph_relationship_row(item: GraphRelationshipProjection) -> dict[str, Any]:
    return {**_graph_symbol_row(item.symbol), "confidence": item.confidence}


def _graph_entrypoint_row(item: GraphEntrypointProjection) -> dict[str, Any]:
    result = {
        **_graph_symbol_row(item.symbol),
        "entrypointKind": item.entrypoint_kind,
        "trustLevel": item.trust_level,
        "assetValue": item.asset_value,
    }
    if item.description is not None:
        result["description"] = item.description
    return result


def _graph_complexity_row(item: GraphComplexityProjection) -> dict[str, Any]:
    return {**_graph_symbol_row(item.symbol), "complexity": item.complexity}


def _query_string(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.strip()
        or len(value) > MAX_QUERY_CHARS
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise CodeAnalysisError("code_analysis_input_invalid")
    return value


def _path(value: str) -> str:
    if not isinstance(value, str):
        raise CodeAnalysisError("code_analysis_input_invalid")
    try:
        return normalize_project_path(value, allow_root=True)
    except ProjectPathError:
        raise CodeAnalysisError("code_analysis_input_invalid") from None


def _language(value: str) -> Language | None:
    if not isinstance(value, str):
        raise CodeAnalysisError("code_analysis_input_invalid")
    if not value:
        return None
    try:
        return Language(value)
    except ValueError:
        raise CodeAnalysisError("code_analysis_input_invalid") from None


def _node_type(value: str) -> str:
    if not isinstance(value, str):
        raise CodeAnalysisError("code_analysis_input_invalid")
    if not value:
        return ""
    if len(value) > MAX_QUERY_CHARS or _NODE_TYPE_PATTERN.fullmatch(value) is None:
        raise CodeAnalysisError("code_analysis_input_invalid")
    return value


def _limit(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_PAGE_ITEMS:
        raise CodeAnalysisError("code_analysis_input_invalid")
    return value


def _path_limit(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 50:
        raise CodeAnalysisError("code_analysis_input_invalid")
    return value


def _path_depth(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 20:
        raise CodeAnalysisError("code_analysis_input_invalid")
    return value


def _complexity_threshold(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10_000:
        raise CodeAnalysisError("code_analysis_input_invalid")
    return value


def _symbol_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or len(value) > MAX_SYMBOL_ID_BYTES
    ):
        raise CodeAnalysisError("code_analysis_symbol_not_found")
    return value


def _snapshot_has_path(snapshot: WorkspaceSnapshot, path: str) -> bool:
    return (
        path in snapshot.directories
        or path in snapshot.binary_paths
        or any(item.path == path for item in snapshot.files)
    )


def _under_path(candidate: str, root: str) -> bool:
    return not root or candidate == root or candidate.startswith(root + "/")


def _bare_name(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("::", ".").replace("#", ".").rsplit(".", 1)[-1]


def _symbol_matches(extracted: str, query: str) -> bool:
    return _bare_name(extracted).casefold() == _bare_name(query).casefold()


def _symbol_sort_key(item: SymbolRecord) -> tuple[str, int, int, str, str]:
    return item.path, item.line, item.column, item.name, item.node_type


def _symbol_row(item: SymbolRecord) -> dict[str, Any]:
    return {
        "name": item.name,
        "path": item.path,
        "line": item.line,
        "endLine": item.end_line,
        "column": item.column,
        "nodeType": item.node_type,
        "language": item.language,
    }


def _definition_row(item: SymbolRecord, file: WorkspaceTextFile) -> dict[str, Any]:
    row = _symbol_row(item)
    preview = _preview(file.text, item.start_byte, item.end_byte)
    if preview:
        row["preview"] = preview
    return row


def _definition_rows(
    symbols: tuple[SymbolRecord, ...],
    files: Mapping[str, WorkspaceTextFile],
) -> list[dict[str, Any]]:
    return [_definition_row(item, files[item.path]) for item in symbols]


def _contains_casefold(text: str, needle: str) -> bool:
    return needle in text.casefold()


def _parse_symbols_text(
    parser: Parser,
    text: str,
    path: str,
    language: Language,
    limit: int,
) -> language_support.ParseResult:
    return language_support.parse_symbols(
        parser,
        text.encode("utf-8"),
        path,
        language,
        limit,
    )


def _preview(text: str, start_byte: int, end_byte: int) -> str:
    source = text.encode("utf-8")
    selected = source[start_byte:end_byte].decode("utf-8", errors="strict")
    selected = "\n".join(selected.splitlines()[:MAX_PREVIEW_LINES])
    encoded = selected.encode("utf-8")
    if len(encoded) <= MAX_PREVIEW_BYTES:
        return selected
    return _utf8_prefix(encoded, MAX_PREVIEW_BYTES)


def _utf8_prefix(value: bytes, maximum: int) -> str:
    return value[:maximum].decode("utf-8", errors="ignore")


def _query_digest(document: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(jcs.canonicalize(dict(document))).hexdigest()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _b64(decoded) != value:
        raise ValueError
    return decoded


async def _to_thread_cancellation_safe(function: Any, *arguments: Any) -> Any:
    """Do not let cancelled CPU work mutate allocation state after lock release."""

    task = asyncio.create_task(
        asyncio.to_thread(function, *arguments),
        name="code-analysis-cpu",
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Python cannot stop a thread which is already inside Tree-sitter. Keep
        # the session owner locked until it really returns; the allocation-wide
        # stop deadline will fence and terminate the Runtime if that cannot be
        # confirmed in time.
        with suppress(Exception):
            await task
        raise


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
