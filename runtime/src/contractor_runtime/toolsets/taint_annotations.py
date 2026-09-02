"""Atomic structured taint annotations over one project workspace."""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.observations import (
    WorkspaceToolObservation,
    annotation_tool_observation,
)
from contractor_runtime.projectfs.paths import ProjectPathError, normalize_project_path
from contractor_runtime.projectfs.storage import WorkspaceStorageError, WorkspaceWriter
from contractor_runtime.toolsets import code_analysis_languages as language_support
from contractor_runtime.toolsets.code_analysis import (
    SHALLOW_PINNED_DEPENDENCIES,
    dependency_versions_match,
)
from contractor_runtime.toolsets.code_analysis_languages import Language
from contractor_runtime.toolsets.run_artifacts import ToolMetrics
from contractor_runtime.toolsets.taint_annotation_languages import (
    AnnotationParseResult,
    AnnotationTarget,
    parse_annotation_targets,
)
from contractor_runtime.workspace import AllocationWorkspace

TAINT_ANNOTATIONS_REF = "taint-annotations@1"
EXPORTED_TOOLS = frozenset({"annotate_trace", "annotate_validate", "annotate_sink"})
PINNED_DEPENDENCIES = MappingProxyType(dict(SHALLOW_PINNED_DEPENDENCIES))

MAX_SOURCE_FILE_BYTES = 4 * 1024 * 1024
MAX_SYMBOL_CHARS = 256
MAX_TOKEN_CHARS = 128
MAX_LIST_ENTRIES = 32
MAX_ANNOTATION_BYTES = 4096
MAX_DEFINITION_LINE = 2_147_483_647

_ARG_STATES = frozenset({"tainted", "validated", "clean", "derived"})
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.$:/{}\[\]*?+\-<>]+\Z")
_ARGUMENT_PATTERN = re.compile(r"[A-Za-z0-9_.$\[\]*?+\-<>]+\Z")
_COMMENT_MARKERS = MappingProxyType(
    {
        Language.PYTHON: "#",
        Language.RUBY: "#",
        Language.BASH: "#",
        Language.ELIXIR: "#",
        Language.HASKELL: "--",
        Language.LUA: "--",
    }
)
_ERROR_CODES = frozenset(
    {
        "workspace_required",
        "taint_annotation_input_invalid",
        "taint_annotation_language_unsupported",
        "taint_annotation_target_not_found",
        "taint_annotation_target_ambiguous",
        "taint_annotation_conflict",
        "taint_annotation_workspace_changed",
        "taint_annotation_capacity_exceeded",
        "taint_annotation_cancelled",
        "taint_annotation_closing",
        "taint_annotation_unavailable",
    }
)

AnnotationKind = Literal["trace", "validate", "sink"]


class TaintAnnotationError(RuntimeError):
    """Stable model-facing failure without source, path, or annotation values."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        if code not in _ERROR_CODES:
            code = "taint_annotation_unavailable"
            retryable = True
        self.code = code
        self.retryable = retryable
        super().__init__(f"Taint annotation operation failed ({code})")


class _WorkspaceChanged(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _AnnotationRequest:
    path: str
    symbol: str
    kind: AnnotationKind
    body: str
    logical_key: str
    definition_line: int


@dataclass(frozen=True, slots=True)
class _MutationPlan:
    target: AnnotationTarget
    source: str
    line: str
    marker: str
    indent: str


class TaintAnnotationsToolsetFactory:
    ref = TAINT_ANNOTATIONS_REF
    exported_tools = EXPORTED_TOOLS
    infrastructure_channels = MappingProxyType({})
    requires_workspace = True
    workspace_access = "write"

    def __init__(self) -> None:
        self._available_tools = EXPORTED_TOOLS

    async def probe(self) -> frozenset[str]:
        self._available_tools = frozenset()
        if not dependency_versions_match(PINNED_DEPENDENCIES):
            return frozenset()
        if not await asyncio.to_thread(language_support.probe_all_parsers):
            return frozenset()
        self._available_tools = EXPORTED_TOOLS
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
        project_workspace: WorkspaceWriter | None = None,
    ) -> Mapping[str, Any]:
        del allocation_id, run_id, namespace, runtime_settings, workspace, adapter_handles
        unavailable = sorted(set(selected) - self._available_tools)
        if unavailable:
            raise ValueError(
                f"unavailable selected taint annotation tools: {', '.join(unavailable)}"
            )
        if project_workspace is None:
            raise TaintAnnotationError("workspace_required")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("taint-annotations@1 requires State.metrics")
        session = _TaintAnnotationSession(project_workspace)
        builders: dict[str, Callable[[], Any]] = {
            "annotate_trace": lambda: AnnotateTraceTool(session, metrics),
            "annotate_validate": lambda: AnnotateValidateTool(session, metrics),
            "annotate_sink": lambda: AnnotateSinkTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


class _TaintAnnotationSession:
    def __init__(self, writer: WorkspaceWriter) -> None:
        self._writer: WorkspaceWriter | None = writer
        self._lock = asyncio.Lock()
        self._closed = False
        self._closing = False

    async def close(self) -> None:
        self._closing = True
        async with self._lock:
            self._closed = True
            self._writer = None

    async def annotate(self, request: _AnnotationRequest) -> dict[str, Any]:
        async with self._lock:
            if self._closing or self._closed or self._writer is None:
                raise TaintAnnotationError("taint_annotation_closing")
            writer = self._writer
            try:
                source = await writer.read_text(request.path)
            except asyncio.CancelledError:
                raise
            except WorkspaceStorageError as error:
                raise _map_workspace_error(error) from None
            except Exception:
                raise TaintAnnotationError("taint_annotation_unavailable", retryable=True) from None

            try:
                encoded = source.encode("utf-8", errors="strict")
            except UnicodeError:
                raise TaintAnnotationError("taint_annotation_language_unsupported") from None
            if len(encoded) > MAX_SOURCE_FILE_BYTES:
                raise TaintAnnotationError("taint_annotation_capacity_exceeded")
            language = language_support.detect_language(request.path)
            if language is None:
                raise TaintAnnotationError("taint_annotation_language_unsupported")
            try:
                parsed = await _to_thread_cancellation_safe(
                    _parse_target_file,
                    encoded,
                    language,
                )
                plan = _plan_mutation(source, language, parsed, request)
            except asyncio.CancelledError:
                raise
            except TaintAnnotationError:
                raise
            except Exception:
                raise TaintAnnotationError("taint_annotation_unavailable", retryable=True) from None

            result: dict[str, Any] | None = None

            def transform(current: str) -> str:
                nonlocal result
                if current != plan.source:
                    raise _WorkspaceChanged
                updated, result = _apply_plan(current, plan, request)
                return updated

            try:
                await writer.update_text(request.path, transform)
            except asyncio.CancelledError:
                raise
            except _WorkspaceChanged:
                raise TaintAnnotationError(
                    "taint_annotation_workspace_changed", retryable=True
                ) from None
            except TaintAnnotationError:
                raise
            except WorkspaceStorageError as error:
                raise _map_workspace_error(error) from None
            except Exception:
                raise TaintAnnotationError("taint_annotation_unavailable", retryable=True) from None
            if result is None:
                raise TaintAnnotationError("taint_annotation_unavailable", retryable=True)
            return result


class _BaseAnnotationTool:
    # Keep these runtime class attributes unannotated. Google ADK asks
    # typing.get_type_hints() for a callable instance while building its JSON
    # Schema; inherited forward annotations are then evaluated without this
    # module's globals and fail before the first model call.
    name = ""
    kind = ""
    description = ""

    def __init__(self, session: _TaintAnnotationSession, metrics: ToolMetrics) -> None:
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        await self._session.close()

    def contractor_observation(
        self,
        tool_args: Mapping[str, Any],
        result: Any,
    ) -> WorkspaceToolObservation:
        return annotation_tool_observation(tool_args, result)

    def contractor_raw_argument_error(self, args: object) -> TaintAnnotationError | None:
        if _valid_raw_arguments(self.name, args):
            return None
        error = TaintAnnotationError("taint_annotation_input_invalid")
        self._record_failure(error, time.perf_counter_ns())
        return error

    async def _invoke(self, request_factory: Callable[[], _AnnotationRequest]) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            request = request_factory()
            result = await self._session.annotate(request)
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                result={"kind": self.kind, "changed": result["changed"]},
                duration_ms=_elapsed_ms(started),
            )
            return result
        except asyncio.CancelledError:
            self._record_failure(
                TaintAnnotationError("taint_annotation_cancelled", retryable=True),
                started,
            )
            raise
        except Exception as error:
            bounded = _normalize_error(error)
            self._record_failure(bounded, started)
            raise bounded from None

    def _record_failure(self, error: TaintAnnotationError, started: int) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments={},
            error=error,
            duration_ms=_elapsed_ms(started),
        )


class AnnotateTraceTool(_BaseAnnotationTool):
    name = "annotate_trace"
    kind = "trace"
    description = "Insert one canonical trace annotation above an exact function."

    async def __call__(
        self,
        path: str,
        symbol: str,
        target: str = "unknown",
        args: str = "",
        calls: str = "",
        definition_line: int = 0,
    ) -> dict[str, Any]:
        def build_request() -> _AnnotationRequest:
            normalized_path = _path(path)
            normalized_symbol = _symbol(symbol)
            normalized_target = _token(target)
            normalized_args = _args(args)
            normalized_calls = _calls(calls)
            body = f"target={normalized_target}"
            if normalized_args:
                body += f" args={normalized_args}"
            if normalized_calls:
                body += f" calls={normalized_calls}"
            return _request(
                normalized_path,
                normalized_symbol,
                "trace",
                body,
                normalized_target,
                definition_line,
            )

        return await self._invoke(build_request)


class AnnotateValidateTool(_BaseAnnotationTool):
    name = "annotate_validate"
    kind = "validate"
    description = "Insert one canonical validation annotation above an exact function."

    async def __call__(
        self,
        path: str,
        symbol: str,
        arg: str,
        kind: str,
        definition_line: int = 0,
    ) -> dict[str, Any]:
        def build_request() -> _AnnotationRequest:
            normalized_arg = _argument_token(arg)
            normalized_kind = _token(kind)
            return _request(
                _path(path),
                _symbol(symbol),
                "validate",
                f"arg={normalized_arg} kind={normalized_kind}",
                f"{normalized_arg}:{normalized_kind}",
                definition_line,
            )

        return await self._invoke(build_request)


class AnnotateSinkTool(_BaseAnnotationTool):
    name = "annotate_sink"
    kind = "sink"
    description = "Insert one canonical sink annotation above an exact function."

    async def __call__(
        self,
        path: str,
        symbol: str,
        kind: str,
        arg: str = "unknown",
        definition_line: int = 0,
    ) -> dict[str, Any]:
        def build_request() -> _AnnotationRequest:
            normalized_kind = _token(kind)
            normalized_arg = _argument_token(arg)
            return _request(
                _path(path),
                _symbol(symbol),
                "sink",
                f"kind={normalized_kind} arg={normalized_arg}",
                f"{normalized_kind}:{normalized_arg}",
                definition_line,
            )

        return await self._invoke(build_request)


def _request(
    path: str,
    symbol: str,
    kind: AnnotationKind,
    body: str,
    logical_key: str,
    definition_line: int,
) -> _AnnotationRequest:
    if (
        not isinstance(definition_line, int)
        or isinstance(definition_line, bool)
        or definition_line < 0
        or definition_line > MAX_DEFINITION_LINE
    ):
        raise TaintAnnotationError("taint_annotation_input_invalid")
    line = f"@{kind} {body}"
    if len(line.encode("utf-8")) > MAX_ANNOTATION_BYTES:
        raise TaintAnnotationError("taint_annotation_capacity_exceeded")
    return _AnnotationRequest(path, symbol, kind, body, logical_key, definition_line)


def _plan_mutation(
    source: str,
    language: Language,
    parsed: AnnotationParseResult,
    request: _AnnotationRequest,
) -> _MutationPlan:
    matches = [target for target in parsed.targets if target.name == request.symbol]
    if request.definition_line:
        matches = [target for target in matches if request.definition_line in target.selector_lines]
    if not matches:
        code = (
            "taint_annotation_unavailable"
            if parsed.parse_error
            else "taint_annotation_target_not_found"
        )
        raise TaintAnnotationError(code, retryable=parsed.parse_error)
    if len(matches) != 1:
        raise TaintAnnotationError("taint_annotation_target_ambiguous")
    target = matches[0]
    lines = source.splitlines(keepends=True)
    if target.insertion_line < 1 or target.insertion_line > len(lines):
        raise TaintAnnotationError("taint_annotation_unavailable", retryable=True)
    source_line = _without_newline(lines[target.insertion_line - 1])
    indent = source_line[: len(source_line) - len(source_line.lstrip(" \t"))]
    marker = _COMMENT_MARKERS.get(language, "//")
    line = f"{indent}{marker} @{request.kind} {request.body}"
    encoded_line = line.encode("utf-8")
    newline = _newline_style(source).encode("ascii")
    if (
        len(encoded_line) > MAX_ANNOTATION_BYTES
        or len(source.encode("utf-8")) + len(encoded_line) + len(newline) > MAX_SOURCE_FILE_BYTES
    ):
        raise TaintAnnotationError("taint_annotation_capacity_exceeded")
    return _MutationPlan(target, source, line, marker, indent)


def _apply_plan(
    source: str,
    plan: _MutationPlan,
    request: _AnnotationRequest,
) -> tuple[str, dict[str, Any]]:
    lines = source.splitlines(keepends=True)
    insertion_index = plan.target.insertion_line - 1
    block = _annotation_block(lines, insertion_index, plan.indent, plan.marker)
    exact = next((index for index, value in block if value == plan.line), None)
    if exact is not None:
        return source, _result(request, exact + 1, plan.target.definition_line, False)
    if request.kind == "trace":
        for _, value in block:
            if not value.startswith(f"{plan.indent}{plan.marker} @trace "):
                continue
            existing_target = _trace_target(value, plan.indent, plan.marker)
            if existing_target is None or existing_target == request.logical_key:
                raise TaintAnnotationError("taint_annotation_conflict")
    elif any(
        value.startswith(f"{plan.indent}{plan.marker} @{request.kind}")
        and not _canonical_existing(value, plan.indent, plan.marker, request.kind)
        for _, value in block
    ):
        raise TaintAnnotationError("taint_annotation_conflict")
    newline = _newline_style(source)
    lines.insert(insertion_index, plan.line + newline)
    return "".join(lines), _result(
        request,
        insertion_index + 1,
        plan.target.definition_line + 1,
        True,
    )


def _annotation_block(
    lines: Sequence[str],
    insertion_index: int,
    indent: str,
    marker: str,
) -> list[tuple[int, str]]:
    prefix = f"{indent}{marker} @"
    result: list[tuple[int, str]] = []
    index = insertion_index - 1
    while index >= 0:
        value = _without_newline(lines[index])
        if not value.startswith(prefix):
            break
        result.append((index, value))
        index -= 1
    result.reverse()
    return result


def _trace_target(value: str, indent: str, marker: str) -> str | None:
    prefix = f"{indent}{marker} @trace target="
    if not value.startswith(prefix):
        return None
    target = value[len(prefix) :].split(" ", 1)[0]
    try:
        return _token(target)
    except TaintAnnotationError:
        return None


def _canonical_existing(
    value: str,
    indent: str,
    marker: str,
    kind: AnnotationKind,
) -> bool:
    body = value.removeprefix(f"{indent}{marker} @{kind} ")
    try:
        if kind == "validate":
            match = re.fullmatch(r"arg=([^ ]+) kind=([^ ]+)", body)
            return bool(match and _argument_token(match.group(1)) and _token(match.group(2)))
        match = re.fullmatch(r"kind=([^ ]+) arg=([^ ]+)", body)
        return bool(match and _token(match.group(1)) and _argument_token(match.group(2)))
    except TaintAnnotationError:
        return False


def _result(
    request: _AnnotationRequest,
    annotation_line: int,
    definition_line: int,
    changed: bool,
) -> dict[str, Any]:
    return {
        "path": request.path,
        "symbol": request.symbol,
        "kind": request.kind,
        "annotationLine": annotation_line,
        "definitionLine": definition_line,
        "changed": changed,
    }


def _path(value: str) -> str:
    if not isinstance(value, str):
        raise TaintAnnotationError("taint_annotation_input_invalid")
    try:
        return normalize_project_path(value, allow_root=False)
    except ProjectPathError:
        raise TaintAnnotationError("taint_annotation_input_invalid") from None


def _symbol(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_SYMBOL_CHARS
        or unicodedata.normalize("NFC", value) != value
        or any(
            character.isspace() or unicodedata.category(character).startswith("C")
            for character in value
        )
        or any(character in ",=#@" for character in value)
    ):
        raise TaintAnnotationError("taint_annotation_input_invalid")
    return value


def _token(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TOKEN_CHARS
        or not value.isascii()
        or _TOKEN_PATTERN.fullmatch(value) is None
    ):
        raise TaintAnnotationError("taint_annotation_input_invalid")
    return value


def _argument_token(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TOKEN_CHARS
        or not value.isascii()
        or _ARGUMENT_PATTERN.fullmatch(value) is None
    ):
        raise TaintAnnotationError("taint_annotation_input_invalid")
    return value


def _args(value: str) -> str:
    if not isinstance(value, str):
        raise TaintAnnotationError("taint_annotation_input_invalid")
    if not value:
        return ""
    result: list[str] = []
    names: set[str] = set()
    chunks = value.split(",")
    if len(chunks) > MAX_LIST_ENTRIES:
        raise TaintAnnotationError("taint_annotation_capacity_exceeded")
    for chunk in chunks:
        parts = chunk.strip().split(":")
        if len(parts) != 2:
            raise TaintAnnotationError("taint_annotation_input_invalid")
        name = _argument_token(parts[0].strip())
        state = parts[1].strip()
        if state not in _ARG_STATES or name in names:
            raise TaintAnnotationError("taint_annotation_input_invalid")
        names.add(name)
        result.append(f"{name}:{state}")
    return ",".join(result)


def _calls(value: str) -> str:
    if not isinstance(value, str):
        raise TaintAnnotationError("taint_annotation_input_invalid")
    if not value:
        return ""
    chunks = value.split(",")
    if len(chunks) > MAX_LIST_ENTRIES:
        raise TaintAnnotationError("taint_annotation_capacity_exceeded")
    result: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        item = _token(chunk.strip())
        if item in seen:
            raise TaintAnnotationError("taint_annotation_input_invalid")
        seen.add(item)
        result.append(item)
    return ",".join(result)


_RAW_ARGUMENT_FIELDS = MappingProxyType(
    {
        "annotate_trace": (
            {"path": str, "symbol": str},
            {"target": str, "args": str, "calls": str, "definition_line": int},
        ),
        "annotate_validate": (
            {"path": str, "symbol": str, "arg": str, "kind": str},
            {"definition_line": int},
        ),
        "annotate_sink": (
            {"path": str, "symbol": str, "kind": str},
            {"arg": str, "definition_line": int},
        ),
    }
)


def _valid_raw_arguments(tool_name: str, args: object) -> bool:
    if type(args) is not dict or tool_name not in _RAW_ARGUMENT_FIELDS:
        return False
    required, optional = _RAW_ARGUMENT_FIELDS[tool_name]
    if not set(required).issubset(args) or set(args) - set(required) - set(optional):
        return False
    fields = dict(required)
    fields.update(optional)
    for name, value in args.items():
        expected = fields[name]
        if expected is int:
            if not isinstance(value, int) or isinstance(value, bool):
                return False
        elif type(value) is not expected:
            return False
    return True


def _map_workspace_error(error: WorkspaceStorageError) -> TaintAnnotationError:
    code = error.args[0] if error.args and isinstance(error.args[0], str) else ""
    if code == "binary_file_unsupported":
        return TaintAnnotationError("taint_annotation_language_unsupported")
    if code == "workspace_not_found":
        return TaintAnnotationError("taint_annotation_target_not_found")
    if code in {"workspace_limit_exceeded"}:
        return TaintAnnotationError("taint_annotation_capacity_exceeded")
    if code in {"workspace_path_invalid", "workspace_type_conflict"}:
        return TaintAnnotationError("taint_annotation_input_invalid")
    return TaintAnnotationError("taint_annotation_unavailable", retryable=True)


def _normalize_error(error: Exception) -> TaintAnnotationError:
    if isinstance(error, TaintAnnotationError):
        return error
    return TaintAnnotationError("taint_annotation_unavailable", retryable=True)


def _parse_target_file(source: bytes, language: Language) -> AnnotationParseResult:
    parser = language_support.load_parser(language)
    return parse_annotation_targets(parser, source, language)


def _newline_style(value: str) -> str:
    for index, character in enumerate(value):
        if character == "\n":
            return "\n"
        if character == "\r":
            return "\r\n" if index + 1 < len(value) and value[index + 1] == "\n" else "\r"
    return "\n"


def _without_newline(value: str) -> str:
    return value.removesuffix("\n").removesuffix("\r")


async def _to_thread_cancellation_safe(function: Any, *arguments: Any) -> Any:
    task = asyncio.create_task(
        asyncio.to_thread(function, *arguments),
        name="taint-annotation-cpu",
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
