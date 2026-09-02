"""Strict backend-independent text and tree editing tools."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.observations import WorkspaceToolObservation, edit_tool_observation
from contractor_runtime.projectfs.paths import ProjectPathError
from contractor_runtime.projectfs.storage import WorkspaceStorageError, WorkspaceWriter
from contractor_runtime.toolsets.filesystem import FilesystemToolError
from contractor_runtime.toolsets.run_artifacts import ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAX_EDIT_REPLACEMENTS = 1000
MAX_EDIT_INPUT_BYTES = 16 * 1024 * 1024


class EditFilesToolsetFactory:
    ref = "edit-files@1"
    exported_tools = frozenset(
        {
            "write_file",
            "append_file",
            "mkdir",
            "rm",
            "cp",
            "mv",
            "insert_line",
            "edit",
            "replace_range",
        }
    )
    infrastructure_channels = MappingProxyType({})
    requires_workspace = True
    workspace_access = "write"

    async def probe(self) -> frozenset[str]:
        return self.exported_tools

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
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        if project_workspace is None:
            raise FilesystemToolError("workspace_required")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("edit-files@1 requires State.metrics")
        session = _EditSession(project_workspace)
        builders: dict[str, Callable[[], Any]] = {
            "write_file": lambda: WriteFileTool(session, metrics),
            "append_file": lambda: AppendFileTool(session, metrics),
            "mkdir": lambda: MakeDirectoryTool(session, metrics),
            "rm": lambda: RemovePathTool(session, metrics),
            "cp": lambda: CopyPathTool(session, metrics),
            "mv": lambda: MovePathTool(session, metrics),
            "insert_line": lambda: InsertLineTool(session, metrics),
            "edit": lambda: EditTextTool(session, metrics),
            "replace_range": lambda: ReplaceRangeTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


class _EditSession:
    def __init__(self, writer: WorkspaceWriter) -> None:
        self._writer = writer
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    async def invoke(self, operation: Callable[[], Awaitable[None]]) -> dict[str, bool]:
        if self._closed:
            raise FilesystemToolError("workspace_not_found")
        try:
            await operation()
            return {"changed": True}
        except asyncio.CancelledError:
            raise
        except FilesystemToolError:
            raise
        except ProjectPathError:
            raise FilesystemToolError("workspace_path_invalid") from None
        except WorkspaceStorageError as error:
            code = error.args[0] if error.args else "workspace_unavailable"
            if code not in {
                "binary_file_unsupported",
                "workspace_limit_exceeded",
                "workspace_not_found",
                "workspace_path_invalid",
                "workspace_type_conflict",
                "workspace_unavailable",
            }:
                code = "workspace_unavailable"
            raise FilesystemToolError(code) from None
        except Exception:
            raise FilesystemToolError("workspace_unavailable") from None


class _BaseEditTool:
    name: str
    description: str

    def __init__(self, session: _EditSession, metrics: ToolMetrics) -> None:
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
    ) -> WorkspaceToolObservation | None:
        return edit_tool_observation(self.name, tool_args, result)

    async def _invoke(self, operation: Callable[[], Awaitable[None]]) -> dict[str, bool]:
        started = time.perf_counter_ns()
        try:
            result = await self._session.invoke(operation)
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                result=result,
                duration_ms=_elapsed_ms(started),
            )
            return result
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                error=error,
                duration_ms=_elapsed_ms(started),
            )
            raise


class WriteFileTool(_BaseEditTool):
    name = "write_file"
    description = "Atomically create or replace one UTF-8 project file."

    async def __call__(self, path: str, content: str) -> dict[str, bool]:
        return await self._invoke(lambda: self._session._writer.write_text(path, content))


class AppendFileTool(_BaseEditTool):
    name = "append_file"
    description = "Append UTF-8 text while preserving the file newline style."

    async def __call__(self, path: str, content: str) -> dict[str, bool]:
        def transform(current: str) -> str:
            _require_editable_text(current)
            style = _newline_style(current)
            addition = _normalize_newlines(content, style)
            separator = style if current and addition and not _ends_newline(current) else ""
            return current + separator + addition

        return await self._invoke(lambda: self._session._writer.update_text(path, transform))


class MakeDirectoryTool(_BaseEditTool):
    name = "mkdir"
    description = "Create one project directory, optionally including absent parents."

    async def __call__(self, path: str, parents: bool = False) -> dict[str, bool]:
        _require_bool(parents)
        return await self._invoke(
            lambda: self._session._writer.make_directory(path, parents=parents)
        )


class RemovePathTool(_BaseEditTool):
    name = "rm"
    description = "Remove one text file or an explicitly recursive project directory."

    async def __call__(self, path: str, recursive: bool = False) -> dict[str, bool]:
        _require_bool(recursive)
        return await self._invoke(
            lambda: self._session._writer.delete_path(path, recursive=recursive)
        )


class CopyPathTool(_BaseEditTool):
    name = "cp"
    description = "Copy one text file or an explicitly recursive project directory."

    async def __call__(
        self, source: str, destination: str, recursive: bool = False
    ) -> dict[str, bool]:
        _require_bool(recursive)
        return await self._invoke(
            lambda: self._session._writer.copy_path(source, destination, recursive=recursive)
        )


class MovePathTool(_BaseEditTool):
    name = "mv"
    description = "Move one managed project path without crossing the workspace root."

    async def __call__(self, source: str, destination: str) -> dict[str, bool]:
        return await self._invoke(lambda: self._session._writer.move_path(source, destination))


class InsertLineTool(_BaseEditTool):
    name = "insert_line"
    description = "Insert UTF-8 content at one 1-based line boundary."

    async def __call__(self, path: str, line: int, content: str) -> dict[str, bool]:
        _require_positive_line(line)

        def transform(current: str) -> str:
            _require_editable_text(current)
            lines = current.splitlines(keepends=True)
            if line > len(lines) + 1:
                raise FilesystemToolError("workspace_line_invalid")
            style = _newline_style(current)
            addition = _normalize_newlines(content, style)
            if line <= len(lines) and addition and not _ends_newline(addition):
                addition += style
            lines.insert(line - 1, addition)
            return "".join(lines)

        return await self._invoke(lambda: self._session._writer.update_text(path, transform))


class EditTextTool(_BaseEditTool):
    name = "edit"
    description = "Replace exactly one text match unless replace_all is explicit."

    async def __call__(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> dict[str, bool]:
        _require_bool(replace_all)
        if not isinstance(old, str) or old == "":
            raise FilesystemToolError("workspace_edit_match_invalid")

        def transform(current: str) -> str:
            _require_editable_text(current)
            style = _newline_style(current)
            needle = _normalize_newlines(old, style)
            replacement = _normalize_newlines(new, style)
            count = current.count(needle)
            if count == 0 or (not replace_all and count != 1):
                raise FilesystemToolError("workspace_edit_match_invalid")
            if count > MAX_EDIT_REPLACEMENTS:
                raise FilesystemToolError("workspace_limit_exceeded")
            return current.replace(needle, replacement, -1 if replace_all else 1)

        return await self._invoke(lambda: self._session._writer.update_text(path, transform))


class ReplaceRangeTool(_BaseEditTool):
    name = "replace_range"
    description = "Replace one inclusive 1-based line range with UTF-8 content."

    async def __call__(
        self,
        path: str,
        start_line: int,
        end_line: int,
        content: str,
    ) -> dict[str, bool]:
        _require_positive_line(start_line)
        _require_positive_line(end_line)
        if end_line < start_line:
            raise FilesystemToolError("workspace_line_invalid")

        def transform(current: str) -> str:
            _require_editable_text(current)
            lines = current.splitlines(keepends=True)
            if start_line > len(lines) or end_line > len(lines):
                raise FilesystemToolError("workspace_line_invalid")
            style = _newline_style(current)
            replacement = _normalize_newlines(content, style)
            if end_line < len(lines) and replacement and not _ends_newline(replacement):
                replacement += style
            lines[start_line - 1 : end_line] = [replacement] if replacement else []
            return "".join(lines)

        return await self._invoke(lambda: self._session._writer.update_text(path, transform))


def _newline_style(text: str) -> str:
    for index, character in enumerate(text):
        if character == "\n":
            return "\n"
        if character == "\r":
            return "\r\n" if index + 1 < len(text) and text[index + 1] == "\n" else "\r"
    return "\n"


def _normalize_newlines(value: str, newline: str) -> str:
    if not isinstance(value, str):
        raise FilesystemToolError("binary_file_unsupported")
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)


def _ends_newline(value: str) -> bool:
    return value.endswith(("\n", "\r"))


def _require_editable_text(value: str) -> None:
    if len(value) > MAX_EDIT_INPUT_BYTES:
        raise FilesystemToolError("workspace_limit_exceeded")
    try:
        if len(value.encode("utf-8")) > MAX_EDIT_INPUT_BYTES:
            raise FilesystemToolError("workspace_limit_exceeded")
    except UnicodeError:
        raise FilesystemToolError("binary_file_unsupported") from None


def _require_bool(value: bool) -> None:
    if not isinstance(value, bool):
        raise FilesystemToolError("workspace_operation_invalid")


def _require_positive_line(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FilesystemToolError("workspace_line_invalid")


def _elapsed_ms(started: int) -> int:
    return max(0, (time.perf_counter_ns() - started) // 1_000_000)
