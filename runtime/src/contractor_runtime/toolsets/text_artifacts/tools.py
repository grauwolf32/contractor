"""Bounded UTF-8 tools over one allocation-bound RunScope ArtifactClient."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.common.artifact_visibility import (
    artifact_observation_cursor,
    clear_artifact_observations,
    model_visible_exact_refs,
    model_visible_observations_since,
    require_model_visible_binding,
)
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _reject_unconfigured_client,
    gateway_secrets,
)
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.common.lines import split_lines
from contractor_runtime.toolsets.common.metrics import ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAX_TEXT_WRITE_BYTES = 1024 * 1024
MAX_TEXT_READ_LINES = 400
DEFAULT_TEXT_READ_LINES = 200
MAX_VISIBLE_TEXT_BYTES = 128 * 1024


class TextArtifactsToolsetFactory:
    """Construct only the explicitly selected ``text-artifacts@1`` tools."""

    ref = "text-artifacts@1"
    exported_tools = frozenset({"read_text_artifact", "write_text_artifact"})
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _reject_unconfigured_client

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
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, adapter_handles
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("text-artifacts@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        secrets = gateway_secrets(runtime_settings)
        builders: dict[str, Callable[[], Any]] = {
            "read_text_artifact": lambda: ReadTextArtifactTool(client, metrics, secrets),
            "write_text_artifact": lambda: WriteTextArtifactTool(
                client, metrics, secrets, namespace
            ),
        }
        return {name: builders[name]() for name in selected}


class _BaseTextTool:
    name: str
    description: str

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return model_visible_exact_refs(getattr(self._client, "known_exact_refs", ()))

    @property
    def artifact_observation_cursor(self) -> int:
        return artifact_observation_cursor(self._client)

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        return model_visible_observations_since(self._client, cursor)

    def clear_artifact_observations(self) -> None:
        clear_artifact_observations(self._client)

    async def close(self) -> None:
        self._secrets = ()

    def _success(
        self, arguments: Mapping[str, Any], result: Mapping[str, Any], started_ns: int
    ) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments=arguments,
            result=result,
            secrets=self._secrets,
            duration_ms=_elapsed_ms(started_ns),
        )

    def _failure(self, arguments: Mapping[str, Any], error: Exception, started_ns: int) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments=arguments,
            error=error,
            secrets=self._secrets,
            duration_ms=_elapsed_ms(started_ns),
        )


class ReadTextArtifactTool(_BaseTextTool):
    name = "read_text_artifact"
    description = """Read a bounded UTF-8 line window from an artifact in this Workflow Run.

    Output is limited to 128 KiB; inspect truncated and partialLine before treating
    the window as complete. Page with nextStartLine and nextLineOffset: a line
    longer than the output limit is returned in parts, continued by passing
    line_offset.

    Args:
        namespace: Artifact namespace.
        name: Exact artifact binding name.
        revision: Exact revision to read; omit for the current revision. Use the
            exact revision supplied in Stage context when available.
        start_line: First line to read, 1-based and inclusive; defaults to 1.
        max_lines: Maximum lines to return, from 1 to 400; defaults to 200.
        line_offset: UTF-8 byte offset into start_line to continue a partial
            line; use the returned nextLineOffset. Defaults to 0.

    Returns:
        Exact artifact metadata, text, startLine, endLine, totalLines, truncated,
        partialLine, and nextStartLine/nextLineOffset (null when the artifact
        has been read to its end). Non-UTF-8 content is rejected.
    """

    async def __call__(
        self,
        namespace: str,
        name: str,
        revision: str | None = None,
        start_line: int = 1,
        max_lines: int = DEFAULT_TEXT_READ_LINES,
        line_offset: int = 0,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        arguments = {
            "namespace": namespace,
            "name": name,
            "revision": revision,
            "start_line": start_line,
            "max_lines": max_lines,
            "line_offset": line_offset,
        }
        try:
            require_model_visible_binding(namespace, name)
            _validate_line_window(start_line, max_lines)
            if isinstance(line_offset, bool) or not isinstance(line_offset, int) or line_offset < 0:
                raise ToolInputError("line_offset must be a non-negative integer")
            value = await self._client.read_artifact(
                ArtifactRef(namespace=namespace, name=name, revision=revision)
            )
            try:
                content = value.data.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ToolInputError("artifact is not valid UTF-8") from error
            # Same boundaries as read_file and grep, so line numbers agree.
            lines = split_lines(content, keepends=True)
            total_lines = len(lines)
            if total_lines > 0 and start_line > total_lines:
                raise ToolInputError("start_line exceeds artifact line count")
            window = _bounded_lines(
                lines, start_line=start_line, max_lines=max_lines, line_offset=line_offset
            )
            selected, end_line, partial_line = window.text, window.end_line, window.partial_line
            truncated = partial_line or end_line < total_lines
            result = {
                "artifact": value.artifact.model_dump(by_alias=True),
                "mediaType": value.media_type,
                "size": len(value.data),
                "totalLines": total_lines,
                "startLine": start_line,
                "endLine": end_line,
                "text": selected,
                "truncated": truncated,
                "partialLine": partial_line,
                "nextStartLine": window.next_start_line if truncated else None,
                "nextLineOffset": window.next_line_offset if truncated else None,
            }
            self._success(
                arguments,
                {
                    "artifact": result["artifact"],
                    "mediaType": value.media_type,
                    "size": len(value.data),
                    "totalLines": total_lines,
                    "startLine": start_line,
                    "endLine": end_line,
                    "visibleUtf8Bytes": len(selected.encode("utf-8")),
                    "truncated": truncated,
                    "partialLine": partial_line,
                },
                started_ns,
            )
            return result
        except Exception as error:
            self._failure(arguments, error, started_ns)
            raise


class WriteTextArtifactTool(_BaseTextTool):
    name = "write_text_artifact"
    description = """Create or update a UTF-8 artifact in the Worker's fixed namespace.

    Updates require the expected current revision to prevent overwriting a
    concurrent write.

    Args:
        name: Destination artifact binding name within the Worker's namespace.
        text: Complete UTF-8 text, limited to 1 MiB after encoding.
        media_type: MIME type describing the text, such as "text/markdown".
        expected_revision: Current destination revision for an update; omit only
            to create a new binding.

    Returns:
        Saved artifact metadata including its exact revision, mediaType and size.
    """

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        namespace: str,
    ) -> None:
        super().__init__(client, metrics, secrets)
        self._namespace = namespace

    async def __call__(
        self,
        name: str,
        text: str,
        media_type: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        arguments = {
            "name": name,
            "media_type": media_type,
            "expected_revision": expected_revision,
            "content": text,
        }
        try:
            if not isinstance(text, str):
                raise ToolInputError("text must be a string")
            try:
                data = text.encode("utf-8")
            except UnicodeError:
                raise ToolInputError("text must be valid UTF-8") from None
            arguments["utf8_size"] = len(data)
            require_model_visible_binding(self._namespace, name)
            if len(data) > MAX_TEXT_WRITE_BYTES:
                raise ToolInputError("UTF-8 artifact text exceeds the 1 MiB tool limit")
            result = await self._client.write_artifact(
                ArtifactRef(namespace=self._namespace, name=name),
                data=data,
                media_type=media_type,
                expected_revision=expected_revision,
            )
            serialized = result.model_dump(by_alias=True)
            self._success(
                arguments,
                {
                    "artifact": serialized["artifact"],
                    "mediaType": result.media_type,
                    "size": result.size,
                },
                started_ns,
            )
            return serialized
        except Exception as error:
            self._failure(arguments, error, started_ns)
            raise


def _validate_line_window(start_line: int, max_lines: int) -> None:
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ToolInputError("start_line must be a positive integer")
    if (
        isinstance(max_lines, bool)
        or not isinstance(max_lines, int)
        or max_lines < 1
        or max_lines > MAX_TEXT_READ_LINES
    ):
        raise ToolInputError(f"max_lines must be between 1 and {MAX_TEXT_READ_LINES}")


@dataclass(frozen=True)
class _LineWindow:
    text: str
    end_line: int
    partial_line: bool
    next_start_line: int
    next_line_offset: int


def _bounded_lines(
    lines: list[str], *, start_line: int, max_lines: int, line_offset: int = 0
) -> _LineWindow:
    if not lines:
        if line_offset:
            raise ToolInputError("line_offset exceeds the line length")
        return _LineWindow("", 0, False, 1, 0)
    first = lines[start_line - 1].encode("utf-8")
    if line_offset and line_offset >= len(first):
        raise ToolInputError("line_offset exceeds the line length")
    if line_offset and (first[line_offset] & 0xC0) == 0x80:
        raise ToolInputError("line_offset must start a UTF-8 character")
    window = [
        first[line_offset:],
        *(line.encode("utf-8") for line in lines[start_line : start_line - 1 + max_lines]),
    ]
    result: list[bytes] = []
    size = 0
    for encoded in window:
        remaining = MAX_VISIBLE_TEXT_BYTES - size
        if len(encoded) <= remaining:
            result.append(encoded)
            size += len(encoded)
            continue
        if not result:
            # Cut on a UTF-8 boundary so the continuation offset starts a character.
            cut = remaining
            while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
                cut -= 1
            consumed = line_offset + cut
            return _LineWindow(
                encoded[:cut].decode("utf-8"), start_line, True, start_line, consumed
            )
        break
    end_line = start_line + len(result) - 1
    return _LineWindow(b"".join(result).decode("utf-8"), end_line, False, end_line + 1, 0)


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
