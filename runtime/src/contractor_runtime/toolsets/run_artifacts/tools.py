"""Selected model-visible tools backed by one allocation-bound ArtifactClient."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactTransportError,
)
from contractor_runtime.contracts import (
    ARTIFACT_LIST_CURSOR_PATTERN,
    ARTIFACT_NAME_PATTERN,
    ArtifactRef,
    RuntimeSettings,
)
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_metrics import ToolMetrics
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.shared_config import artifact_list_max_items, artifact_payload_max_bytes
from contractor_runtime.toolsets.common.artifact_media import is_binary_media_type
from contractor_runtime.toolsets.common.artifact_read_cache import (
    ExactArtifactReadCache,
    LineIndex,
    allocation_artifact_read_cache,
)
from contractor_runtime.toolsets.common.artifact_visibility import (
    is_model_hidden_binding,
    require_model_visible_binding,
    require_model_visible_namespace,
)
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _unconfigured_client,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.common.tool_base import ArtifactTool
from contractor_runtime.workspace import AllocationWorkspace

MAX_TEXT_READ_LINES = 400
DEFAULT_TEXT_READ_LINES = 200
MAX_VISIBLE_TEXT_BYTES = 128 * 1024
# Refs per list_artifacts page: a page stays a few thousand tokens for small
# models; the fetched private API ceiling bounds this preferred size.
LIST_PAGE_SIZE = 100


class RunArtifactsToolsetFactory:
    ref = "run-artifacts@1"
    exported_tools = frozenset({"list_artifacts", "read_artifact", "write_artifact"})
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

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
        resources: ToolResources,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del adapter_handles
        require_selected_tools(selected, self.exported_tools)
        metrics = require_metrics(resources, "run-artifacts@1")
        client = self._client_factory(allocation_id, runtime_settings)
        secrets = runtime_secrets(runtime_settings)
        cache = allocation_artifact_read_cache(resources) if "read_artifact" in selected else None
        constructors = {
            "list_artifacts": ListArtifactsTool,
            "read_artifact": partial(ReadArtifactTool, cache=cache),
            "write_artifact": WriteArtifactTool,
        }
        return {name: constructors[name](client, metrics, secrets) for name in selected}


class ListArtifactsTool(ArtifactTool):
    name = "list_artifacts"
    description = f"""List current artifacts visible in this Workflow Run, one page at a time.

    Artifacts are ordered by namespace and name. Listed references carry no
    revision; read_artifact without a revision reads the current one. When
    nextCursor is not null, call again with the same namespace and cursor set
    to nextCursor for the next page.

    Args:
        namespace: Namespace to list; omit to list every visible namespace.
        cursor: nextCursor returned by the previous page; omit for the first page.

    Returns:
        Up to {LIST_PAGE_SIZE} artifact references with namespace and name, and
        nextCursor, which is null on the last page.
    """

    async def __call__(
        self, namespace: str | None = None, cursor: str | None = None
    ) -> dict[str, Any]:
        arguments = {"namespace": namespace, "cursor": cursor}
        with self._recorded(arguments) as call:
            require_model_visible_namespace(namespace)
            if namespace is not None and not _matches(ARTIFACT_NAME_PATTERN, namespace):
                raise ToolInputError("namespace must be an artifact namespace")
            if cursor is not None and not _matches(ARTIFACT_LIST_CURSOR_PATTERN, cursor):
                raise ToolInputError("cursor must be the nextCursor of a previous page")
            page = await self._client.list_artifact_page(
                namespace, limit=min(LIST_PAGE_SIZE, artifact_list_max_items()), cursor=cursor
            )
            # The Server excludes hidden bindings before its limit; one here is
            # a contract violation, not something to filter into a short page.
            if any(is_model_hidden_binding(ref.namespace, ref.name) for ref in page.artifacts):
                raise ArtifactTransportError("Artifact API listed a binding hidden from models")
            result = {
                "artifacts": [
                    ref.model_dump(by_alias=True, exclude_none=True) for ref in page.artifacts
                ],
                "nextCursor": page.next_cursor,
            }
            call.succeed({"count": len(page.artifacts), "hasMore": page.next_cursor is not None})
            return result


class ReadArtifactTool(ArtifactTool):
    name = "read_artifact"
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
        has been read to its end). Non-UTF-8 content is rejected, a binary
        media type before the artifact is downloaded.
    """

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        *,
        cache: ExactArtifactReadCache | None = None,
    ) -> None:
        super().__init__(client, metrics, secrets)
        self._cache = cache or ExactArtifactReadCache()

    async def close(self) -> None:
        await self._cache.clear()
        await super().close()

    async def __call__(
        self,
        namespace: str,
        name: str,
        revision: str | None = None,
        start_line: int = 1,
        max_lines: int = DEFAULT_TEXT_READ_LINES,
        line_offset: int = 0,
    ) -> dict[str, Any]:
        arguments = {
            "namespace": namespace,
            "name": name,
            "revision": revision,
            "start_line": start_line,
            "max_lines": max_lines,
            "line_offset": line_offset,
        }
        with self._recorded(arguments) as call:
            require_model_visible_binding(namespace, name)
            _validate_line_window(start_line, max_lines)
            if isinstance(line_offset, bool) or not isinstance(line_offset, int) or line_offset < 0:
                raise ToolInputError("line_offset must be a non-negative integer")
            read = await self._cache.read(
                self._client,
                ArtifactRef(
                    namespace=namespace,
                    name=name,
                    **({"revision": revision} if revision is not None else {}),
                ),
                accept=_refuse_binary_media_type,
                lines=True,
            )
            value, lines = read.value, read.lines
            if lines is None:
                raise ToolInputError("artifact is not valid UTF-8")
            # Same boundaries as read_file and grep, so line numbers agree.
            total_lines = lines.total_lines
            if total_lines > 0 and start_line > total_lines:
                raise ToolInputError("start_line exceeds artifact line count")
            window = _bounded_lines(
                value.data,
                lines,
                start_line=start_line,
                max_lines=max_lines,
                line_offset=line_offset,
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
            call.succeed(
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
                }
            )
            return result


def _refuse_binary_media_type(media_type: str) -> None:
    if is_binary_media_type(media_type):
        raise ToolInputError(f"artifact is binary ({media_type}) and is not valid UTF-8")


class WriteArtifactTool(ArtifactTool):
    name = "write_artifact"
    description = """Create or update an artifact in this Workflow Run from base64 bytes.

    Updates require the expected current revision to prevent overwriting a
    concurrent write.

    Args:
        namespace: Destination artifact namespace.
        name: Destination artifact binding name.
        media_type: MIME type describing the decoded bytes.
        data_base64: Canonical base64 payload within the configured decoded artifact limit.
        expected_revision: Current destination revision for an update; omit only
            to create a new binding.

    Returns:
        Saved artifact metadata including its exact revision, mediaType and size.
    """

    async def __call__(
        self,
        namespace: str,
        name: str,
        media_type: str,
        data_base64: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "namespace": namespace,
            "name": name,
            "media_type": media_type,
            "data_base64": data_base64,
            "expected_revision": expected_revision,
        }
        with self._recorded(arguments) as call:
            require_model_visible_binding(namespace, name)
            if not isinstance(data_base64, str):
                raise ToolInputError("data_base64 must be a canonical base64 string")
            if len(data_base64) > ((artifact_payload_max_bytes() + 2) // 3) * 4:
                raise ToolInputError(
                    "base64 artifact payload exceeds the configured Artifact limit"
                )
            try:
                data = base64.b64decode(data_base64, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ToolInputError("data_base64 is not valid canonical base64") from error
            if base64.b64encode(data).decode("ascii") != data_base64:
                raise ToolInputError("data_base64 is not valid canonical base64")
            result = await self._client.write_artifact(
                ArtifactRef(namespace=namespace, name=name),
                data=data,
                media_type=media_type,
                expected_revision=expected_revision,
            )
            serialized = result.model_dump(by_alias=True)
            call.succeed(
                {
                    "artifact": serialized["artifact"],
                    "mediaType": result.media_type,
                    "size": result.size,
                }
            )
            return serialized


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
    data: bytes, lines: LineIndex, *, start_line: int, max_lines: int, line_offset: int = 0
) -> _LineWindow:
    # Works on the UTF-8 bytes of a validated payload and decodes only the
    # returned window, so a page costs its window, not the whole artifact.
    if lines.total_lines == 0:
        if line_offset:
            raise ToolInputError("line_offset exceeds the line length")
        return _LineWindow("", 0, False, 1, 0)
    line_start, line_end = lines.line_span(start_line)
    first = line_start + line_offset
    if line_offset and first >= line_end:
        raise ToolInputError("line_offset exceeds the line length")
    if line_offset and (data[first] & 0xC0) == 0x80:
        raise ToolInputError("line_offset must start a UTF-8 character")
    limit = first + MAX_VISIBLE_TEXT_BYTES
    if line_end > limit:
        # Cut on a UTF-8 boundary so the continuation offset starts a character.
        cut = limit
        while cut > first and (data[cut] & 0xC0) == 0x80:
            cut -= 1
        return _LineWindow(lines.text(first, cut), start_line, True, start_line, cut - line_start)
    end, end_line = lines.extend_whole_lines(
        line_end, limit=limit, line=start_line, count=max_lines - 1
    )
    return _LineWindow(lines.text(first, end), end_line, False, end_line + 1, 0)


def _matches(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None
