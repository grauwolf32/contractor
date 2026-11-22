"""Selected model-visible tools backed by one allocation-bound ArtifactClient."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

from google.genai import types

from contractor_runtime import threads
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import (
    API_VERSION,
    ARTIFACT_LIST_CURSOR_PATTERN,
    ARTIFACT_NAME_PATTERN,
    ArtifactReadResult,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_metrics import ToolMetrics
from contractor_runtime.llm.image_parts import (
    ARTIFACT_IMAGE_PART,
    IMAGE_MEDIA_TYPES,
    MAX_CONTEXT_IMAGE_BYTES,
    MAX_CONTEXT_IMAGES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
)
from contractor_runtime.projectfs.storage import WorkspaceReader, WorkspaceStorageError
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.shared_config import artifact_list_max_items, artifact_payload_max_bytes
from contractor_runtime.toolsets.common.artifact_images import (
    ImageInputBudget,
    ImageReservation,
    validate_image,
)
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
from contractor_runtime.toolsets.common.artifact_write import (
    definite_write_rejection,
    write_artifact_exact,
)
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _unconfigured_client,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.repeats import RepeatRule
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.common.tool_base import ArtifactTool
from contractor_runtime.toolsets.common.versions import ModelArtifacts
from contractor_runtime.workspace import AllocationWorkspace

MAX_CONSECUTIVE_EXACT_READS = 2
MAX_TEXT_READ_LINES = 400
DEFAULT_TEXT_READ_LINES = 200
MAX_VISIBLE_TEXT_BYTES = 128 * 1024
# Refs per list_artifacts page: a page stays a few thousand tokens for small
# models; the fetched private API ceiling bounds this preferred size.
LIST_PAGE_SIZE = 100


class RunArtifactsToolsetFactory:
    ref = "run-artifacts@1"
    workspace_access = "read"
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
        client = ModelArtifacts(
            self._client_factory(allocation_id, runtime_settings), resources.selected_versions
        )
        secrets = runtime_secrets(runtime_settings)
        cache = allocation_artifact_read_cache(resources) if "read_artifact" in selected else None
        constructors = {
            "list_artifacts": ListArtifactsTool,
            "read_artifact": partial(ReadArtifactTool, cache=cache, images=resources.image_input),
            "write_artifact": partial(WriteArtifactTool, source=project_workspace),
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


class _ImageUnavailable(Exception):
    def __init__(self, metadata: ArtifactReadResult, reason: str) -> None:
        super().__init__("Image delivery is unavailable")
        self.metadata = metadata
        self.reason = reason


class ReadArtifactTool(ArtifactTool):
    name = "read_artifact"
    description = f"""Read text or a supported image from an artifact in this Workflow Run.

    PNG/JPEG images are delivered natively when the configured model route confirms
    image support. Images are read whole; leave the line arguments at their defaults.
    Images must have one frame, at most {MAX_IMAGE_BYTES // (1024 * 1024)} MiB,
    {MAX_IMAGE_PIXELS:,} pixels and {MAX_IMAGE_DIMENSION} pixels per side. The current
    model context holds at most {MAX_CONTEXT_IMAGES} images /
    {MAX_CONTEXT_IMAGE_BYTES // (1024 * 1024)} MiB.
    Unavailable image delivery returns metadata and a reason, without image bytes.
    Image attachments are transient: after history recovery or compaction, read the
    exact revision again when the image itself is needed.

    At most {MAX_CONSECUTIVE_EXACT_READS} consecutive successful reads of the same explicit
    revision/window are allowed; further identical calls are refused. Another tool/window
    or a committed context checkpoint starts a new series. Failed reads may be retried.

    Text output is limited to 128 KiB; inspect truncated and partialLine before treating
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
        has been read to its end); or image metadata and a native image part.
        An unavailable image result contains authorized metadata and its reason.
    """

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        *,
        cache: ExactArtifactReadCache | None = None,
        images: ImageInputBudget | None = None,
    ) -> None:
        super().__init__(client, metrics, secrets)
        self._cache = cache or ExactArtifactReadCache()
        self._images = images or ImageInputBudget()

    def contractor_repeat_rule(self, arguments: Mapping[str, Any]) -> RepeatRule | None:
        revision = arguments.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            return None
        namespace, name = arguments.get("namespace"), arguments.get("name")
        start = arguments.get("start_line", 1)
        lines = arguments.get("max_lines", DEFAULT_TEXT_READ_LINES)
        offset = arguments.get("line_offset", 0)
        if (
            not _matches(ARTIFACT_NAME_PATTERN, namespace)
            or not _matches(ARTIFACT_NAME_PATTERN, name)
            or any(type(value) is not int for value in (start, lines, offset))
            or start < 1
            or not 1 <= lines <= MAX_TEXT_READ_LINES
            or offset < 0
        ):
            return None
        return RepeatRule(
            (
                namespace,
                name,
                hashlib.sha256(revision.encode()).hexdigest(),
                start,
                lines,
                offset,
                self._images.generation,
            ),
            MAX_CONSECUTIVE_EXACT_READS,
            "This exact artifact window was already read consecutively. Reuse its result or "
            "continue with nextStartLine/nextLineOffset. "
            "After a context checkpoint, reread if needed.",
        )

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
            reservation: ImageReservation | None = None
            admitted: ArtifactReadResult | None = None

            def accept(metadata: ArtifactReadResult) -> int | None:
                nonlocal reservation, admitted
                admitted = metadata
                media_type = metadata.media_type
                if media_type in IMAGE_MEDIA_TYPES:
                    if start_line != 1 or max_lines != DEFAULT_TEXT_READ_LINES or line_offset != 0:
                        raise ToolInputError("Images require the default line arguments")
                    if self._images.supports_images is not True:
                        reason = (
                            "text_only_route"
                            if self._images.supports_images is False
                            else "image_support_unconfirmed"
                        )
                        raise _ImageUnavailable(metadata, reason)
                    if not 0 < metadata.size <= MAX_IMAGE_BYTES:
                        raise ToolInputError("Image exceeds the supported encoded byte bound")
                    reservation = self._images.reserve(metadata.size)
                    return min(MAX_IMAGE_BYTES, artifact_payload_max_bytes())
                if media_type.startswith("image/") and media_type != "image/svg+xml":
                    raise _ImageUnavailable(metadata, "unsupported_image_format")
                _refuse_binary_media_type(media_type)
                return None

            try:
                read = await self._cache.read(
                    self._client,
                    ArtifactRef(
                        namespace=namespace,
                        name=name,
                        **({"revision": revision} if revision is not None else {}),
                    ),
                    accept=accept,
                    lines=True,
                )
                if reservation is not None:
                    value = read.value
                    if (
                        admitted is None
                        or value.artifact != admitted.artifact
                        or value.media_type != admitted.media_type
                        or len(value.data) != admitted.size
                    ):
                        raise ArtifactTransportError("Image payload disagrees with exact metadata")
                    async with self._images.decode_lock:
                        width, height = await threads.to_thread_until_done(
                            validate_image,
                            value.data,
                            value.media_type,
                            name="artifact-image-decode",
                        )
                    result = {
                        "artifact": value.artifact.model_dump(by_alias=True),
                        "mediaType": value.media_type,
                        "size": len(value.data),
                        "width": width,
                        "height": height,
                        "imageDelivery": "inline",
                        ARTIFACT_IMAGE_PART: types.Part.from_bytes(
                            data=value.data, mime_type=value.media_type
                        ),
                    }
                    call.succeed(
                        {key: item for key, item in result.items() if key != ARTIFACT_IMAGE_PART}
                    )
                    reservation.commit()
                    return result
            except _ImageUnavailable as error:
                result = {
                    "artifact": error.metadata.artifact.model_dump(by_alias=True),
                    "mediaType": error.metadata.media_type,
                    "size": error.metadata.size,
                    "imageDelivery": "unavailable",
                    "reason": error.reason,
                }
                call.succeed(result)
                return result
            finally:
                if reservation is not None:
                    reservation.release()
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
    description = """Publish an existing workspace file as an artifact in this Workflow Run.

    Runtime reads and publishes the actual file bytes. An available project
    workspace is required. Read an existing destination before replacing it.
    Runtime checks its version.

    Args:
        namespace: Destination artifact namespace.
        name: Destination artifact binding name.
        source_path: Existing file path relative to the Worker's project workspace.
        media_type: MIME type describing that file.

    Returns:
        Saved artifact metadata with its exact revision, mediaType and size.
    """

    def __init__(self, client, metrics, secrets, *, source: WorkspaceReader | None = None):
        super().__init__(client, metrics, secrets)
        self._source = source

    async def __call__(
        self,
        namespace: str,
        name: str,
        source_path: str,
        media_type: str,
    ) -> dict[str, Any]:
        arguments = {
            "namespace": namespace,
            "name": name,
            "source_path": source_path,
            "media_type": media_type,
        }
        with self._recorded(arguments) as call:
            require_model_visible_binding(namespace, name)
            if self._source is None:
                raise ToolInputError("source_path requires an available project workspace")
            expected_revision = self._client.versions.base(namespace, name)
            try:
                data = await self._source.read_bytes(source_path)
            except WorkspaceStorageError as error:
                raise ToolInputError(f"source_path is unavailable ({error})") from None
            if len(data) > artifact_payload_max_bytes():
                raise ToolInputError("source file exceeds the configured Artifact limit")
            target = ArtifactRef(namespace=namespace, name=name)
            self._client.versions.admit_write(target)
            result = await write_artifact_exact(
                self._client.client,
                ArtifactRef(namespace=namespace, name=name),
                data=data,
                media_type=media_type,
                expected_revision=expected_revision,
                max_bytes=artifact_payload_max_bytes(),
                reject_first=definite_write_rejection,
                reject_replay=lambda error: (
                    definite_write_rejection(error) and error.code != "artifact_conflict"
                ),
            )
            if isinstance(result, ArtifactValue):
                result = ArtifactWriteResult(
                    apiVersion=API_VERSION,
                    artifact=result.artifact,
                    mediaType=result.media_type,
                    size=len(result.data),
                )
            self._client.versions.observe(result.artifact)
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
