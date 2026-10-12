"""Selected model-visible tools backed by one allocation-bound ArtifactClient."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import (
    MAX_ARTIFACT_BYTES,
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
from contractor_runtime.toolsets.common.artifact_media import is_binary_media_type
from contractor_runtime.toolsets.common.artifact_read_cache import (
    ExactArtifactReadCache,
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
from contractor_runtime.toolsets.common.tool_base import ArtifactTool
from contractor_runtime.workspace import AllocationWorkspace

MAX_BASE64_PAYLOAD_LENGTH = ((MAX_ARTIFACT_BYTES + 2) // 3) * 4
MAX_READ_CHUNK_BYTES = 256 * 1024
# Refs per list_artifacts page: a page stays a few thousand tokens for small
# models; the private API allows up to MAX_BINDING_LIST_LIMIT.
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
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del adapter_handles
        require_selected_tools(selected, self.exported_tools)
        metrics = require_metrics(state, "run-artifacts@1")
        client = self._client_factory(allocation_id, runtime_settings)
        secrets = runtime_secrets(runtime_settings)
        cache = allocation_artifact_read_cache(state) if "read_artifact" in selected else None
        builders = {
            "list_artifacts": lambda: ListArtifactsTool(client, metrics, secrets),
            "read_artifact": lambda: ReadArtifactTool(client, metrics, secrets, cache=cache),
            "write_artifact": lambda: WriteArtifactTool(client, metrics, secrets),
        }
        return {name: builders[name]() for name in selected}


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
                namespace, limit=LIST_PAGE_SIZE, cursor=cursor
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
    description = """Read artifact bytes from this Workflow Run as base64, in pages.

    Only UTF-8 text artifacts are returned; a binary artifact such as a source
    archive is rejected, so read source code with the filesystem or source
    tools. A binary media type is rejected before the artifact is downloaded.
    Each call returns at most 256 KiB of decoded bytes. When hasMore is true,
    call again with offset set to nextOffset and the same exact revision. Use
    read_text_artifact for a bounded UTF-8 text preview.

    Args:
        namespace: Artifact namespace.
        name: Exact artifact binding name.
        revision: Exact revision to read; omit for the current revision. Use the
            exact revision supplied in Stage context when available.
        offset: Zero-based byte offset to start reading from; defaults to 0.
        length: Maximum bytes to return, from 1 to 262144; defaults to 262144.

    Returns:
        The exact artifact reference, mediaType, total byte size, offset, the
        returned byte length, dataBase64, hasMore and nextOffset (null when no
        bytes remain).
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
        offset: int = 0,
        length: int = MAX_READ_CHUNK_BYTES,
    ) -> dict[str, Any]:
        arguments = {
            "namespace": namespace,
            "name": name,
            "revision": revision,
            "offset": offset,
            "length": length,
        }
        with self._recorded(arguments) as call:
            require_model_visible_binding(namespace, name)
            if type(offset) is not int or offset < 0:
                raise ToolInputError("offset must be a non-negative integer")
            if type(length) is not int or not 1 <= length <= MAX_READ_CHUNK_BYTES:
                raise ToolInputError(f"length must be between 1 and {MAX_READ_CHUNK_BYTES}")
            # The Artifact API has no range reads, so page a cached exact revision.
            value = await self._cache.read(
                self._client,
                ArtifactRef(namespace=namespace, name=name, revision=revision),
                accept=_refuse_binary_media_type,
            )
            try:
                value.data.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise _binary_artifact(value.media_type) from None
            size = len(value.data)
            if offset > size:
                raise ToolInputError("offset exceeds the artifact size")
            chunk = value.data[offset : offset + length]
            end = offset + len(chunk)
            result = {
                "artifact": value.artifact.model_dump(by_alias=True),
                "mediaType": value.media_type,
                "size": size,
                "offset": offset,
                "length": len(chunk),
                "dataBase64": base64.b64encode(chunk).decode("ascii"),
                "hasMore": end < size,
                "nextOffset": end if end < size else None,
            }
            call.succeed(
                {
                    "artifact": result["artifact"],
                    "mediaType": value.media_type,
                    "size": size,
                    "offset": offset,
                    "length": len(chunk),
                    "hasMore": result["hasMore"],
                }
            )
            return result


class WriteArtifactTool(ArtifactTool):
    name = "write_artifact"
    description = """Create or update an artifact in this Workflow Run from base64 bytes.

    Updates require the expected current revision to prevent overwriting a
    concurrent write.

    Args:
        namespace: Destination artifact namespace.
        name: Destination artifact binding name.
        media_type: MIME type describing the decoded bytes.
        data_base64: Canonical base64 payload within the 64 MiB decoded artifact limit.
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
            if len(data_base64) > MAX_BASE64_PAYLOAD_LENGTH:
                raise ToolInputError("base64 artifact payload exceeds the 64 MiB limit")
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


def _refuse_binary_media_type(media_type: str) -> None:
    if is_binary_media_type(media_type):
        raise _binary_artifact(media_type)


def _binary_artifact(media_type: str) -> ToolInputError:
    # Base64 of binary bytes is unreadable to the model and floods its
    # context: a 70 KB source ZIP became about 68k prompt tokens.
    return ToolInputError(
        f"artifact is binary ({media_type}) and is not returned; "
        "read source code with the filesystem or source tools instead"
    )


def _matches(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None
