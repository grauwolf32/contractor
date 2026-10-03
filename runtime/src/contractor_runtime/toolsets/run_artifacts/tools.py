"""Selected model-visible tools backed by one allocation-bound ArtifactClient."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import MAX_ARTIFACT_BYTES, ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.common.artifact_read_cache import (
    ExactArtifactReadCache,
    allocation_artifact_read_cache,
)
from contractor_runtime.toolsets.common.artifact_visibility import (
    is_model_hidden_binding,
    require_model_visible_binding,
)
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _unconfigured_client,
    runtime_secrets,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.common.metrics import ToolMetrics
from contractor_runtime.toolsets.common.tool_base import ArtifactTool
from contractor_runtime.workspace import AllocationWorkspace

MAX_BASE64_PAYLOAD_LENGTH = ((MAX_ARTIFACT_BYTES + 2) // 3) * 4
MAX_READ_CHUNK_BYTES = 256 * 1024


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
    description = """List current artifact references visible in this Workflow Run.

    Args:
        namespace: Namespace to filter by; omit to list all visible namespaces.

    Returns:
        Artifact references with namespace, name and current revision.
    """

    async def __call__(self, namespace: str | None = None) -> list[dict[str, Any]]:
        arguments = {"namespace": namespace}
        with self._recorded(arguments) as call:
            if namespace == "skills":
                require_model_visible_binding(namespace, "selected")
            refs = await self._client.list_artifacts(namespace)
            result = [
                ref.model_dump(by_alias=True, exclude_none=True)
                for ref in refs
                if not is_model_hidden_binding(ref.namespace, ref.name)
            ]
            call.succeed({"count": len(result)})
            return result


class ReadArtifactTool(ArtifactTool):
    name = "read_artifact"
    description = """Read artifact bytes from this Workflow Run as base64, in pages.

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
                self._client, ArtifactRef(namespace=namespace, name=name, revision=revision)
            )
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
