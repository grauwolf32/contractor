"""Bounded UTF-8 tools over one allocation-bound RunScope ArtifactClient."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Any

from contractor_runtime import threads
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_metrics import ToolMetrics
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.toolsets.common.artifact_visibility import (
    require_model_visible_binding,
)
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _reject_unconfigured_client,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.finding_analysis import (
    MEDIA_TYPE as ANALYSIS_MEDIA_TYPE,
)
from contractor_runtime.toolsets.common.finding_analysis import (
    canonical_analysis,
)
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.common.tool_base import ArtifactTool
from contractor_runtime.workspace import AllocationWorkspace

MAX_TEXT_WRITE_BYTES = 1024 * 1024


class TextArtifactsToolsetFactory:
    """Construct only the explicitly selected ``text-artifacts@1`` tools."""

    ref = "text-artifacts@1"
    exported_tools = frozenset({"write_text_artifact"})
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
        resources: ToolResources,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, adapter_handles
        require_selected_tools(selected, self.exported_tools)
        metrics = require_metrics(resources, "text-artifacts@1")
        client = self._client_factory(allocation_id, runtime_settings)
        secrets = runtime_secrets(runtime_settings)
        constructors = {
            "write_text_artifact": partial(WriteTextArtifactTool, namespace=namespace),
        }
        return {name: constructors[name](client, metrics, secrets) for name in selected}


class WriteTextArtifactTool(ArtifactTool):
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
        arguments = {
            "name": name,
            "media_type": media_type,
            "expected_revision": expected_revision,
            "content": text,
        }
        with self._recorded(arguments) as call:
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
            if media_type == ANALYSIS_MEDIA_TYPE:
                try:
                    data = await threads.to_thread_until_done(
                        lambda: canonical_analysis(text), name="finding-analysis-authoring"
                    )
                except ValueError:
                    raise ToolInputError("finding analysis is invalid") from None
            result = await self._client.write_artifact(
                ArtifactRef(namespace=self._namespace, name=name),
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
