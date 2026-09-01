"""Selected model-visible tools backed by one allocation-bound ArtifactClient."""

from __future__ import annotations

import base64
import binascii
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import MAX_ARTIFACT_BYTES, ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.artifact_visibility import (
    is_model_hidden_binding,
    model_visible_exact_refs,
    require_model_visible_binding,
)
from contractor_runtime.workspace import AllocationWorkspace

MAX_BASE64_PAYLOAD_LENGTH = ((MAX_ARTIFACT_BYTES + 2) // 3) * 4


class ToolMetrics(Protocol):
    def record_tool_call(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
        secrets: tuple[str, ...] = (),
        duration_ms: int | None = None,
    ) -> None: ...


ArtifactClientFactory = Callable[[str, RuntimeSettings], ArtifactClient]


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
    ) -> Mapping[str, Any]:
        del adapter_handles
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("run-artifacts@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        secrets = gateway_secrets(runtime_settings)
        builders = {
            "list_artifacts": lambda: ListArtifactsTool(client, metrics, secrets),
            "read_artifact": lambda: ReadArtifactTool(client, metrics, secrets),
            "write_artifact": lambda: WriteArtifactTool(client, metrics, secrets),
        }
        return {name: builders[name]() for name in selected}


class _BaseTool:
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
        value = getattr(self._client, "known_exact_refs", ())
        return model_visible_exact_refs(value)

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


class ListArtifactsTool(_BaseTool):
    name = "list_artifacts"
    description = "List current artifact refs in this Workflow Run, optionally by namespace."

    async def __call__(self, namespace: str | None = None) -> list[dict[str, Any]]:
        started_ns = time.perf_counter_ns()
        arguments = {"namespace": namespace}
        try:
            if namespace == "skills":
                require_model_visible_binding(namespace, "selected")
            refs = await self._client.list_artifacts(namespace)
            result = [
                ref.model_dump(by_alias=True, exclude_none=True)
                for ref in refs
                if not is_model_hidden_binding(ref.namespace, ref.name)
            ]
            self._success(arguments, {"count": len(result)}, started_ns)
            return result
        except Exception as error:
            self._failure(arguments, error, started_ns)
            raise


class ReadArtifactTool(_BaseTool):
    name = "read_artifact"
    description = "Read current or exact artifact bytes from this Workflow Run as base64."

    async def __call__(
        self,
        namespace: str,
        name: str,
        revision: str | None = None,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        arguments = {"namespace": namespace, "name": name, "revision": revision}
        try:
            require_model_visible_binding(namespace, name)
            value = await self._client.read_artifact(
                ArtifactRef(namespace=namespace, name=name, revision=revision)
            )
            result = {
                "artifact": value.artifact.model_dump(by_alias=True),
                "mediaType": value.media_type,
                "size": len(value.data),
                "dataBase64": base64.b64encode(value.data).decode("ascii"),
            }
            self._success(
                arguments,
                {
                    "artifact": result["artifact"],
                    "mediaType": value.media_type,
                    "size": len(value.data),
                },
                started_ns,
            )
            return result
        except Exception as error:
            self._failure(arguments, error, started_ns)
            raise


class WriteArtifactTool(_BaseTool):
    name = "write_artifact"
    description = "Create or CAS-update artifact bytes in this Workflow Run from base64."

    async def __call__(
        self,
        namespace: str,
        name: str,
        media_type: str,
        data_base64: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        arguments = {
            "namespace": namespace,
            "name": name,
            "media_type": media_type,
            "data_base64": data_base64,
            "expected_revision": expected_revision,
        }
        try:
            require_model_visible_binding(namespace, name)
            if len(data_base64) > MAX_BASE64_PAYLOAD_LENGTH:
                raise ValueError("base64 artifact payload exceeds the 16 MiB limit")
            try:
                data = base64.b64decode(data_base64, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("data_base64 is not valid canonical base64") from error
            result = await self._client.write_artifact(
                ArtifactRef(namespace=namespace, name=name),
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


def gateway_secrets(settings: RuntimeSettings) -> tuple[str, ...]:
    token = settings.llm_gateway_token
    return () if token is None else (token.get_secret_value(),)


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    return ArtifactClient(allocation_id, _UnavailableTransport())


class _UnavailableTransport:
    async def request(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("Artifact transport is not configured")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
