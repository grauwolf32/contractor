from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

import contractor_runtime.toolsets.run_artifacts as run_artifacts
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.metrics import MAX_METRIC_TOOL_CALLS
from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-tool-secret"


def test_factory_constructs_only_explicitly_selected_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_write_tool(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unselected write_artifact was instantiated")

    monkeypatch.setattr(run_artifacts, "WriteArtifactTool", unexpected_write_tool)

    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["list_artifacts", "read_artifact"])
        assert set(tools) == {"list_artifacts", "read_artifact"}
        assert "write_artifact" not in tools

        listed = await tools["list_artifacts"]("inputs")
        assert listed == [{"namespace": "inputs", "name": "source"}]
        read = await tools["read_artifact"]("inputs", "source")
        assert read["artifact"]["revision"] == "revision-read"
        assert base64.b64decode(read["dataBase64"]) == b"payload"
        assert state.metrics.counters["tool_calls"] == 2
        assert all(entry.result_size_bytes is not None for entry in state.metrics.tool_calls)
        assert "dataBase64" not in repr(state.metrics.tool_calls)

    asyncio.run(scenario())


def test_metrics_bound_events_and_redact_credential_bearing_urls() -> None:
    state = WorkerState()
    for index in range(MAX_METRIC_TOOL_CALLS + 3):
        state.metrics.record_tool_call(
            "probe",
            arguments={
                "callback": f"https://example.test/path?access_token={index}",
                "plain": "https://example.test/path",
            },
        )

    assert len(state.metrics.tool_calls) == MAX_METRIC_TOOL_CALLS
    assert state.metrics.truncated
    assert state.metrics.tool_calls[0].call_id == "tool-00000004"
    assert state.metrics.tool_calls[0].arguments == {
        "callback": "[REDACTED_URL]",
        "plain": "https://example.test/path",
    }


def test_write_tool_preserves_exact_revision_and_redacts_payload_metrics() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["write_artifact"])
        encoded = base64.b64encode(f"payload-{SECRET}".encode()).decode()

        result = await tools["write_artifact"](
            "inputs",
            "source",
            "text/plain",
            encoded,
            "revision-old",
        )
        assert result["artifact"]["revision"] == "revision-write"
        assert client.write_expected_revision == "revision-old"
        metric = state.metrics.tool_calls[0]
        assert metric.arguments is not None
        assert metric.arguments["data_base64"] == {
            "redacted": True,
            "size": len(encoded),
        }
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_tool_error_metrics_are_bounded_typed_and_secret_free() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient(
            write_error=ArtifactAPIError(403, "artifact_access_denied", False)
        )
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["write_artifact"])

        with pytest.raises(ArtifactAPIError):
            await tools["write_artifact"](
                "outputs",
                "result",
                "text/plain",
                base64.b64encode(SECRET.encode()).decode(),
            )
        assert state.metrics.counters["tool_errors"] == 1
        assert state.metrics.errors[0].code == "tool_write_artifact_failed"
        assert state.metrics.tool_calls[0].error is not None
        assert state.metrics.tool_calls[0].error.code == "artifact_access_denied"
        assert state.metrics.tool_calls[0].error.retryable is False
        assert "ArtifactAPIError" in state.metrics.tool_calls[0].error.message
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


async def create_tools(
    factory: RunArtifactsToolsetFactory,
    state: WorkerState,
    selected: list[str],
) -> dict[str, object]:
    settings = RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken=SECRET,
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )
    workspace_path = Path("/tmp/contractor-tool-test/allocation-test")
    result = await factory.create_selected(
        selected=selected,
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="builder",
        runtime_settings=settings,
        workspace=AllocationWorkspace(root=workspace_path.parent, path=workspace_path),
        state=state,
    )
    return dict(result)


class FakeArtifactClient:
    def __init__(self, *, write_error: Exception | None = None) -> None:
        self.write_error = write_error
        self.write_expected_revision: str | None = None

    async def list_artifacts(self, namespace: str | None = None) -> list[ArtifactRef]:
        assert namespace == "inputs"
        return [ArtifactRef(namespace="inputs", name="source")]

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        assert ref == ArtifactRef(namespace="inputs", name="source")
        return ArtifactValue(
            artifact=ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
            media_type="text/plain",
            data=b"payload",
        )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        if self.write_error is not None:
            raise self.write_error
        self.write_expected_revision = expected_revision
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=ArtifactRef(
                namespace=target.namespace,
                name=target.name,
                revision="revision-write",
            ),
            mediaType=media_type,
            size=len(data),
        )
