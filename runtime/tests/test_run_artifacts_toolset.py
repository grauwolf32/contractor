from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime
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
from contractor_runtime.toolsets.artifact_visibility import is_reserved_memory_binding
from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-tool-secret"


def test_memory_prefix_policy_is_namespace_aware() -> None:
    assert is_reserved_memory_binding("analysis", "memory.note")
    assert not is_reserved_memory_binding("analysis", "report")
    for namespace in ("inputs", "outputs", "skills"):
        assert not is_reserved_memory_binding(namespace, "memory.note")


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
        assert listed == [
            {"namespace": "inputs", "name": "source"},
            {"namespace": "inputs", "name": "memory.allowed_input"},
        ]
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


def test_memory_artifacts_are_hidden_from_generic_list_read_write_and_known_refs() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(
            factory, state, ["list_artifacts", "read_artifact", "write_artifact"]
        )

        assert await tools["list_artifacts"]("analysis") == [
            {"namespace": "analysis", "name": "report"}
        ]
        calls_before = (client.read_calls, client.write_calls)
        list_calls_before = client.list_calls
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["list_artifacts"]("skills")
        assert client.list_calls == list_calls_before
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"]("skills", "likec4", "revision-skill")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"]("skills", "likec4", "application/octet-stream", "")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"]("analysis", "memory.hidden", "revision-hidden")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"](
                "analysis",
                "memory.hidden",
                "application/json",
                base64.b64encode(b"hidden").decode(),
            )
        assert (client.read_calls, client.write_calls) == calls_before
        observed = {
            (ref.namespace, ref.name) for tool in tools.values() for ref in tool.known_exact_refs
        }
        assert ("analysis", "memory.hidden") not in observed
        assert ("skills", "likec4") not in observed
        assert ("inputs", "memory.allowed_input") in observed

    asyncio.run(scenario())


def test_reserved_skills_namespace_is_hidden_before_generic_client_access() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(
            factory, state, ["list_artifacts", "read_artifact", "write_artifact"]
        )

        calls_before = (client.list_calls, client.read_calls, client.write_calls)
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["list_artifacts"]("skills")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"]("skills", "likec4", "revision-skill")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"]("skills", "likec4", "application/octet-stream", "")
        assert (client.list_calls, client.read_calls, client.write_calls) == calls_before
        assert all(
            ref.namespace != "skills" for tool in tools.values() for ref in tool.known_exact_refs
        )

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
        self.read_calls = 0
        self.write_calls = 0
        self.list_calls = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return (
            ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
            ArtifactRef(namespace="inputs", name="memory.allowed_input", revision="revision-input"),
            ArtifactRef(namespace="analysis", name="report", revision="revision-report"),
            ArtifactRef(namespace="analysis", name="memory.hidden", revision="revision-hidden"),
            ArtifactRef(namespace="skills", name="likec4", revision="revision-skill"),
        )

    async def list_artifacts(self, namespace: str | None = None) -> list[ArtifactRef]:
        self.list_calls += 1
        if namespace == "inputs":
            return [
                ArtifactRef(namespace="inputs", name="source"),
                ArtifactRef(namespace="inputs", name="memory.allowed_input"),
            ]
        if namespace == "analysis":
            return [
                ArtifactRef(namespace="analysis", name="report"),
                ArtifactRef(namespace="analysis", name="memory.hidden"),
            ]
        raise AssertionError(f"unexpected namespace {namespace!r}")

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.read_calls += 1
        assert ref == ArtifactRef(namespace="inputs", name="source")
        return ArtifactValue(
            artifact=ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
            media_type="text/plain",
            data=b"payload",
            binding_created_at=datetime.now(UTC),
            revision_created_at=datetime.now(UTC),
        )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        self.write_calls += 1
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
