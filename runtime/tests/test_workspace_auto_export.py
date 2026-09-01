from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes.model import json_result, scripted_model, text_result

from contractor_runtime.adk_runtime import AdkWorkerRuntime, AdkWorkerRuntimeFactory
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    AllocationWorkspaceExportV2,
    ArtifactRef,
    ArtifactWriteResult,
    ModelPolicyRef,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    RuntimeSettings,
    SandboxProfileRef,
    StageContentRequest,
    StageContentResult,
    StageOutcome,
    TerminationError,
    WorkerRuntimeRef,
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.projectfs import (
    WORKSPACE_DIFF_MEDIA_TYPE,
    WORKSPACE_OVERLAY_MEDIA_TYPE,
    ManagedWorkspaceTree,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
    WorkspaceAutoExporter,
    WorkspaceExportError,
    decode_workspace_state,
)
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.workspace import AllocationWorkspace


@pytest.mark.parametrize("outcome", [StageOutcome.SUCCEEDED, StageOutcome.FAILED])
def test_export_persists_exact_cumulative_state_and_checkpoint_diff(
    tmp_path: Path, outcome: StageOutcome
) -> None:
    async def scenario() -> None:
        session = await overlay("complete")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")

        first = await exporter.export(stage_result(outcome))

        assert set(first.result.artifacts) == {"workspace_state", "workspace_diff"}
        assert first.result.artifacts["workspace_state"].revision == "revision-1"
        assert first.result.artifacts["workspace_diff"].revision == "revision-1"
        state = client.binding("workspace_state")
        diff = client.binding("workspace_diff")
        assert state.media_type == WORKSPACE_OVERLAY_MEDIA_TYPE
        assert diff.media_type == WORKSPACE_DIFF_MEDIA_TYPE
        assert b"first result" in diff.data
        assert json.loads(state.data)["resultWorkspaceDigest"] == first.result_workspace_digest
        reconstructed = decode_workspace_state(
            state.data,
            session._source,
            session.limits,  # type: ignore[attr-defined]
        )
        assert reconstructed.snapshot() == await session.snapshot()
        assert await session.changed_paths() == ()

        await session.write_text("source.txt", "second result\n")
        second = await exporter.export(stage_result(outcome))

        assert second.result.artifacts["workspace_state"].revision == "revision-2"
        assert second.result.artifacts["workspace_diff"].revision == "revision-2"
        assert b"first result" in client.binding("workspace_diff").data
        assert b"second result" in client.binding("workspace_diff").data
        cumulative = client.binding("workspace_state").data
        reconstructed = decode_workspace_state(
            cumulative,
            session._source,
            session.limits,  # type: ignore[attr-defined]
        )
        assert reconstructed.snapshot() == await session.snapshot()
        assert await session.changed_paths() == ()

    asyncio.run(scenario())


def test_partial_write_and_lost_response_never_advance_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("retry")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")
        client.fail_before("workspace_diff")

        with pytest.raises(WorkspaceExportError) as partial:
            await exporter.export(stage_result(StageOutcome.SUCCEEDED))
        assert partial.value.retryable
        assert client.binding("workspace_state").revision == "revision-1"
        assert "workspace_diff" not in client.bindings
        assert await session.changed_paths() == ("source.txt",)

        completed = await exporter.export(stage_result(StageOutcome.SUCCEEDED))
        assert completed.result.artifacts["workspace_state"].revision == "revision-1"
        assert completed.result.artifacts["workspace_diff"].revision == "revision-1"
        assert await session.changed_paths() == ()

        await session.write_text("source.txt", "second result\n")
        client.fail_after("workspace_state")
        with pytest.raises(WorkspaceExportError):
            await exporter.export(stage_result(StageOutcome.SUCCEEDED))
        assert client.binding("workspace_state").revision == "revision-2"
        assert client.binding("workspace_diff").revision == "revision-1"
        assert await session.changed_paths() == ("source.txt",)

        recovered = await exporter.export(stage_result(StageOutcome.SUCCEEDED))
        assert recovered.result.artifacts["workspace_state"].revision == "revision-2"
        assert recovered.result.artifacts["workspace_diff"].revision == "revision-2"
        assert await session.changed_paths() == ()

    asyncio.run(scenario())


def test_reserved_result_collision_is_rejected_without_artifact_io(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("collision")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        result = stage_result(StageOutcome.SUCCEEDED).model_copy(
            update={
                "artifacts": {
                    "workspace_state": ArtifactRef(
                        namespace="editor", name="invented", revision="revision-invented"
                    )
                }
            }
        )

        with pytest.raises(WorkspaceExportError) as collision:
            await exporter.export(result)
        assert collision.value.cause == "reserved_result_slot"
        assert not collision.value.retryable
        assert client.calls == []

    asyncio.run(scenario())


def test_adk_maps_partial_export_to_stable_retryable_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("runtime-partial")
        client = MemoryArtifactClient()
        client.fail_before("workspace_diff")
        runtime = await runtime_with_export(
            tmp_path,
            session,
            client,
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Model completed",
                        "artifacts": {},
                    }
                )
            ],
        )
        await session.write_text("source.txt", "partial export\n")

        result = await runtime.invoke(stage_request())

        assert result.outcome is StageOutcome.FAILED
        assert result.error is not None
        assert result.error.code == "workspace_export_failed"
        assert result.error.retryable
        assert result.artifacts == {}
        assert set(client.bindings) == {"workspace_state"}
        assert await session.changed_paths() == ("source.txt",)
        assert runtime._metrics.counters["workspace_exports.failed"] == 1
        assert runtime._metrics.errors[-1].code == "workspace_export_failed"
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_treats_protocol_shaped_text_as_summary_before_workspace_export(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        failed_session = await overlay("semantic-failure")
        failed_client = MemoryArtifactClient()
        failed_runtime = await runtime_with_export(
            tmp_path,
            failed_session,
            failed_client,
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "failed",
                        "summary": "Useful partial analysis",
                        "artifacts": {},
                        "error": {
                            "code": "analysis_incomplete",
                            "message": "Analysis is incomplete",
                            "retryable": True,
                        },
                    }
                )
            ],
        )
        await failed_session.write_text("source.txt", "useful partial state\n")
        failed = await failed_runtime.invoke(stage_request())
        assert failed.error is None
        assert '"outcome":"failed"' in failed.summary
        assert set(failed.artifacts) == {"workspace_state", "workspace_diff"}
        assert failed_runtime._metrics.counters["workspace_exports.succeeded"] == 1
        await failed_runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

        malformed_session = await overlay("malformed")
        malformed_client = MemoryArtifactClient()
        malformed_runtime = await runtime_with_export(
            tmp_path,
            malformed_session,
            malformed_client,
            [text_result("ordinary summary")],
        )
        await malformed_session.write_text("source.txt", "must stay private\n")
        malformed = await malformed_runtime.invoke(stage_request())
        assert malformed.error is None
        assert malformed.summary == "ordinary summary"
        assert set(malformed.artifacts) == {"workspace_state", "workspace_diff"}
        assert await malformed_session.changed_paths() == ()
        await malformed_runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

        reserved_session = await overlay("reserved")
        reserved_client = MemoryArtifactClient()
        reserved_runtime = await runtime_with_export(
            tmp_path,
            reserved_session,
            reserved_client,
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Invented export",
                        "artifacts": {
                            "workspace_diff": {
                                "namespace": "editor",
                                "name": "workspace_diff",
                                "revision": "invented",
                            }
                        },
                    }
                )
            ],
        )
        reserved = await reserved_runtime.invoke(stage_request())
        assert reserved.error is None
        assert '"revision":"invented"' in reserved.summary
        assert reserved.artifacts["workspace_diff"].revision == "revision-1"
        assert reserved.artifacts["workspace_state"].revision == "revision-1"
        await reserved_runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_invocation_cancellation_during_export_writes_nothing_and_keeps_checkpoint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session = await overlay("cancelled")
        client = MemoryArtifactClient()
        client.block_before("workspace_state")
        runtime = await runtime_with_export(
            tmp_path,
            session,
            client,
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Would export",
                        "artifacts": {},
                    }
                )
            ],
        )
        await session.write_text("source.txt", "cancelled result\n")
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(client.blocked.wait(), timeout=1)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert client.bindings == {}
        assert await session.changed_paths() == ("source.txt",)
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_factory_attaches_runtime_owned_exporter(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("factory")
        client = MemoryArtifactClient()
        state = WorkerState()
        model = scripted_model(
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Factory result",
                        "artifacts": {},
                    }
                )
            ]
        )
        factory = AdkWorkerRuntimeFactory(
            model_factory=lambda _: model,
            artifact_client_factory=lambda *_: client,  # type: ignore[arg-type,return-value]
        )
        runtime = await factory.create(build_context(tmp_path, session, state))
        await session.write_text("source.txt", "factory export\n")

        result = await runtime.invoke(stage_request())

        assert set(result.artifacts) == {"workspace_state", "workspace_diff"}
        assert runtime._workspace_exporter is not None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


async def overlay(name: str) -> OverlayWorkspaceSession:
    limits = WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=1 << 20,
        max_managed_text_bytes=1 << 19,
        max_file_bytes=1 << 18,
    )
    provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=limits))
    storage = await provider.create(name)
    source = ManagedWorkspaceTree(text_files={"source.txt": "source\n"})
    return OverlayWorkspaceSession(
        storage=storage,
        content_root=f"{storage.root}/run_workdir",
        limits=limits,
        directories=source.directories,
        text_files=source.text_files,
        binary_paths=source.binary_paths,
        stored_binary_paths=source.stored_binary_paths,
    )


def make_exporter(
    session: OverlayWorkspaceSession, client: MemoryArtifactClient
) -> WorkspaceAutoExporter:
    return WorkspaceAutoExporter(
        workspace=session,
        client=client,  # type: ignore[arg-type]
        namespace="editor",
        slots=AllocationWorkspaceExportV2(state="workspace_state", diff="workspace_diff"),
    )


def stage_result(outcome: StageOutcome) -> StageContentResult:
    error = None
    if outcome is StageOutcome.FAILED:
        error = TerminationError(
            code="analysis_incomplete", message="Analysis is incomplete", retryable=True
        )
    return StageContentResult(
        apiVersion=API_VERSION,
        outcome=outcome,
        summary="Worker completed gracefully",
        artifacts={},
        error=error,
    )


async def runtime_with_export(
    tmp_path: Path,
    session: OverlayWorkspaceSession,
    client: MemoryArtifactClient,
    responses: list[Any],
) -> AdkWorkerRuntime:
    state = WorkerState()
    context = build_context(tmp_path, session, state)
    runtime = AdkWorkerRuntime(
        context,
        scripted_model(responses),
        workspace_exporter=make_exporter(session, client),
    )
    await runtime.start()
    return runtime


def build_context(
    tmp_path: Path, session: OverlayWorkspaceSession, state: WorkerState
) -> WorkerBuildContext:
    template = agent_template()
    return WorkerBuildContext(
        allocation_id=f"allocation-{session.storage.owner_token}",
        run_id="run-export",
        stage_execution_id="stage-export",
        logical_agent_name="editor",
        namespace="editor",
        description=template.description,
        instruction=template.instructions.text,
        card_version=template.ref.version,
        model_policy=template.model_policy,
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        tools={},
        state=state,
        a2a_base_url="https://runtime.example",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken="test-token",
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
        project_workspace=session,
        workspace_export=AllocationWorkspaceExportV2(
            state="workspace_state", diff="workspace_diff"
        ),
    )


def agent_template() -> ResolvedAgentTemplate:
    policy = ResolvedModelPolicy(
        ref=ModelPolicyRef(policyId="worker", version="1", digest="sha256:" + "2" * 64),
        model="worker-model",
        maxOutputTokens=4096,
        maxModelCalls=8,
        maxToolCalls=16,
        maxTotalTokens=32768,
        temperature=0.1,
    )
    return ResolvedAgentTemplate(
        ref=AgentTemplateRef(
            templateId="workspace_editor", version="1", digest="sha256:" + "0" * 64
        ),
        description="Edit a project workspace",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/workspace.md",
            digest="sha256:" + "1" * 64,
            text="Analyze the workspace.",
        ),
        modelPolicy=policy,
        toolsets=[],
        sandboxProfile=SandboxProfileRef(sandboxProfileId="local-workdir", version="1"),
    )


def stage_request() -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        objective="Analyze source",
        instructions="Return a result.",
        parameters={},
        artifacts={},
    )


class MemoryArtifactClient:
    def __init__(self) -> None:
        self.bindings: dict[str, StoredArtifact] = {}
        self.calls: list[tuple[str, str]] = []
        self._fail_before: set[str] = set()
        self._fail_after: set[str] = set()
        self._block_before: set[str] = set()
        self.blocked = asyncio.Event()

    def fail_before(self, name: str) -> None:
        self._fail_before.add(name)

    def fail_after(self, name: str) -> None:
        self._fail_after.add(name)

    def block_before(self, name: str) -> None:
        self._block_before.add(name)

    def binding(self, name: str) -> StoredArtifact:
        return self.bindings[name]

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.calls.append(("read", ref.name))
        current = self.bindings.get(ref.name)
        if current is None:
            raise ArtifactAPIError(404, "artifact_not_found", False)
        exact = ArtifactRef(namespace=ref.namespace, name=ref.name, revision=current.revision)
        return ArtifactValue(
            artifact=exact,
            media_type=current.media_type,
            data=current.data,
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
        self.calls.append(("write", target.name))
        if target.name in self._block_before:
            self.blocked.set()
            await asyncio.Event().wait()
        if target.name in self._fail_before:
            self._fail_before.remove(target.name)
            raise ArtifactTransportError("synthetic failure before commit")
        current = self.bindings.get(target.name)
        if (current is None and expected_revision is not None) or (
            current is not None and current.revision != expected_revision
        ):
            raise ArtifactAPIError(409, "artifact_conflict", True)
        revision_number = 1 if current is None else int(current.revision.rsplit("-", 1)[1]) + 1
        revision = f"revision-{revision_number}"
        self.bindings[target.name] = StoredArtifact(revision, media_type, data)
        if target.name in self._fail_after:
            self._fail_after.remove(target.name)
            raise ArtifactTransportError("synthetic lost response")
        exact = ArtifactRef(namespace=target.namespace, name=target.name, revision=revision)
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=exact,
            mediaType=media_type,
            size=len(data),
        )


class StoredArtifact:
    def __init__(self, revision: str, media_type: str, data: bytes) -> None:
        self.revision = revision
        self.media_type = media_type
        self.data = data
