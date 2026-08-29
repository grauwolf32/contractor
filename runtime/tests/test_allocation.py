from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    AgentTemplateRef,
    AllocationSpec,
    FinalizeAllocationRequest,
    ModelPolicyRef,
    ReleaseAllocationRequest,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    RuntimeSettings,
    SandboxProfileRef,
    TerminationError,
    ToolsetRef,
    ToolsetSelection,
    WorkerRuntimeRef,
)
from contractor_runtime.digests import (
    _agent_template_digest,
    _digest_bytes,
    _model_policy_digest,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    RunArtifactsToolsetFactory,
    StubADKWorkerRuntimeFactory,
    StubWorkerRuntime,
    WorkerBuildContext,
    built_in_factories,
)
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory

NOW = datetime(2026, 8, 29, 10, 0, tzinfo=UTC)
SECRET = "allocation-only-recognizable-secret"


def test_prepare_is_single_slot_idempotent_and_constructs_only_selected_tools(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path)
        spec = make_spec(tools=["read_artifact"])

        first = await service.prepare(spec)
        second = await service.prepare(spec)
        assert second == first
        assert first.worker_handle.allocation_id == spec.allocation_id
        assert first.worker_handle.lease_expires_at == spec.lease_expires_at
        assert SECRET not in repr(first)
        snapshot = await service.snapshot()
        assert snapshot is not None
        assert snapshot.tool_names == ("read_artifact",)
        assert snapshot.has_runtime_settings
        assert snapshot.has_worker
        assert Path(snapshot.workspace).is_dir()
        assert SECRET not in repr(service._context)

        other = make_spec(allocation_id="allocation-other")
        with pytest.raises(AllocationError, match="another allocation") as conflict:
            await service.prepare(other)
        assert conflict.value.code == "allocation_conflict"
        assert (await state.snapshot()).process_state is ProcessState.ALLOCATED

    asyncio.run(scenario())


def test_bad_digest_and_unsupported_ref_leave_no_residue(tmp_path: Path) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path)
        bad_digest = make_spec()
        bad_digest.agent_template.ref.digest = "sha256:" + "0" * 64
        with pytest.raises(AllocationError) as mismatch:
            await service.prepare(bad_digest)
        assert mismatch.value.code == "template_digest_mismatch"
        assert not mismatch.value.retryable
        assert await service.snapshot() is None
        assert list(tmp_path.iterdir()) == []

        unsupported = make_spec()
        unsupported.agent_template.runtime = WorkerRuntimeRef(runtimeId="unknown", version="1")
        resign_template(unsupported.agent_template)
        with pytest.raises(AllocationError) as missing:
            await service.prepare(unsupported)
        assert missing.value.code == "unsupported_worker_runtime"
        assert await service.snapshot() is None
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_prepare_failure_rolls_back_workspace_tools_and_slot(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-test")
        await state.mark_registered()
        sandbox = LocalWorkdirFactory(tmp_path)
        failing_tools = FailingToolsetFactory()
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={"run-artifacts@1": failing_tools},
            sandbox_profiles={"local-workdir@1": sandbox},
        )
        service = AllocationService(
            state, registry, a2a_base_url="https://runtime.example", now=lambda: NOW
        )

        with pytest.raises(AllocationError) as failure:
            await service.prepare(make_spec())
        assert failure.value.code == "allocation_preparation_failed"
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert await service.snapshot() is None
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_finalize_report_and_release_erase_context_and_workspace(tmp_path: Path) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path)
        spec = make_spec()
        await service.prepare(spec)
        workspace = Path((await service.snapshot()).workspace)  # type: ignore[union-attr]
        request = FinalizeAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
            finalizationId="finalization-1",
            deadline=NOW + timedelta(seconds=30),
        )

        first = await service.finalize(request)
        second = await service.finalize(request)
        assert second == first
        assert first.report.complete
        assert first.report.allocation_id == spec.allocation_id
        assert SECRET not in first.model_dump_json(by_alias=True)
        with pytest.raises(AllocationError) as conflict:
            await service.finalize(request.model_copy(update={"finalization_id": "finalization-2"}))
        assert conflict.value.code == "allocation_conflict"
        assert (await state.snapshot()).process_state is ProcessState.DRAINING
        assert not (await service.snapshot()).has_worker  # type: ignore[union-attr]

        release = ReleaseAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
        )
        await service.release(release)
        await service.release(release)
        assert await service.snapshot() is None
        assert not workspace.exists()
        state_snapshot = await state.snapshot()
        assert state_snapshot.process_state is ProcessState.IDLE
        assert state_snapshot.allocation_id is None

    asyncio.run(scenario())


def test_cleanup_failure_fences_slot_until_idempotent_release_retry(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-test")
        await state.mark_registered()
        sandbox = FailOnceSandbox(tmp_path)
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={"run-artifacts@1": RunArtifactsToolsetFactory()},
            sandbox_profiles={"local-workdir@1": sandbox},
        )
        service = AllocationService(
            state, registry, a2a_base_url="https://runtime.example", now=lambda: NOW
        )
        spec = make_spec()
        await service.prepare(spec)
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalization-1",
                deadline=NOW + timedelta(seconds=30),
            )
        )
        release = ReleaseAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
        )

        with pytest.raises(AllocationError) as cleanup:
            await service.release(release)
        assert cleanup.value.code == "allocation_cleanup_failed"
        assert cleanup.value.retryable
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert (await service.snapshot()).has_runtime_settings  # type: ignore[union-attr]
        with pytest.raises(AllocationError):
            await service.prepare(make_spec(allocation_id="allocation-new"))

        await service.release(release)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_abort_adds_reason_and_forces_exit_if_worker_cannot_stop(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-test")
        await state.mark_registered()
        runtime_factory = FailingStopRuntimeFactory()
        builtins = built_in_factories(tmp_path)
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": runtime_factory},
            toolsets=builtins.toolsets,
            sandbox_profiles=builtins.sandbox_profiles,
        )
        exit_codes: list[int] = []
        service = AllocationService(
            state,
            registry,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
            force_exit=exit_codes.append,
        )
        spec = make_spec()
        await service.prepare(spec)
        request = AbortAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
            abortId="abort-1",
            reason=TerminationError(
                code="run_cancelled", message="WorkflowRun was cancelled", retryable=False
            ),
            deadline=NOW + timedelta(seconds=30),
        )

        with pytest.raises(AllocationError) as stopped:
            await service.abort(request)
        assert stopped.value.code == "worker_stop_unconfirmed"
        assert exit_codes == [70]
        assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())


async def make_service(tmp_path: Path) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-test")
    await state.mark_registered()
    service = AllocationService(
        state,
        built_in_factories(tmp_path),
        a2a_base_url="https://runtime.example",
        now=lambda: NOW,
        force_exit=lambda _: None,
    )
    return state, service


def make_spec(
    *,
    allocation_id: str = "allocation-1",
    tools: list[str] | None = None,
) -> AllocationSpec:
    instructions_text = "Read declared inputs and write the requested result."
    policy = ResolvedModelPolicy(
        ref=ModelPolicyRef(
            policyId="worker",
            version="1",
            digest="sha256:" + "0" * 64,
        ),
        model="worker-model",
        maxOutputTokens=4096,
        temperature=0.1,
    )
    policy.ref.digest = _model_policy_digest(policy)
    template = ResolvedAgentTemplate(
        ref=AgentTemplateRef(
            templateId="artifact_builder",
            version="1",
            digest="sha256:" + "0" * 64,
        ),
        description="Builds the requested artifact",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/artifact-builder.md",
            digest=_digest_bytes(instructions_text.encode()),
            text=instructions_text,
        ),
        modelPolicy=policy,
        toolsets=[
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="run-artifacts", version="1"),
                tools=tools or ["read_artifact", "write_artifact"],
            )
        ],
        sandboxProfile=SandboxProfileRef(sandboxProfileId="local-workdir", version="1"),
    )
    resign_template(template)
    return AllocationSpec(
        apiVersion=API_VERSION,
        allocationId=allocation_id,
        runId="run-1",
        stageExecutionId="stage-execution-1",
        logicalAgentName="builder",
        namespace="builder",
        leaseExpiresAt=NOW + timedelta(seconds=60),
        agentTemplate=template,
        runtimeSettings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken=SECRET,
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
    )


def resign_template(template: ResolvedAgentTemplate) -> None:
    template.ref.digest = _agent_template_digest(template)


class FailingToolsetFactory(RunArtifactsToolsetFactory):
    async def create_selected(self, **_: object) -> dict[str, object]:
        raise RuntimeError("synthetic tool construction failure")


class FailOnceSandbox(LocalWorkdirFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.failures_remaining = 1

    async def cleanup(self, workspace: AllocationWorkspace) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise OSError("synthetic cleanup failure")
        await super().cleanup(workspace)


class FailingStopRuntimeFactory:
    ref = "adk@1"

    async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
        return FailingStopRuntime(context)


class FailingStopRuntime(StubWorkerRuntime):
    async def abort(self, deadline: datetime) -> None:
        raise RuntimeError("synthetic stop failure")
