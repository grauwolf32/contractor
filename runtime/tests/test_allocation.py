from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.spec import allocation_spec

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    AgentTemplateRef,
    AllocationSpec,
    FinalizeAllocationRequest,
    ModelPolicyRef,
    PerformanceMetricsRequest,
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
    WorkerSessionMode,
    encode_private_v2,
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
from contractor_runtime.resource_metrics import ProcessReading, ResourceCollector
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory

NOW = datetime(2026, 8, 29, 10, 0, tzinfo=UTC)
SECRET = "allocation-only-recognizable-secret"


def test_worker_build_context_does_not_expose_agent_template() -> None:
    fields = WorkerBuildContext.__dataclass_fields__

    assert "agent_template" not in fields
    assert "run_metadata_labels" not in fields
    assert {"description", "instruction", "card_version"} <= fields.keys()


def test_prepare_is_single_slot_idempotent_and_constructs_only_selected_tools(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
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

        changed_policy = make_spec()
        changed_policy.model_policy.model = "another-effective-model"
        changed_policy.model_policy.ref.digest = _model_policy_digest(changed_policy.model_policy)
        with pytest.raises(AllocationError, match="differs from the active allocation"):
            await service.prepare(changed_policy)

        changed_labels = make_spec()
        changed_labels.run_metadata_labels["purpose"] = "eval"
        with pytest.raises(AllocationError, match="differs from the active allocation"):
            await service.prepare(changed_labels)

    asyncio.run(scenario())


@pytest.mark.parametrize("namespace", ["inputs", "outputs", "skills"])
def test_prepare_rejects_every_purpose_reserved_agent_namespace(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    namespace: str,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        spec.namespace = namespace

        with pytest.raises(AllocationError) as raised:
            await service.prepare(spec)
        assert raised.value.code == "invalid_agent_namespace"
        assert not raised.value.retryable
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_bad_digest_and_unsupported_ref_leave_no_residue(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        bad_digest = make_spec()
        bad_digest.agent_template.ref.digest = "sha256:" + "0" * 64
        with pytest.raises(AllocationError) as mismatch:
            await service.prepare(bad_digest)
        assert mismatch.value.code == "template_digest_mismatch"
        assert not mismatch.value.retryable
        assert await service.snapshot() is None
        assert list(tmp_path.iterdir()) == []

        bad_policy_digest = make_spec()
        bad_policy_digest.model_policy.ref.digest = "sha256:" + "0" * 64
        with pytest.raises(AllocationError) as policy_mismatch:
            await service.prepare(bad_policy_digest)
        assert policy_mismatch.value.code == "template_digest_mismatch"
        assert not policy_mismatch.value.retryable
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


def test_mutated_run_metadata_labels_fail_before_resource_creation(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        spec.run_metadata_labels = {"Eval.ID": "invalid"}

        with pytest.raises(AllocationError) as failure:
            await service.prepare(spec)
        assert failure.value.code == "invalid_run_metadata_labels"
        assert not failure.value.retryable
        assert await service.snapshot() is None
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_tool_omitted_from_snapshot_is_rejected_before_workspace_creation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        capabilities = CapabilitySnapshot.create(
            runtimes=["adk@1"],
            toolsets={"run-artifacts@1": ["read_artifact"]},
            sandbox_profiles=["local-workdir@1"],
        )
        state, service = await make_service(tmp_path, capabilities)

        with pytest.raises(AllocationError) as unsupported:
            await service.prepare(make_spec(tools=["write_artifact"]))

        assert unsupported.value.code == "unsupported_tool"
        assert await service.snapshot() is None
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_prepare_failure_rolls_back_workspace_tools_and_slot(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
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
            state,
            registry,
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
        )

        with pytest.raises(AllocationError) as failure:
            await service.prepare(make_spec())
        assert failure.value.code == "allocation_preparation_failed"
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert await service.snapshot() is None
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_finalize_report_and_release_erase_context_and_workspace(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
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
        assert first.report.worker.complete
        assert first.report.runtime.complete
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
        assert await service.snapshot() is not None
        assert not workspace.exists()
        state_snapshot = await state.snapshot()
        assert state_snapshot.process_state is ProcessState.FENCED
        assert state_snapshot.allocation_id == spec.allocation_id

        await service.confirm_release(spec.allocation_id)
        await service.confirm_release(spec.allocation_id)
        assert await service.snapshot() is None
        state_snapshot = await state.snapshot()
        assert state_snapshot.process_state is ProcessState.IDLE
        assert state_snapshot.allocation_id is None

    asyncio.run(scenario())


def test_finalize_closes_allocation_tools_before_return_and_release_does_not_repeat(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-terminal-tools")
        await state.mark_registered()
        tool = TrackingTool()
        toolset = TrackingToolsetFactory(tool)
        sandbox = LocalWorkdirFactory(tmp_path)
        service = AllocationService(
            state,
            FactoryRegistry(
                worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
                toolsets={toolset.ref: toolset},
                sandbox_profiles={sandbox.ref: sandbox},
            ),
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
        )
        spec = make_spec(tools=["read_artifact"])
        await service.prepare(spec)

        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-tools",
                deadline=NOW + timedelta(seconds=30),
            )
        )
        assert tool.close_calls == 1
        assert service._context is not None and service._context.tools == {}

        release = ReleaseAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
        )
        await service.release(release)
        assert tool.close_calls == 1

    asyncio.run(scenario())


def test_unconfirmed_terminal_tool_cleanup_fences_slot_and_requests_process_exit(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-failed-tool-cleanup")
        await state.mark_registered()
        tool = TrackingTool(fail=True)
        toolset = TrackingToolsetFactory(tool)
        sandbox = LocalWorkdirFactory(tmp_path)
        exits: list[int] = []
        service = AllocationService(
            state,
            FactoryRegistry(
                worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
                toolsets={toolset.ref: toolset},
                sandbox_profiles={sandbox.ref: sandbox},
            ),
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
            force_exit=exits.append,
        )
        spec = make_spec(tools=["read_artifact"])
        await service.prepare(spec)

        with pytest.raises(AllocationError) as rejected:
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalize-failed-tools",
                    deadline=NOW + timedelta(seconds=30),
                )
            )
        assert rejected.value.code == "tool_cleanup_unconfirmed"
        assert not rejected.value.retryable
        assert tool.close_calls == 1
        assert exits == [70]
        assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())


def test_release_of_unknown_allocation_is_idempotent_only_when_slot_is_empty(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        _, service = await make_service(tmp_path, runtime_capabilities)
        unknown = ReleaseAllocationRequest(
            apiVersion=API_VERSION,
            allocationId="allocation-from-previous-process",
        )

        await service.release(unknown)

        active = make_spec()
        await service.prepare(active)
        with pytest.raises(AllocationError) as conflict:
            await service.release(unknown)
        assert conflict.value.code == "allocation_not_found"
        snapshot = await service.snapshot()
        assert snapshot is not None
        assert snapshot.allocation_id == active.allocation_id

    asyncio.run(scenario())


def test_cleanup_failure_fences_slot_until_idempotent_release_retry(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
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
            state,
            registry,
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
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
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_release_timeout_retains_one_cleanup_task_and_keeps_loop_responsive(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-test")
        await state.mark_registered()
        sandbox = BlockingCleanupSandbox(tmp_path)
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={"run-artifacts@1": RunArtifactsToolsetFactory()},
            sandbox_profiles={"local-workdir@1": sandbox},
        )
        service = AllocationService(
            state,
            registry,
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
        )
        spec = make_spec()
        spec.runtime_settings.request_timeout_seconds = 1
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

        first_release = asyncio.create_task(service.release(release))
        await asyncio.wait_for(sandbox.cleanup_started.wait(), timeout=0.5)

        # The cleanup is deliberately stuck, but unrelated Runtime state work
        # must still be scheduled by the event loop.
        heartbeat = await asyncio.wait_for(state.heartbeat(1, 0), timeout=0.1)
        assert heartbeat.allocation_id == spec.allocation_id

        with pytest.raises(AllocationError) as timed_out:
            await first_release
        assert timed_out.value.code == "allocation_cleanup_failed"
        assert timed_out.value.retryable
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        context = service._context
        assert context is not None
        retained_task = context.release_cleanup_task
        assert retained_task is not None
        assert not retained_task.done()
        assert sandbox.cleanup_attempts == 1

        retry = asyncio.create_task(service.release(release))
        await asyncio.sleep(0)
        assert context.release_cleanup_task is retained_task
        assert sandbox.cleanup_attempts == 1
        sandbox.allow_cleanup.set()
        await asyncio.wait_for(retry, timeout=0.5)
        assert retained_task.done()
        assert context.release_prepared
        assert sandbox.cleanup_attempts == 1
        assert (await state.snapshot()).process_state is ProcessState.FENCED

        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_abort_adds_reason_and_forces_exit_if_worker_cannot_stop(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
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
            runtime_capabilities,
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


async def make_service(
    tmp_path: Path, capabilities: CapabilitySnapshot
) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-test")
    await state.mark_registered()
    factories = built_in_factories(tmp_path)
    service = AllocationService(
        state,
        factories,
        capabilities,
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
        maxModelCalls=8,
        maxToolCalls=16,
        maxTotalTokens=32768,
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
        workerSessionMode=WorkerSessionMode.ISOLATED,
        runMetadataLabels={},
        leaseExpiresAt=NOW + timedelta(seconds=60),
        agentTemplate=template,
        resolvedSkills=[],
        modelPolicy=policy.model_copy(deep=True),
        runtimeSettings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken=SECRET,
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
    )


def resign_template(template: ResolvedAgentTemplate) -> None:
    template.ref.digest = _agent_template_digest(template)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_model_calls", 9),
        ("max_tool_calls", 17),
        ("max_total_tokens", 32769),
    ],
)
def test_each_worker_budget_changes_policy_and_template_digest(field: str, value: int) -> None:
    baseline = make_spec().agent_template
    variant = baseline.model_copy(deep=True)
    original_policy_digest = baseline.model_policy.ref.digest
    original_template_digest = baseline.ref.digest
    setattr(variant.model_policy, field, value)
    variant.model_policy.ref.digest = _model_policy_digest(variant.model_policy)
    resign_template(variant)
    assert variant.model_policy.ref.digest != original_policy_digest
    assert variant.ref.digest != original_template_digest


class FailingToolsetFactory(RunArtifactsToolsetFactory):
    async def create_selected(self, **_: object) -> dict[str, object]:
        raise RuntimeError("synthetic tool construction failure")


class TrackingToolsetFactory(RunArtifactsToolsetFactory):
    def __init__(self, tool: TrackingTool) -> None:
        super().__init__()
        self.tool = tool

    async def create_selected(self, **_: object) -> dict[str, TrackingTool]:
        return {self.tool.name: self.tool}


class TrackingTool:
    name = "read_artifact"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.fail:
            raise RuntimeError("synthetic tool cleanup failure")


class FailOnceSandbox(LocalWorkdirFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.failures_remaining = 1

    async def cleanup(self, workspace: AllocationWorkspace) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise OSError("synthetic cleanup failure")
        await super().cleanup(workspace)


class BlockingCleanupSandbox(LocalWorkdirFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.cleanup_attempts = 0

    async def cleanup(self, workspace: AllocationWorkspace) -> None:
        self.cleanup_attempts += 1
        self.cleanup_started.set()
        await self.allow_cleanup.wait()
        await super().cleanup(workspace)


class FailingStopRuntimeFactory:
    ref = "adk@1"

    async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
        return FailingStopRuntime(context)


class FailingStopRuntime(StubWorkerRuntime):
    async def abort(self, deadline: datetime) -> None:
        raise RuntimeError("synthetic stop failure")


class ResourceProbe:
    def __init__(self) -> None:
        self.now = 100.0
        self.rss = 1000
        self.reads: list[bool] = []
        self.collectors: list[ResourceCollector] = []

    def read(self, boundary: bool) -> ProcessReading:
        self.reads.append(boundary)
        return ProcessReading(self.now, self.now / 2, self.rss)

    def factory(self) -> ResourceCollector:
        collector = ResourceCollector(clock=lambda: self.now, reader=self.read)
        self.collectors.append(collector)
        return collector


def test_resource_collection_disabled_never_constructs_sampler(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        _, service = await make_service(tmp_path, runtime_capabilities)

        def forbidden() -> ResourceCollector:
            pytest.fail("disabled metrics must not construct a sampler or perform reads")

        service._resource_collector_factory = forbidden
        await asyncio.sleep(0)
        await service.prepare(allocation_spec())
        result = await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId="allocation-1",
                finalizationId="final-1",
                deadline=NOW + timedelta(seconds=30),
            )
        )
        assert result.report.runtime.resources is None
        assert result.report.runtime.resources_error is None
        assert b'"resources"' not in encode_private_v2(result)
        await service.release(
            ReleaseAllocationRequest(
                apiVersion=API_VERSION,
                allocationId="allocation-1",
            )
        )
        await service.confirm_release("allocation-1")
        await asyncio.sleep(0)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["finalize", "abort", "lease", "drain"])
def test_resources_cover_prepare_and_teardown_freeze_before_release_and_reset(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    async def scenario() -> None:
        _, service = await make_service(tmp_path, runtime_capabilities)
        probe = ResourceProbe()
        service._resource_collector_factory = probe.factory
        sandbox = service._factories.sandbox_profiles["local-workdir@1"]
        original_prepare = sandbox.prepare

        async def prepare() -> AllocationWorkspace:
            assert probe.reads[-1] is True  # Boundary precedes expensive preparation.
            probe.now += 2
            return await original_prepare()

        monkeypatch.setattr(sandbox, "prepare", prepare)
        for index in range(2):
            spec = allocation_spec(allocation_id=f"allocation-{index}")
            spec.performance_metrics = PerformanceMetricsRequest(version=1, intervalSeconds=15)
            probe.rss = 1000 if index == 0 else 100
            await service.prepare(spec)
            await service.prepare(spec)  # Retry must not start a second sampler.
            context = service._context
            assert context is not None and context.worker is not None
            worker = context.worker

            async def stop(deadline: datetime, current: StubWorkerRuntime = worker) -> None:
                probe.now += 3
                current.stopped = True

            async def adapters(**_: object) -> None:
                probe.now += 4
                probe.rss += 10

            monkeypatch.setattr(worker, "finalize", stop)
            monkeypatch.setattr(worker, "abort", stop)
            monkeypatch.setattr(context.adapter_host, "terminate", adapters)
            final = FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId=f"final-{index}",
                deadline=NOW + timedelta(seconds=30),
            )
            abort = AbortAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                abortId=f"abort-{index}",
                deadline=NOW + timedelta(seconds=30),
                reason=TerminationError(code="run_cancelled", message="cancelled", retryable=False),
            )
            if operation == "lease":
                await service.expire_control_lease(30)
            elif operation == "drain":
                await service.reconcile_drain(spec.allocation_id, 30)
            if operation == "abort":
                result = await service.abort(abort)
            else:
                result = await service.finalize(final)
            resources = result.report.runtime.resources
            assert resources is not None and resources.status == "complete"
            assert result.report.runtime.complete
            assert resources.duration_seconds == 9
            assert resources.cpu_user_seconds == 9 and resources.cpu_system_seconds == 4.5
            assert resources.rss_start_bytes == (1000 if index == 0 else 100)
            assert resources.rss_peak_observed_bytes == (1010 if index == 0 else 110)
            assert resources.rss_sample_count == 2
            encoded = encode_private_v2(result)
            assert len(encoded) < 1024 * 1024
            probe.now += 500  # Finalized, idle and release time must not extend coverage.
            retry = (
                await service.abort(abort)
                if operation == "abort"
                else await service.finalize(final)
            )
            assert encode_private_v2(retry) == encoded
            await service.release(
                ReleaseAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                )
            )
            await service.confirm_release(spec.allocation_id)
            await asyncio.sleep(0)
            assert probe.collectors[-1]._task.done()
            assert probe.reads == [True] * (2 * (index + 1))
        assert len(probe.collectors) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_resource_sampler_discarded_after_failed_or_cancelled_prepare(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        probe = ResourceProbe()
        service._resource_collector_factory = probe.factory
        entered = asyncio.Event()

        async def fail(*args: object, **kwargs: object) -> None:
            entered.set()
            if cancel:
                await asyncio.Event().wait()
            raise RuntimeError("synthetic prepare failure")

        monkeypatch.setattr(service, "_create_tools", fail)
        spec = allocation_spec()
        spec.performance_metrics = PerformanceMetricsRequest(version=1, intervalSeconds=15)
        pending = asyncio.create_task(service.prepare(spec))
        await entered.wait()
        if cancel:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else AllocationError):
            await pending
        await asyncio.sleep(0)
        assert probe.reads == [True]
        assert probe.collectors[0]._task.done()
        assert service._context is None
        assert list(tmp_path.iterdir()) == []
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["worker", "adapters", "cancel", "deadline"])
def test_unconfirmed_stop_discards_resources_without_recovered_final_report(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        probe = ResourceProbe()
        service._resource_collector_factory = probe.factory
        spec = allocation_spec()
        spec.performance_metrics = PerformanceMetricsRequest(version=1, intervalSeconds=15)
        await service.prepare(spec)
        context = service._context
        assert context is not None
        entered = asyncio.Event()

        async def fail(*args: object, **kwargs: object) -> None:
            entered.set()
            if failure == "cancel":
                await asyncio.Event().wait()
            raise RuntimeError("synthetic stop failure")

        if failure == "adapters":
            monkeypatch.setattr(context.adapter_host, "terminate", fail)
        else:
            monkeypatch.setattr(context.worker, "finalize", fail)
        request = FinalizeAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
            finalizationId="final-1",
            deadline=NOW + timedelta(seconds=-1 if failure == "deadline" else 30),
        )
        pending = asyncio.create_task(service.finalize(request))
        if failure == "cancel":
            await entered.wait()
            pending.cancel()
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else AllocationError):
            await pending
        await asyncio.sleep(0)
        assert probe.reads == [True]
        assert probe.collectors[0]._task.done()
        assert context.terminal_response is None
        assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["read", "factory"])
def test_resource_failure_does_not_change_worker_completeness_or_slot_reuse(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    failure: str,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)

        def unavailable(_: bool) -> ProcessReading:
            raise OSError("secret-canary")

        def factory() -> ResourceCollector:
            if failure == "factory":
                raise OSError("secret-canary")
            return ResourceCollector(reader=unavailable)

        service._resource_collector_factory = factory
        spec = allocation_spec()
        spec.performance_metrics = PerformanceMetricsRequest(version=1, intervalSeconds=15)
        await service.prepare(spec)
        response = await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="final-1",
                deadline=NOW + timedelta(seconds=30),
            )
        )
        assert response.report.worker.complete and response.report.runtime.complete
        assert response.report.runtime.resources.status == "unavailable"
        assert response.report.runtime.resources.reason == "read_failed"
        assert b"secret-canary" not in encode_private_v2(response)
        await service.release(
            ReleaseAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
            )
        )
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())
