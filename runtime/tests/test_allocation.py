from __future__ import annotations

import asyncio
import gc
import json
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock
from fakes.model import scripted_model, text_result, tool_call
from fakes.private_codec import encode_private
from fakes.spec import allocation_spec, runtime_provenance
from fakes.worker_runtime import StubADKWorkerRuntimeFactory, StubWorkerRuntime
from google.adk.tools import FunctionTool
from target_policy_fixtures import SCAN_TEST_POLICY
from test_adk_runtime import CapturingScriptedLlm, FakeArtifactClient
from test_adk_runtime import stage_request as adk_stage_request
from test_http_caido_process_e2e import MockCaidoAdapterFactory, MockHTTPProxyAdapterFactory
from test_text_artifacts_toolset import MemoryArtifactClient

import contractor_runtime.allocation.cleanup as allocation_cleanup_module
import contractor_runtime.worker.adk_caches as adk_caches
import contractor_runtime.worker.state as worker_state_module
from contractor_runtime.adapters import AdapterHandles, AdapterPreparationError
from contractor_runtime.adapters.host import RuntimeAdapterBuildContext, RuntimeAdapterMetricsState
from contractor_runtime.adapters.interactsh import InteractshClient
from contractor_runtime.agent_skills.runtime import AgentSkillPreparationError
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.allocation.preparation import AllocationPreparation
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    AgentTemplateRef,
    AllocationFinalResponse,
    AllocationSpec,
    CaidoSettings,
    FinalizeAllocationRequest,
    HTTPProxySettings,
    InteractshSettings,
    ModelPolicyRef,
    NoWorkerContextManagement,
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
    decode_private,
)
from contractor_runtime.core import target_policy as target_policy_module
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.digests import (
    _agent_template_digest,
    _model_policy_digest,
    sha256_digest,
    verify_template_digests,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    RunArtifactsToolsetFactory,
    WorkerBuildContext,
    built_in_factories,
)
from contractor_runtime.projectfs import WorkspacePreparationError
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.telemetry.metrics import (
    MAX_METRIC_ERRORS,
    MAX_METRIC_TOOL_CALLS,
    MAX_REPORT_JSON_BYTES,
    MetricsState,
)
from contractor_runtime.telemetry.resources import ProcessReading, ResourceCollector
from contractor_runtime.toolsets.http.errors import HTTPToolError
from contractor_runtime.toolsets.security_findings.collection import FindingsError
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory

NOW = datetime(2026, 8, 29, 10, 0, tzinfo=UTC)
SECRET = "allocation-only-recognizable-secret"


def test_worker_build_context_does_not_expose_agent_template() -> None:
    fields = WorkerBuildContext.__dataclass_fields__

    assert "agent_template" not in fields
    assert "run_metadata_labels" not in fields
    assert {"description", "instruction", "card_version"} <= fields.keys()


def test_prepare_uses_pinned_summary_model_without_changing_template(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    contexts: list[WorkerBuildContext] = []

    class CapturingFactory(StubADKWorkerRuntimeFactory):
        async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
            contexts.append(context)
            return StubWorkerRuntime(context)

    async def scenario() -> None:
        state = RuntimeState(instance_id="summary-selection-test")
        await state.mark_registered()
        registry = replace(
            built_in_factories(tmp_path), worker_runtimes={"adk@1": CapturingFactory()}
        )
        service = AllocationService(
            state,
            registry,
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
        )
        spec = allocation_spec(summarizer=True)
        assert spec.agent_template.summarizer is not None
        original = spec.agent_template.model_dump_json(by_alias=True, exclude_none=True)
        assert spec.summarizer_model_policy is not None
        policy = spec.summarizer_model_policy.model_copy(deep=True)
        policy.ref.policy_id = "selected_summary"
        policy.model = "fixture/selected-summary-model"
        policy.context_window_tokens = 65536
        policy.ref.digest = _model_policy_digest(policy)
        spec.summarizer_model_policy = policy
        await service.prepare(spec)
        assert len(contexts) == 1
        assert contexts[0].summarizer is not None
        assert contexts[0].summarizer.model_policy == policy
        assert (
            contexts[0].summarizer.context_window_ratio
            == spec.agent_template.summarizer.context_window_ratio
        )
        assert spec.agent_template.model_dump_json(by_alias=True, exclude_none=True) == original
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="summary-model-final",
                deadline=NOW + timedelta(seconds=30),
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )

    asyncio.run(scenario())


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
        renewed = spec.model_copy(
            update={"lease_expires_at": spec.lease_expires_at + timedelta(minutes=2)}
        )
        assert await service.prepare(renewed) == first
        with pytest.raises(AllocationError, match="differs from the active allocation"):
            await service.prepare(
                spec.model_copy(
                    update={"lease_expires_at": spec.lease_expires_at - timedelta(seconds=1)}
                )
            )
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


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (FindingsError("findings_document_conflict"), 422),
        (FindingsError("findings_collection_unavailable", retryable=True), 503),
        (HTTPToolError("http_cookie_state_invalid"), 422),
        (HTTPToolError("http_cookie_storage_unavailable"), 503),
    ],
    ids=["conflict", "transient", "cookie-invalid", "cookie-storage"],
)
def test_prepare_keeps_findings_error_code_and_retryability(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    error: FindingsError | HTTPToolError,
    status_code: int,
) -> None:
    class FindingsToolsetFactory(RunArtifactsToolsetFactory):
        async def create_selected(self, **_: object) -> dict[str, object]:
            raise error

    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-test")
        await state.mark_registered()
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={"run-artifacts@1": FindingsToolsetFactory()},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
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
        assert failure.value.payload() == {
            "code": error.code,
            "message": (
                f"allocation findings preparation failed ({error.code})"
                if isinstance(error, FindingsError)
                else f"allocation HTTP preparation failed ({error.code})"
            ),
            "retryable": error.retryable,
        }
        assert failure.value.status_code == status_code
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert await service.snapshot() is None
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error",
    [
        AdapterPreparationError(retryable=False, cleanup_confirmed=True),
        AdapterPreparationError(retryable=True, cleanup_confirmed=False),
        AgentSkillPreparationError("skill_unavailable", retryable=True, status_code=503),
        WorkspacePreparationError("workspace_invalid", retryable=False, status_code=422),
        WorkspacePreparationError(
            "workspace_unavailable", retryable=True, status_code=503, cleanup_confirmed=False
        ),
        AllocationError(
            "allocation_conflict", "original message", retryable=False, status_code=409
        ),
        asyncio.CancelledError("original cancellation"),
        RuntimeError("private failure"),
    ],
)
def test_failed_prepare_rolls_back_once_before_classifying(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        exits: list[int] = []
        service._force_exit = exits.append
        rollbacks = []
        rollback = service._rollback_prepare

        async def fail(*args: object, **kwargs: object) -> None:
            raise error

        async def track_rollback(*args: object, **kwargs: object) -> None:
            rollbacks.append("rollback")
            await rollback(*args, **kwargs)

        monkeypatch.setattr(AllocationPreparation, "_create_tools", fail)
        monkeypatch.setattr(service, "_rollback_prepare", track_rollback)
        expected = (
            asyncio.CancelledError if isinstance(error, asyncio.CancelledError) else AllocationError
        )
        with pytest.raises(expected) as failure:
            await service.prepare(make_spec())
        assert rollbacks == ["rollback"]
        assert list(tmp_path.iterdir()) == []
        assert await service.snapshot() is None
        fenced = (
            isinstance(error, (AdapterPreparationError, WorkspacePreparationError))
            and not error.cleanup_confirmed
        )
        assert exits == ([70] if fenced else [])
        assert ((await state.snapshot()).process_state is ProcessState.IDLE) is not fenced
        if isinstance(error, (AllocationError, asyncio.CancelledError)):
            assert failure.value is error
        else:
            if isinstance(error, AdapterPreparationError):
                code, message = (
                    "runtime_adapter_prepare_failed",
                    "allocation Runtime adapter preparation failed",
                )
            elif isinstance(error, AgentSkillPreparationError):
                code, message = (
                    error.code,
                    f"allocation Agent Skill preparation failed ({error.code})",
                )
            elif isinstance(error, WorkspacePreparationError):
                code, message = (
                    error.code,
                    f"allocation workspace preparation failed ({error.code})",
                )
            else:
                code, message = (
                    "allocation_preparation_failed",
                    "allocation resource preparation failed (RuntimeError)",
                )
            assert failure.value.payload() == {
                "code": code,
                "message": message,
                "retryable": getattr(error, "retryable", True),
            }
            assert failure.value.status_code == getattr(error, "status_code", 503)

    asyncio.run(scenario())


def test_prepare_does_not_translate_rollback_failure(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        _, service = await make_service(tmp_path, runtime_capabilities)
        cleanup_error = RuntimeError("rollback failed")

        async def fail_prepare(*args: object, **kwargs: object) -> None:
            raise ValueError("prepare failed")

        async def fail_rollback(*args: object, **kwargs: object) -> None:
            raise cleanup_error

        monkeypatch.setattr(AllocationPreparation, "_create_tools", fail_prepare)
        monkeypatch.setattr(service, "_rollback_prepare", fail_rollback)
        with pytest.raises(RuntimeError) as failure:
            await service.prepare(make_spec())
        assert failure.value is cleanup_error

    asyncio.run(scenario())


def test_prepare_recovery_deadline_uses_the_monotonic_clock(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    class SkewedClockLoop(asyncio.SelectorEventLoop):
        def time(self) -> float:
            return super().time() + 1_000_000

    class RecordingRecovery:
        def __init__(self) -> None:
            self.deadlines: list[float] = []

        async def recover(self, *, deadline: float) -> None:
            self.deadlines.append(deadline)

    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-test")
        await state.mark_registered()
        lifecycle = RecordingRecovery()
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={"run-artifacts@1": RunArtifactsToolsetFactory()},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
            execution_lifecycle=lifecycle,  # type: ignore[arg-type]
        )
        service = AllocationService(
            state,
            registry,
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
            force_exit=lambda _: None,
        )
        spec = make_spec()
        before = time.monotonic()
        await service.prepare(spec)
        timeout = min(60, spec.runtime_settings.request_timeout_seconds)
        assert lifecycle.deadlines
        assert before < lifecycle.deadlines[0] <= time.monotonic() + timeout

    asyncio.run(scenario(), loop_factory=SkewedClockLoop)


@pytest.mark.parametrize("failure_kind", ["exception", "cancelled", "wrong_set", "wrong_name"])
def test_partial_tool_preparation_retains_cleanup_ownership(
    tmp_path: Path, failure_kind: str
) -> None:
    async def scenario() -> None:
        first = TrackingTool()
        second = TrackingTool()

        class SecondToolsetFactory(RunArtifactsToolsetFactory):
            ref = "second@1"
            exported_tools = frozenset({"write_artifact"})

            async def create_selected(self, **_: object) -> dict[str, TrackingTool]:
                if failure_kind == "exception":
                    raise RuntimeError("second Toolset preparation failed")
                if failure_kind == "cancelled":
                    raise asyncio.CancelledError
                if failure_kind == "wrong_set":
                    # Even a malformed return must retain both owners when
                    # the invalid name collides with an earlier Toolset.
                    return {"read_artifact": second}
                return {"write_artifact": second}

        state = RuntimeState(instance_id="runtime-partial-tools")
        await state.mark_registered()
        factories = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={
                "run-artifacts@1": TrackingToolsetFactory(first),
                "second@1": SecondToolsetFactory(),
            },
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=factories.worker_runtimes,
            toolsets={ref: factory.exported_tools for ref, factory in factories.toolsets.items()},
            sandbox_profiles=factories.sandbox_profiles,
        )
        exit_codes: list[int] = []
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
            force_exit=exit_codes.append,
        )
        spec = make_spec(tools=["read_artifact"])
        spec.agent_template.toolsets.append(
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="second", version="1"), tools=["write_artifact"]
            )
        )
        resign_template(spec.agent_template)

        expected = asyncio.CancelledError if failure_kind == "cancelled" else AllocationError
        with pytest.raises(expected):
            await service.prepare(spec)

        assert first.close_calls == 1
        assert second.close_calls == (1 if failure_kind in {"wrong_set", "wrong_name"} else 0)
        assert exit_codes == []
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


def test_maximum_worker_detail_crosses_the_private_report_boundary(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    """The Runtime trims a full Worker report before it reaches the 1 MiB wire bound."""

    class LongInputError(ToolInputError):
        pass

    async def scenario() -> None:
        _, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        await service.prepare(spec)
        assert service._context is not None and service._context.worker_state is not None
        metrics = service._context.worker_state.metrics
        message = "界" * 1365 + "a"  # 4096 UTF-8 bytes, the error-message bound
        for index in range(MAX_METRIC_TOOL_CALLS):
            failed = index >= MAX_METRIC_TOOL_CALLS - MAX_METRIC_ERRORS
            metrics.record_tool_call(
                "read_source_file",
                arguments={"path": f"{index:04d}" + "ж" * 1990},
                error=LongInputError(message) if failed else None,
            )
        reason = TerminationError(code="run_cancelled", message=message, retryable=False)
        response = await service.abort(
            AbortAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                abortId="abort-maximum-detail",
                reason=reason,
                deadline=NOW + timedelta(seconds=30),
            )
        )

        # The body the control endpoint sends (Starlette's compact JSONResponse).
        body = json.dumps(
            response.model_dump(mode="json", by_alias=True, exclude_none=True),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        assert len(body) <= MAX_REPORT_JSON_BYTES
        decoded = decode_private(AllocationFinalResponse, body)
        assert decoded == response
        worker = decoded.report.worker
        retained = list(metrics.tool_calls)
        removed = len(retained) - len(worker.tool_calls)
        assert 0 < removed < MAX_METRIC_TOOL_CALLS
        assert worker.tool_calls == retained[removed:]
        assert worker.truncated
        # The termination reason is the newest error; the oldest one gave way.
        assert len(worker.errors) == MAX_METRIC_ERRORS
        assert worker.errors[-1].code == "run_cancelled"
        assert worker.errors[:-1] == list(metrics.errors)[1:]
        assert worker.metrics.tools["read_source_file"].calls == MAX_METRIC_TOOL_CALLS
        assert worker.metrics.tools["read_source_file"].failed == MAX_METRIC_ERRORS

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
    # Prepare and the retried release do real directory I/O inside the request
    # budget, so it is real time no loaded host exhausts. The test expires it
    # by moving the loop clock while the cleanup is stuck; the safety net only
    # ends a broken scenario.
    request_timeout_seconds = 60
    safety_net_seconds = 30.0

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
        spec.runtime_settings.request_timeout_seconds = request_timeout_seconds
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
        await asyncio.wait_for(sandbox.cleanup_started.wait(), timeout=safety_net_seconds)

        # The cleanup is deliberately stuck, but unrelated Runtime state work
        # must still be scheduled by the event loop.
        heartbeat = await asyncio.wait_for(state.heartbeat(1, 0), timeout=0.1)
        assert heartbeat.allocation_id == spec.allocation_id

        # The stuck cleanup outlives the release request budget.
        assert not first_release.done()
        advance_clock(request_timeout_seconds)
        with pytest.raises(AllocationError) as timed_out:
            await asyncio.wait_for(first_release, safety_net_seconds)
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
        await asyncio.wait_for(retry, timeout=safety_net_seconds)
        assert retained_task.done()
        assert context.release_prepared
        assert sandbox.cleanup_attempts == 1
        assert (await state.snapshot()).process_state is ProcessState.FENCED

        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    run_with_advanceable_clock(scenario())


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
    tmp_path: Path,
    capabilities: CapabilitySnapshot,
    *,
    artifact_client_factory: Callable[[str, RuntimeSettings], Any] | None = None,
    model_factory: Callable[[WorkerBuildContext], Any] | None = None,
) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-test")
    await state.mark_registered()
    factories = built_in_factories(
        tmp_path,
        artifact_client_factory=artifact_client_factory,
        model_factory=model_factory,
    )
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
            digest=sha256_digest(instructions_text.encode()),
            text=instructions_text,
        ),
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
        contextManagement=NoWorkerContextManagement(strategy="none"),
        runMetadataLabels={},
        leaseExpiresAt=NOW + timedelta(seconds=60),
        agentTemplate=template,
        resolvedSkills=[],
        modelPolicy=policy.model_copy(deep=True),
        resolvedRuntimeConfigProvenance=runtime_provenance(),
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
def test_each_worker_budget_changes_policy_without_changing_template_digest(
    field: str, value: int
) -> None:
    baseline = make_spec()
    variant = baseline.model_copy(deep=True)
    setattr(variant.model_policy, field, value)
    variant.model_policy.ref.digest = _model_policy_digest(variant.model_policy)
    assert variant.model_policy.ref.digest != baseline.model_policy.ref.digest
    assert variant.agent_template == baseline.agent_template
    verify_template_digests(variant.agent_template)


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
        assert b'"resources"' not in encode_private(result)
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
            encoded = encode_private(result)
            assert len(encoded) < 1024 * 1024
            probe.now += 500  # Finalized, idle and release time must not extend coverage.
            retry = (
                await service.abort(abort)
                if operation == "abort"
                else await service.finalize(final)
            )
            assert encode_private(retry) == encoded
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

        monkeypatch.setattr(AllocationPreparation, "_create_tools", fail)
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
        assert b"secret-canary" not in encode_private(response)
        await service.release(
            ReleaseAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
            )
        )
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


class TrackedMetricsState(MetricsState):
    """An allocation metrics reducer whose lifetime a test can observe."""

    __slots__ = ("__weakref__",)


class TrackedArtifactClients:
    """Creates one FakeArtifactClient per call and keeps only weak references to them."""

    def __init__(self) -> None:
        self.created: list[weakref.ref[FakeArtifactClient]] = []

    def __call__(self, allocation_id: str, settings: RuntimeSettings) -> FakeArtifactClient:
        del allocation_id, settings
        client = FakeArtifactClient()
        self.created.append(weakref.ref(client))
        return client

    def live(self) -> int:
        gc.collect()
        return sum(ref() is not None for ref in self.created)


def read_write_script() -> object:
    return scripted_model(
        [
            tool_call(
                "read_artifact",
                {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                call_id="read-1",
            ),
            tool_call(
                "write_artifact",
                {
                    "namespace": "builder",
                    "name": "report",
                    "media_type": "application/json",
                    "data_base64": "e30=",
                    "expected_revision": None,
                },
                call_id="write-1",
            ),
            text_result("Report created"),
        ]
    )


def allocation_references(service: AllocationService) -> dict[str, weakref.ref[Any]]:
    """Weak references to everything release must make unreachable (LIFE-WRK-3)."""

    context = service._context
    if context is None or context.worker is None or context.worker_state is None:
        raise AssertionError("allocation is not prepared")
    worker: Any = context.worker
    references: dict[str, weakref.ref[Any]] = {
        "worker": weakref.ref(worker),
        "worker_state": weakref.ref(context.worker_state),
        "metrics": weakref.ref(context.worker_state.metrics),
    }
    for name, tool in context.tools.items():
        references[f"tool:{name}"] = weakref.ref(tool)
    for tool in worker._agent.tools:
        if isinstance(tool, FunctionTool):
            references[f"adk:{tool.name}"] = weakref.ref(tool)
    return references


def live_references(references: dict[str, weakref.ref[Any]]) -> list[str]:
    gc.collect()
    return sorted(name for name, reference in references.items() if reference() is not None)


async def finalize_and_release(service: AllocationService, allocation_id: str) -> None:
    await service.finalize(
        FinalizeAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=allocation_id,
            finalizationId=f"finalize-{allocation_id}",
            deadline=NOW + timedelta(seconds=30),
        )
    )
    await service.release(
        ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=allocation_id)
    )


class CallableProbe:
    """Return a fixed result."""

    async def __call__(self) -> dict[str, bool]:
        return {"ok": True}


def wrapped_probe(*, declare: bool) -> weakref.ref[CallableProbe]:
    """Wrap a fresh callable in FunctionTool, drop the tool and keep a weak reference."""

    probe = CallableProbe()
    adk_tool = FunctionTool(probe)
    if declare:
        assert adk_tool._get_declaration() is not None
    return weakref.ref(probe)


def test_adk_tool_caches_keep_wrapped_callables_until_released() -> None:
    """Guards the google-adk behaviour that RTA-R116 works around.

    Each cache alone keeps a wrapped callable alive. If an ADK upgrade keys
    either cache weakly or per instance, an assertion that the callable is
    still alive fails: re-check the reversal condition of RTA-R116.
    """

    find_context_parameter, build_declaration, type_hints = adk_caches._TOOL_CALLABLE_CACHES
    assert build_declaration.__name__ == "_build_declaration_cached"
    assert type_hints.__name__ == "_get_type_hints_for_unwrapped"

    constructed = wrapped_probe(declare=False)
    gc.collect()
    assert constructed() is not None
    find_context_parameter.cache_clear()
    gc.collect()
    assert constructed() is None

    declared = wrapped_probe(declare=True)
    find_context_parameter.cache_clear()
    gc.collect()
    assert declared() is not None
    adk_caches.release_adk_tool_caches()
    gc.collect()
    assert declared() is None


def test_adk_type_hint_cache_retains_an_allocation_captured_by_a_tool_function() -> None:
    """An unwrapped function can retain a whole allocation through its closure."""

    def declare() -> weakref.ref[CallableProbe]:
        allocation = CallableProbe()

        async def allocation_tool() -> dict[str, bool]:
            """Use the allocation-owned fixture."""
            return await allocation()

        assert FunctionTool(allocation_tool)._get_declaration() is not None
        return weakref.ref(allocation)

    reference = declare()
    find_context_parameter, build_declaration, type_hints = adk_caches._TOOL_CALLABLE_CACHES
    find_context_parameter.cache_clear()
    build_declaration.cache_clear()
    gc.collect()
    assert reference() is not None
    type_hints.cache_clear()
    gc.collect()
    assert reference() is None


def test_release_collects_allocation_graph_captured_by_declared_tool_closure(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real release clears the new hints cache without test-side cache eviction."""

    monkeypatch.setattr(worker_state_module, "MetricsState", TrackedMetricsState)
    clients = TrackedArtifactClients()

    class ClosureFactory(AdkWorkerRuntimeFactory):
        async def create(self, context: WorkerBuildContext) -> Any:
            original = context.tools["read_artifact"]

            async def read_artifact(
                namespace: str,
                name: str,
                revision: str | None = None,
                offset: int = 0,
                length: int = 262144,
            ) -> dict[str, Any]:
                return await original(namespace, name, revision, offset, length)

            read_artifact.__doc__ = original.description
            return await super().create(
                replace(context, tools={**context.tools, "read_artifact": read_artifact})
            )

    async def scenario() -> None:
        state = RuntimeState(instance_id="closure-release-test")
        await state.mark_registered()
        factories = built_in_factories(
            tmp_path,
            artifact_client_factory=clients,
            model_factory=lambda _: read_write_script(),
        )
        factories = replace(
            factories,
            worker_runtimes={
                "adk@1": ClosureFactory(
                    model_factory=lambda _: read_write_script(), artifact_client_factory=clients
                )
            },
        )
        service = AllocationService(
            state,
            factories,
            runtime_capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
            force_exit=lambda _: None,
        )
        spec = make_spec()
        await service.prepare(spec)
        references = allocation_references(service)
        assert service._context is not None and service._context.worker is not None
        completion = await service._context.worker.invoke(adk_stage_request())
        assert completion.failure is None and completion.result is not None
        assert completion.result.artifacts["report"].revision == "write-r1"
        del completion
        assert clients.live() == 1

        await finalize_and_release(service, spec.allocation_id)

        assert live_references(references) == []
        assert clients.live() == 0

    asyncio.run(scenario())


def test_release_leaves_no_allocation_tool_reachable_after_an_invocation(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release erases tools, their client and MetricsState with no test-side cleanup."""

    monkeypatch.setattr(worker_state_module, "MetricsState", TrackedMetricsState)
    clients = TrackedArtifactClients()

    async def scenario() -> None:
        _, service = await make_service(
            tmp_path,
            runtime_capabilities,
            artifact_client_factory=clients,
            model_factory=lambda _: read_write_script(),
        )
        spec = make_spec()
        await service.prepare(spec)
        references = allocation_references(service)
        assert {"tool:read_artifact", "adk:read_artifact", "tool:write_artifact"} <= set(references)
        assert service._context is not None and service._context.worker is not None
        completion = await service._context.worker.invoke(adk_stage_request())
        assert completion.failure is None and completion.result is not None
        assert completion.result.artifacts["report"].revision == "write-r1"
        del completion
        assert clients.live() == 1

        await finalize_and_release(service, spec.allocation_id)

        assert live_references(references) == []
        assert clients.live() == 0

    asyncio.run(scenario())


def test_failed_cache_release_keeps_the_slot_fenced_until_the_retry_repeats_it(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing ADK's caches is a release cleanup step (RTA-R116, FAIL-RTA-5)."""

    monkeypatch.setattr(worker_state_module, "MetricsState", TrackedMetricsState)
    clients = TrackedArtifactClients()
    attempts: list[bool] = []

    def release_once_failing() -> None:
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("synthetic cache release failure")
        adk_caches.release_adk_tool_caches()

    monkeypatch.setattr(allocation_cleanup_module, "release_adk_tool_caches", release_once_failing)

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_capabilities,
            artifact_client_factory=clients,
            model_factory=lambda _: read_write_script(),
        )
        spec = make_spec()
        await service.prepare(spec)
        references = allocation_references(service)
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-cache-failure",
                deadline=NOW + timedelta(seconds=30),
            )
        )
        release = ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)

        with pytest.raises(AllocationError) as cleanup:
            await service.release(release)
        assert cleanup.value.code == "allocation_cleanup_failed"
        assert cleanup.value.retryable
        del cleanup
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        snapshot = await service.snapshot()
        # State and secrets were already dropped; only the cache step failed.
        assert snapshot is not None and not snapshot.has_runtime_settings
        assert service._context is not None and service._context.worker_state is None
        assert not service._context.release_prepared
        assert "tool:read_artifact" in live_references(references)
        with pytest.raises(AllocationError):
            await service.prepare(make_spec(allocation_id="allocation-new"))

        await service.release(release)
        assert len(attempts) == 2
        assert live_references(references) == []
        assert clients.live() == 0
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_failed_prepare_rollback_leaves_no_allocation_tool_reachable(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    """A prepare that fails after its Worker is built never reaches release."""

    clients = TrackedArtifactClients()

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_capabilities,
            artifact_client_factory=clients,
            model_factory=lambda _: read_write_script(),
        )

        async def lost_commit(allocation_id: str) -> None:
            raise RuntimeError(f"synthetic commit failure for {allocation_id}")

        state.commit_allocation = lost_commit  # type: ignore[method-assign]
        with pytest.raises(AllocationError) as failure:
            await service.prepare(make_spec())
        assert failure.value.code == "allocation_preparation_failed"
        # The suppressed context holds the prepare frame and its locals.
        del failure
        assert await service.snapshot() is None
        assert len(clients.created) == 1
        assert clients.live() == 0

    asyncio.run(scenario())


def test_next_allocation_rebuilds_identical_tool_declarations(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    """Clearing the caches at release costs one rebuild, never different declarations."""

    models: list[CapturingScriptedLlm] = []
    clients = TrackedArtifactClients()

    def model(_: WorkerBuildContext) -> CapturingScriptedLlm:
        scripted = CapturingScriptedLlm(
            model="worker-model",
            responses=[
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                    call_id="read-1",
                ),
                text_result("Read the source"),
            ],
        )
        models.append(scripted)
        return scripted

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_capabilities,
            artifact_client_factory=clients,
            model_factory=model,
        )
        for allocation_id in ("allocation-1", "allocation-2"):
            await service.prepare(make_spec(allocation_id=allocation_id))
            assert service._context is not None and service._context.worker is not None
            completion = await service._context.worker.invoke(adk_stage_request())
            assert completion.failure is None and completion.result is not None
            # The model's read_artifact call ran this allocation's tool.
            assert {
                name: value.model_dump()
                for name, value in completion.result.observations.tools.items()
            } == {"read_artifact": {"calls": 1, "failures": 0}}
            del completion
            await finalize_and_release(service, allocation_id)
            await service.confirm_release(allocation_id)
            assert (await state.snapshot()).process_state is ProcessState.IDLE

        assert len(clients.created) == 2
        assert clients.live() == 0
        # The first allocation's tools were new callables, so ADK built their
        # declarations from scratch as in a fresh process; the second rebuilt
        # them after release had cleared the caches.
        first, second = (scripted.payloads[0]["tools"] for scripted in models)
        assert [tool["function"]["name"] for tool in first] == ["read_artifact", "write_artifact"]
        assert second == first

    asyncio.run(scenario())


class MockInteractshAdapterFactory:
    """interactsh-http@1 with the production client over a deterministic transport."""

    ref = "interactsh-http@1"

    def __init__(self) -> None:
        self.paths: list[str] = []

    async def probe(self) -> bool:
        return True

    async def create(
        self, context: RuntimeAdapterBuildContext, settings: InteractshSettings
    ) -> MockInteractshAdapter:
        del context
        return MockInteractshAdapter(settings, httpx.MockTransport(self._handle))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        message = (
            "registration successful"
            if request.url.path == "/register"
            else "deregistration successful"
        )
        return httpx.Response(200, json={"message": message}, request=request)


class MockInteractshAdapter:
    ref = "interactsh-http@1"

    def __init__(self, settings: InteractshSettings, transport: httpx.MockTransport) -> None:
        self.metrics = RuntimeAdapterMetricsState()
        self._client = InteractshClient(
            settings, timeout_seconds=5, metrics=self.metrics, transport=transport
        )
        self.handles = AdapterHandles(interactsh=self._client)

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        await self._client.close()
        self.handles = AdapterHandles()


def secret_holders(marker: str) -> list[str]:
    """Name every live object that holds a str or bytes containing ``marker``.

    Derived copies count too, for example an encoded Authorization header.
    """

    encoded = marker.encode()
    own = {id(marker), id(encoded)}

    def holds(value: object, depth: int) -> bool:
        if id(value) in own:
            return False
        if isinstance(value, str):
            return marker in value
        if isinstance(value, bytes | bytearray):
            return encoded in value
        # The collector untracks containers of atomic values, so only their
        # tracked owner is listed; look inside them from there.
        if depth and not gc.is_tracked(value):
            if isinstance(value, dict):
                return any(holds(k, depth - 1) or holds(v, depth - 1) for k, v in value.items())
            if isinstance(value, tuple | list | set | frozenset):
                return any(holds(item, depth - 1) for item in value)
        return False

    gc.collect()
    return sorted(
        type(owner).__qualname__
        for owner in gc.get_objects()
        if any(holds(referent, 3) for referent in gc.get_referents(owner))
    )


def test_release_retains_no_adapter_secret_or_session_credential(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP, Caido and Interactsh tools leave no secret reachable after release."""

    monkeypatch.setattr(worker_state_module, "MetricsState", TrackedMetricsState)

    async def forbidden_dns(*_args):
        pytest.fail("mock adapter allocation must not use host DNS")

    monkeypatch.setattr(target_policy_module, "system_resolver", forbidden_dns)
    marker = uuid.uuid4().hex
    http_paths: list[str] = []
    caido_operations: list[str] = []

    async def target(request: httpx.Request) -> httpx.Response:
        http_paths.append(request.url.path)
        return httpx.Response(
            200, content=b"ok", headers={"content-type": "text/plain"}, request=request
        )

    async def caido(request: httpx.Request) -> httpx.Response:
        caido_operations.append(json.loads(request.content)["operationName"])
        data = {"requestsByOffset": {"count": {"value": 0}, "nodes": []}}
        return httpx.Response(200, json={"data": data}, request=request)

    def model(_: WorkerBuildContext) -> object:
        session = {"kind": "bearer", "token": f"session-{marker}"}
        return scripted_model(
            [
                tool_call(
                    "http_session_set",
                    {"auth": session, "replace_cookies": True, "replace_headers": True},
                    call_id="session-set",
                ),
                tool_call(
                    "http_request",
                    {
                        "url": "https://target.example/probe",
                        "method": "GET",
                        "body_type": "none",
                        "follow_redirects": False,
                    },
                    call_id="http-request",
                ),
                tool_call(
                    "caido_history", {"filter": "", "limit": 5, "offset": 0}, call_id="caido"
                ),
                tool_call("interactsh_register", {"action": "http-request"}, call_id="oast"),
                text_result("Collected bounded evidence"),
            ]
        )

    async def scenario() -> None:
        artifacts = MemoryArtifactClient()
        base = built_in_factories(
            tmp_path / "work",
            artifact_client_factory=lambda *_: artifacts,
            model_factory=model,
            enabled_runtime_adapters=[],
            target_policy=SCAN_TEST_POLICY,
        )
        proxy = MockHTTPProxyAdapterFactory(target)
        caido_adapter = MockCaidoAdapterFactory(caido)
        interactsh = MockInteractshAdapterFactory()
        factories = FactoryRegistry(
            worker_runtimes=base.worker_runtimes,
            toolsets=base.toolsets,
            sandbox_profiles=base.sandbox_profiles,
            runtime_adapters={
                proxy.ref: proxy,
                caido_adapter.ref: caido_adapter,
                interactsh.ref: interactsh,
            },
            artifact_client_factory=base.artifact_client_factory,
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=factories.worker_runtimes,
            toolsets={ref: factory.exported_tools for ref, factory in factories.toolsets.items()},
            sandbox_profiles=factories.sandbox_profiles,
            runtime_adapters=factories.runtime_adapters,
        )
        state = RuntimeState(instance_id="runtime-adapter-secrets")
        await state.mark_registered()
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
            force_exit=lambda _: None,
        )
        spec = make_spec()
        spec.agent_template.toolsets = [
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="http-tools", version="1"),
                tools=["http_request", "http_session_set"],
            ),
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="caido", version="1"), tools=["caido_history"]
            ),
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="interactsh", version="1"),
                tools=["interactsh_register"],
            ),
        ]
        resign_template(spec.agent_template)
        spec.runtime_settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            llmGatewayToken=f"gateway-{marker}",
            artifactApiUrl="https://control.example/private/v1",
            httpProxy=HTTPProxySettings(
                adapter="http-proxy@1",
                proxyUrl="https://proxy.example",
                bearerToken=f"proxy-{marker}",
                targets=["tool-http"],
            ),
            caido=CaidoSettings(
                adapter="caido-graphql@1",
                endpoint="https://caido.example",
                bearerToken=f"caido-{marker}",
                requestTimeoutSeconds=5,
            ),
            interactsh=InteractshSettings(
                adapter="interactsh-http@1",
                endpoint="https://oast.example",
                token=f"oast-{marker}",
                requestTimeoutSeconds=5,
            ),
            requestTimeoutSeconds=10,
        )
        spec.resolved_runtime_config_provenance.runtime_adapters = [
            "caido-graphql@1",
            "http-proxy@1",
            "interactsh-http@1",
        ]
        allocation_id = spec.allocation_id
        await service.prepare(spec)
        # The Control Plane's request object is not Runtime state.
        del spec
        references = allocation_references(service)
        assert {
            "tool:http_session_set",
            "tool:caido_history",
            "tool:interactsh_register",
        } <= set(references)
        assert service._context is not None and service._context.worker is not None
        completion = await service._context.worker.invoke(adk_stage_request())
        assert completion.failure is None and completion.result is not None
        del completion
        assert http_paths == ["/probe"]
        assert caido_operations == ["RequestsByOffset"]
        assert interactsh.paths == ["/register"]
        assert secret_holders(marker)

        await finalize_and_release(service, allocation_id)

        assert live_references(references) == []
        assert secret_holders(marker) == []

    asyncio.run(scenario())


class ObservingTrackingTool(TrackingTool):
    def contractor_observation(self, _tool_args: object, _result: object) -> None:
        return None


class DeclaringTrackingToolsetFactory(TrackingToolsetFactory):
    workspace_observation_tools = frozenset({"read_artifact"})


@pytest.mark.parametrize(
    "factory",
    [
        # A tool that reports observations its Toolset does not declare.
        lambda: TrackingToolsetFactory(ObservingTrackingTool()),
        # A declared observing tool without an extractor.
        lambda: DeclaringTrackingToolsetFactory(TrackingTool()),
    ],
    ids=["undeclared-extractor", "declared-without-extractor"],
)
def test_prepare_rejects_a_tool_whose_workspace_observation_differs_from_its_declaration(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    factory: Callable[[], TrackingToolsetFactory],
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-test")
        await state.mark_registered()
        toolsets = factory()
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={"run-artifacts@1": toolsets},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
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
        assert failure.value.payload()["code"] == "invalid_toolset_factory"
        assert failure.value.status_code == 500
        assert toolsets.tool.close_calls == 1
        assert await service.snapshot() is None

    asyncio.run(scenario())
