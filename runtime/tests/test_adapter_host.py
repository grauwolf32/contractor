from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from fakes.spec import allocation_spec

from contractor_runtime.adapters import (
    AdapterFactoryError,
    AdapterHandles,
    RuntimeAdapterBuildContext,
    RuntimeAdapterMetricsState,
)
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AllocationSpecV2,
    FinalizeAllocationRequest,
    HTTPProxySettingsV2,
    ReleaseAllocationRequest,
    RuntimeAdapterRef,
    TelemetrySettingsV2,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    StubADKWorkerRuntimeFactory,
    StubWorkerRuntime,
    WorkerBuildContext,
)
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory

TELEMETRY_SECRET = "recognizable-telemetry-header-secret"
PROXY_SECRET = "recognizable-proxy-password-secret"


def test_second_adapter_prepare_failure_rolls_back_first_before_sandbox(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    proxy = FakeAdapterFactory(
        "http-proxy@1",
        events,
        handles=AdapterHandles(model_http=object()),
    )
    telemetry = FakeAdapterFactory(
        "otlp-http@1",
        events,
        handles=AdapterHandles(instrumentation=object()),
        prepare_error=AdapterFactoryError(retryable=True),
    )

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_adapters={"http-proxy@1": proxy, "otlp-http@1": telemetry},
            sandbox=RecordingSandbox(tmp_path, events),
        )
        with pytest.raises(AllocationError) as failure:
            await service.prepare(configured_spec(telemetry=True, proxy_targets=["llm-gateway"]))

        assert failure.value.code == "runtime_adapter_prepare_failed"
        assert failure.value.retryable
        assert str(failure.value) == "allocation Runtime adapter preparation failed"
        assert events == [
            "create:http-proxy@1",
            "create:otlp-http@1",
            "close:http-proxy@1",
        ]
        assert await service.snapshot() is None
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())
    assert TELEMETRY_SECRET not in repr(telemetry)
    assert PROXY_SECRET not in " ".join(events)


def test_hanging_flush_is_metrics_only_and_replayed_lifecycle_is_idempotent(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    telemetry = FakeAdapterFactory(
        "otlp-http@1",
        events,
        handles=AdapterHandles(instrumentation=object()),
        hang_flush=True,
    )

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_adapters={"otlp-http@1": telemetry},
        )
        spec = configured_spec(telemetry=True)
        await service.prepare(spec)
        snapshot = await service.snapshot()
        assert snapshot is not None
        assert snapshot.runtime_adapter_refs == ("otlp-http@1",)

        request = FinalizeAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
            finalizationId="finalize-adapter-timeout",
            deadline=datetime.now(UTC) + timedelta(milliseconds=50),
        )
        first = await service.finalize(request)
        second = await service.finalize(request)
        assert second == first
        metrics = first.report.runtime.adapters["otlp-http@1"]
        assert metrics.operations == 1
        assert metrics.failed_operations == 1
        assert metrics.flush_attempted
        assert metrics.flush_succeeded is False
        assert metrics.last_error_code == "flush_timeout"
        assert first.report.worker.complete
        assert events.count("flush:otlp-http@1") == 1
        assert events.count("close:otlp-http@1") == 1

        release = ReleaseAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
        )
        await service.release(release)
        await service.release(release)
        await service.confirm_release(spec.allocation_id)
        await service.confirm_release(spec.allocation_id)
        assert events.count("flush:otlp-http@1") == 1
        assert events.count("close:otlp-http@1") == 1
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_handles_are_injected_explicitly_and_erased_on_confirmed_lease_loss(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    model_handle = object()
    tool_handle = object()
    proxy = FakeAdapterFactory(
        "http-proxy@1",
        events,
        handles=AdapterHandles(model_http=model_handle, tool_http=tool_handle),
    )
    toolset = CapturingToolset()
    runtime = CapturingRuntimeFactory()

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_adapters={"http-proxy@1": proxy},
            toolset=toolset,
            runtime=runtime,
        )
        spec = configured_spec(proxy_targets=["llm-gateway", "tool-http"])
        await service.prepare(spec)

        assert toolset.handles is not None
        assert toolset.handles.tool_http is tool_handle
        assert runtime.context is not None
        assert runtime.context.adapter_handles.model_http is model_handle
        assert runtime.context.adapter_handles.tool_http is None
        assert "recognizable" not in repr(runtime.context.adapter_handles)

        await service.expire_control_lease(1)
        assert events[-1] == "close:http-proxy@1"
        snapshot = await service.snapshot()
        assert snapshot is not None
        assert not snapshot.has_worker
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert service._context is not None
        assert service._context.adapter_host.closed
        assert service._context.adapter_host.handles.enabled_channels == ()

    asyncio.run(scenario())


def test_unselected_tool_channels_are_not_injected(tmp_path: Path) -> None:
    events: list[str] = []
    tool_handle = object()
    proxy = FakeAdapterFactory(
        "http-proxy@1",
        events,
        handles=AdapterHandles(tool_http=tool_handle),
    )
    toolset = CapturingToolset()
    toolset.infrastructure_channels = MappingProxyType({})
    runtime = CapturingRuntimeFactory()

    async def scenario() -> None:
        _state, service = await make_service(
            tmp_path,
            runtime_adapters={"http-proxy@1": proxy},
            toolset=toolset,
            runtime=runtime,
        )
        spec = configured_spec(proxy_targets=["tool-http"])
        await service.prepare(spec)

        assert toolset.handles is not None
        assert toolset.handles.enabled_channels == ()
        assert runtime.context is not None
        assert runtime.context.adapter_handles.enabled_channels == ()

        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-no-tool-channel",
                deadline=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)

    asyncio.run(scenario())


def test_unconfirmed_adapter_close_fences_slot_and_requests_process_exit(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    exits: list[int] = []
    telemetry = FakeAdapterFactory(
        "otlp-http@1",
        events,
        handles=AdapterHandles(instrumentation=object()),
        close_error=RuntimeError(f"close failed with {TELEMETRY_SECRET}"),
    )

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_adapters={"otlp-http@1": telemetry},
            force_exit=exits.append,
        )
        spec = configured_spec(telemetry=True)
        await service.prepare(spec)
        with pytest.raises(AllocationError) as failure:
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalize-close-failure",
                    deadline=datetime.now(UTC) + timedelta(seconds=1),
                )
            )
        assert failure.value.code == "runtime_adapter_close_unconfirmed"
        assert TELEMETRY_SECRET not in str(failure.value)
        assert exits == [70]
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        snapshot = await service.snapshot()
        assert snapshot is not None
        assert snapshot.has_runtime_settings

    asyncio.run(scenario())


def test_adapter_secret_in_worker_handle_is_rejected_and_every_resource_rolls_back(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    telemetry = FakeAdapterFactory(
        "otlp-http@1",
        events,
        handles=AdapterHandles(instrumentation=object()),
    )

    async def scenario() -> None:
        state, service = await make_service(
            tmp_path,
            runtime_adapters={"otlp-http@1": telemetry},
            sandbox=RecordingSandbox(tmp_path / "work", events),
            runtime=LeakingRuntimeFactory(),
        )
        with pytest.raises(AllocationError) as failure:
            await service.prepare(configured_spec(telemetry=True))
        assert failure.value.code == "unsafe_worker_handle"
        assert TELEMETRY_SECRET not in str(failure.value)
        assert events == [
            "create:otlp-http@1",
            "sandbox:prepare",
            "sandbox:cleanup",
            "close:otlp-http@1",
        ]
        assert await service.snapshot() is None
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_metrics_saturate_and_reject_unbounded_error_codes() -> None:
    metrics = RuntimeAdapterMetricsState(
        operations=2**64 - 1,
        failed_operations=2**64 - 1,
    )
    metrics.record_operation(succeeded=False, error_code="delivery_failed")
    assert metrics.operations == 2**64 - 1
    assert metrics.failed_operations == 2**64 - 1
    assert metrics.snapshot().last_error_code == "delivery_failed"
    with pytest.raises(ValueError, match="allowlisted"):
        metrics.record_operation(succeeded=False, error_code=TELEMETRY_SECRET)
    metrics.operations = -100
    metrics.failed_operations = 100
    metrics.last_error_code = TELEMETRY_SECRET
    snapshot = metrics.snapshot()
    assert snapshot.operations == 0
    assert snapshot.failed_operations == 0
    assert snapshot.last_error_code is None


async def make_service(
    tmp_path: Path,
    *,
    runtime_adapters: Mapping[str, FakeAdapterFactory],
    sandbox: LocalWorkdirFactory | None = None,
    toolset: CapturingToolset | None = None,
    runtime: CapturingRuntimeFactory | None = None,
    force_exit: Any = lambda _: None,
) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-adapter-test")
    await state.mark_registered()
    selected_toolset = toolset or CapturingToolset()
    registry = FactoryRegistry(
        worker_runtimes={"adk@1": runtime or StubADKWorkerRuntimeFactory()},
        toolsets={"run-artifacts@1": selected_toolset},
        sandbox_profiles={"local-workdir@1": sandbox or LocalWorkdirFactory(tmp_path / "work")},
        runtime_adapters=runtime_adapters,
    )
    capabilities = CapabilitySnapshot.create(
        runtimes=["adk@1"],
        toolsets={"run-artifacts@1": ["read_artifact"]},
        sandbox_profiles=["local-workdir@1"],
        runtime_adapters=runtime_adapters,
    )
    return state, AllocationService(
        state,
        registry,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=force_exit,
    )


def configured_spec(
    *,
    telemetry: bool = False,
    proxy_targets: list[str] | None = None,
) -> AllocationSpecV2:
    spec = allocation_spec(tools=["read_artifact"])
    telemetry_settings = (
        TelemetrySettingsV2(
            adapter="otlp-http@1",
            endpoint="https://telemetry.example/v1/traces",
            headers={"Authorization": TELEMETRY_SECRET},
            captureContent=False,
            flushTimeoutSeconds=1,
        )
        if telemetry
        else None
    )
    proxy_settings = (
        HTTPProxySettingsV2(
            adapter="http-proxy@1",
            proxyUrl="https://proxy.example",
            basicAuth={"username": "worker", "password": PROXY_SECRET},
            targets=sorted(proxy_targets),
        )
        if proxy_targets
        else None
    )
    refs = sorted(
        ref
        for ref, selected in (
            ("http-proxy@1", proxy_settings),
            ("otlp-http@1", telemetry_settings),
        )
        if selected is not None
    )
    return spec.model_copy(
        update={
            "runtime_settings": spec.runtime_settings.model_copy(
                update={"telemetry": telemetry_settings, "http_proxy": proxy_settings}
            ),
            "resolved_runtime_config_provenance": (
                spec.resolved_runtime_config_provenance.model_copy(
                    update={"runtime_adapters": refs}
                )
            ),
        }
    )


class FakeAdapterFactory:
    def __init__(
        self,
        ref: str,
        events: list[str],
        *,
        handles: AdapterHandles,
        prepare_error: Exception | None = None,
        close_error: Exception | None = None,
        hang_flush: bool = False,
    ) -> None:
        self.ref = ref
        self._events = events
        self._handles = handles
        self._prepare_error = prepare_error
        self._close_error = close_error
        self._hang_flush = hang_flush

    async def probe(self) -> bool:
        return True

    async def create(
        self,
        context: RuntimeAdapterBuildContext,
        settings: object,
    ) -> FakeAdapter:
        del context, settings
        self._events.append(f"create:{self.ref}")
        if self._prepare_error is not None:
            raise self._prepare_error
        return FakeAdapter(
            self.ref,
            self._events,
            handles=self._handles,
            close_error=self._close_error,
            hang_flush=self._hang_flush,
        )

    def __repr__(self) -> str:
        return f"FakeAdapterFactory(ref={self.ref!r})"


class FakeAdapter:
    def __init__(
        self,
        ref: str,
        events: list[str],
        *,
        handles: AdapterHandles,
        close_error: Exception | None,
        hang_flush: bool,
    ) -> None:
        self.ref: RuntimeAdapterRef = ref  # type: ignore[assignment]
        self.handles = handles
        self.metrics = RuntimeAdapterMetricsState()
        self._events = events
        self._close_error = close_error
        self._hang_flush = hang_flush

    async def flush(self) -> None:
        self._events.append(f"flush:{self.ref}")
        if self._hang_flush:
            await asyncio.Event().wait()

    async def close(self) -> None:
        self._events.append(f"close:{self.ref}")
        if self._close_error is not None:
            raise self._close_error


class RecordingSandbox(LocalWorkdirFactory):
    def __init__(self, root: Path, events: list[str]) -> None:
        super().__init__(root)
        self._events = events

    async def prepare(self) -> AllocationWorkspace:
        self._events.append("sandbox:prepare")
        return await super().prepare()

    async def cleanup(self, workspace: AllocationWorkspace) -> None:
        self._events.append("sandbox:cleanup")
        await super().cleanup(workspace)


class CapturingToolset:
    ref = "run-artifacts@1"
    exported_tools = frozenset({"read_artifact"})
    infrastructure_channels = MappingProxyType(
        {"read_artifact": frozenset({"runtime-http-client"})}
    )

    def __init__(self) -> None:
        self.handles: AdapterHandles | None = None

    async def probe(self) -> frozenset[str]:
        return self.exported_tools

    async def create_selected(self, **values: Any) -> dict[str, CapturingTool]:
        self.handles = values["adapter_handles"]
        return {"read_artifact": CapturingTool()}


class CapturingTool:
    name = "read_artifact"

    async def close(self) -> None:
        return None


class CapturingRuntimeFactory:
    ref = "adk@1"

    def __init__(self) -> None:
        self.context: WorkerBuildContext | None = None

    async def probe(self) -> bool:
        return True

    async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
        self.context = context
        return StubWorkerRuntime(context)


class LeakingRuntimeFactory(CapturingRuntimeFactory):
    async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
        runtime = await super().create(context)
        runtime._agent_card["description"] = TELEMETRY_SECRET
        return runtime
