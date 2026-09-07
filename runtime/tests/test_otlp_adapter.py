from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx
import pytest
from fakes.spec import allocation_spec
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from contractor_runtime.adapters.host import RuntimeAdapterBuildContext
from contractor_runtime.adapters.otlp_http import (
    MAX_SPAN_ATTRIBUTES,
    MAX_STRING_ATTRIBUTE_BYTES,
    OTLPDeliveryError,
    OTLPHTTPAdapterFactory,
    OTLPInstrumentation,
)
from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AllocationSpec,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    TelemetryExportSettings,
    TelemetrySettings,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    RunArtifactsToolsetFactory,
    StubADKWorkerRuntimeFactory,
)
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import LocalWorkdirFactory

MAX_PENDING_BYTES = TelemetryExportSettings.defaults().max_pending_bytes
MAX_PENDING_SPANS = TelemetryExportSettings.defaults().max_pending_spans

HEADER_SECRET = "recognizable-otlp-header-secret"
PROMPT_SECRET = "recognizable-prompt-content-canary"
TOOL_SECRET = "recognizable-tool-argument-canary"
PROVIDER_SECRET = "recognizable-provider-error-canary"
ENDPOINT = "https://collector.example/v1/traces"


def test_otlp_protobuf_is_bounded_content_free_and_header_scoped() -> None:
    requests: list[httpx.Request] = []
    payloads: list[bytes] = []

    async def collector(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payloads.append(await request.aread())
        return httpx.Response(200)

    async def scenario() -> None:
        assert await OTLPHTTPAdapterFactory().probe()
        factory = OTLPHTTPAdapterFactory(httpx.MockTransport(collector))
        adapter = await factory.create(adapter_context(), telemetry_settings())
        instrumentation = adapter.handles.instrumentation
        assert instrumentation is not None

        model = instrumentation.start_span(
            "contractor.worker.model",
            attributes={
                "operation.kind": "model",
                "model.alias": "worker-model-" + "x" * 400,
                "prompt": PROMPT_SECRET,
                "response": PROMPT_SECRET,
                "provider.body": PROVIDER_SECRET,
                "url": ENDPOINT,
            },
        )
        model.end(
            outcome="succeeded",
            attributes={"tokens.input": 11, "tokens.output": 7, "tokens.total": 18},
        )
        tool = instrumentation.start_span(
            "contractor.worker.tool",
            attributes={
                "operation.kind": "tool",
                "tool.name": "read_artifact",
                "tool.arguments": TOOL_SECRET,
                "artifact.content": TOOL_SECRET,
            },
        )
        tool.end(outcome="failed", attributes={"error.type": "ArtifactUnavailable"})
        task = instrumentation.start_span(
            "contractor.worker.a2a_task",
            attributes={"operation.kind": "a2a_task"},
        )
        task.end(outcome="succeeded")

        await adapter.flush()
        assert adapter.metrics.operations == 1
        assert adapter.metrics.failed_operations == 0
        await adapter.close()
        assert HEADER_SECRET not in repr(adapter)
        assert ENDPOINT not in repr(adapter)

    asyncio.run(scenario())

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == ENDPOINT
    assert request.headers["Authorization"] == f"Bearer {HEADER_SECRET}"
    assert request.headers["Content-Type"] == "application/x-protobuf"
    assert request.headers["Accept"] == "application/x-protobuf"
    payload = payloads[0]
    for forbidden in (
        HEADER_SECRET,
        PROMPT_SECRET,
        TOOL_SECRET,
        PROVIDER_SECRET,
        ENDPOINT,
    ):
        assert forbidden.encode() not in payload

    decoded = ExportTraceServiceRequest.FromString(payload)
    assert len(decoded.resource_spans) == 1
    resource = _attributes(decoded.resource_spans[0].resource.attributes)
    assert resource["service.name"] == "contractor-runtime-worker"
    assert resource["contractor.run.id"] == "run-1"
    assert resource["contractor.run.labels"] == ["debug"]
    assert resource["contractor.agent.labels"] == ["site-a"]
    assert not any(key.startswith("contractor.run.label.") for key in resource)
    spans = decoded.resource_spans[0].scope_spans[0].spans
    assert [span.name for span in spans] == [
        "contractor.worker.model",
        "contractor.worker.tool",
        "contractor.worker.a2a_task",
    ]
    assert len({span.trace_id for span in spans}) == 1
    model_attributes = _attributes(spans[0].attributes)
    assert len(model_attributes["model.alias"].encode()) <= MAX_STRING_ATTRIBUTE_BYTES
    assert model_attributes["tokens.total"] == 18
    assert set(model_attributes) <= {
        "langfuse.observation.type",
        "gen_ai.request.model",
        "operation.kind",
        "model.alias",
        "outcome",
        "duration.ms",
        "tokens.input",
        "tokens.output",
        "tokens.total",
    }
    assert all(len(span.attributes) <= MAX_SPAN_ATTRIBUTES for span in spans)
    task_attributes = _attributes(spans[2].attributes)
    assert task_attributes["contractor.run.label.purpose"] == "eval"
    assert task_attributes["contractor.run.label.eval.id"] == "eval_01"
    assert task_attributes["contractor.run.label.eval.leg"] == "a"
    assert task_attributes["contractor.run.label.eval.case"] == "case_1"
    assert task_attributes["contractor.run.label.eval.note"] == "left = right/β"
    assert task_attributes["contractor.run.label.debug"] == ""
    for span in spans[:2]:
        assert not any(
            key.startswith("contractor.run.label.") for key in _attributes(span.attributes)
        )


@pytest.mark.parametrize("failure", ["disconnect", "partial", "status"])
def test_otlp_delivery_failure_is_safe_metrics_only(failure: str) -> None:
    async def collector(request: httpx.Request) -> httpx.Response:
        del request
        if failure == "disconnect":
            raise httpx.ConnectError(f"disconnect {PROVIDER_SECRET}")
        if failure == "partial":
            response = ExportTraceServiceResponse()
            response.partial_success.rejected_spans = 1
            response.partial_success.error_message = PROVIDER_SECRET
            return httpx.Response(200, content=response.SerializeToString())
        return httpx.Response(503, content=PROVIDER_SECRET.encode())

    async def scenario() -> None:
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), telemetry_settings()
        )
        instrumentation = adapter.handles.instrumentation
        assert instrumentation is not None
        instrumentation.start_span(
            "contractor.worker.error",
            attributes={"operation.kind": "worker_error", "error.type": "RuntimeError"},
        ).end(outcome="failed")

        with pytest.raises(OTLPDeliveryError) as delivery:
            await adapter.flush()
        rendered = f"{delivery.value!s} {delivery.value!r} {adapter!r} {adapter.metrics!r}"
        assert PROVIDER_SECRET not in rendered
        assert HEADER_SECRET not in rendered
        assert ENDPOINT not in rendered
        assert adapter.metrics.operations == (1 if failure == "partial" else 2)
        assert adapter.metrics.failed_operations == (1 if failure == "partial" else 2)
        assert adapter.metrics.last_error_code == "delivery_failed"
        await adapter.close()

    asyncio.run(scenario())


def test_otlp_content_queue_retains_late_spans_beyond_two_mib() -> None:
    metrics = _metrics()
    instrumentation = OTLPInstrumentation(
        metrics, {"service.name": "test"}, secret_values=(), capture_content=True
    )
    for _ in range(16):
        span = instrumentation.start_span("contractor.worker.model", attributes={})
        span.set_content("input", '"' + "x" * (192 * 1024) + '"')
        span.end(outcome="succeeded")
    last = instrumentation.start_span("contractor.worker.a2a_task", attributes={})
    last.set_content("output", '"late-finish-canary"')
    last.end(outcome="succeeded")
    assert instrumentation.pending_spans == 17
    assert 2 * 1024 * 1024 < instrumentation.pending_bytes <= MAX_PENDING_BYTES
    request = ExportTraceServiceRequest.FromString(instrumentation.export_request())
    assert len(request.resource_spans[0].scope_spans[0].spans) == 17
    assert b"late-finish-canary" in request.SerializeToString()
    assert metrics.failed_operations == 0


def test_otlp_queue_enforces_count_and_encoded_byte_bounds() -> None:
    assert MAX_PENDING_SPANS == 2048
    assert MAX_PENDING_BYTES == 64 * 1024 * 1024
    count_metrics = _metrics()
    by_count = OTLPInstrumentation(
        count_metrics,
        {"service.name": "test"},
        secret_values=(),
        max_pending_spans=2,
        max_pending_bytes=MAX_PENDING_BYTES,
    )
    for _ in range(3):
        by_count.start_span(
            "contractor.worker.model",
            attributes={"operation.kind": "model"},
        ).end(outcome="succeeded")
    assert by_count.pending_spans == 2
    assert count_metrics.last_error_code == "queue_overflow"

    byte_metrics = _metrics()
    baseline = OTLPInstrumentation(
        byte_metrics,
        {"service.name": "test"},
        secret_values=(),
        max_pending_spans=MAX_PENDING_SPANS,
        max_pending_bytes=MAX_PENDING_BYTES,
    )
    baseline.start_span(
        "contractor.worker.model",
        attributes={"operation.kind": "model", "model.alias": "x" * 1000},
    ).end(outcome="succeeded")
    single_span_bytes = baseline.pending_bytes

    constrained = OTLPInstrumentation(
        byte_metrics,
        {"service.name": "test"},
        secret_values=(),
        max_pending_spans=MAX_PENDING_SPANS,
        max_pending_bytes=single_span_bytes - 1,
    )
    constrained.start_span(
        "contractor.worker.model",
        attributes={"operation.kind": "model", "model.alias": "x" * 1000},
    ).end(outcome="succeeded")
    assert constrained.pending_spans == 0
    assert byte_metrics.last_error_code == "queue_overflow"


def test_slow_collector_flush_timeout_does_not_change_worker_result_or_release(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def collector(request: httpx.Request) -> httpx.Response:
        await request.aread()
        started.set()
        await asyncio.Event().wait()
        return httpx.Response(200)

    async def scenario() -> None:
        state, service = await allocation_service(
            tmp_path,
            OTLPHTTPAdapterFactory(httpx.MockTransport(collector)),
        )
        spec = configured_allocation()
        await service.prepare(spec)
        assert service._context is not None
        instrumentation = service._context.adapter_host.handles.instrumentation
        assert instrumentation is not None
        instrumentation.start_span(
            "contractor.worker.a2a_task",
            attributes={"operation.kind": "a2a_task"},
        ).end(outcome="succeeded")

        response = await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-slow-otlp",
                deadline=datetime.now(UTC) + timedelta(milliseconds=100),
            )
        )
        assert started.is_set()
        assert response.report.worker.complete
        metrics = response.report.runtime.adapters["otlp-http@1"]
        assert metrics.flush_attempted
        assert metrics.flush_succeeded is False
        assert metrics.last_error_code == "flush_timeout"

        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_allocation_without_telemetry_never_constructs_or_calls_exporter(
    tmp_path: Path,
) -> None:
    requests = 0

    async def collector(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, request=request)

    async def scenario() -> None:
        _state, service = await allocation_service(
            tmp_path,
            OTLPHTTPAdapterFactory(httpx.MockTransport(collector)),
        )
        spec = allocation_spec(tools=["read_artifact"])
        await service.prepare(spec)
        assert service._context is not None
        assert service._context.adapter_host.refs == ()
        response = await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-without-telemetry",
                deadline=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        assert response.report.runtime.adapters == {}
        assert response.report.worker.metrics.model_calls == 0
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)

    asyncio.run(scenario())
    assert requests == 0


def adapter_context() -> RuntimeAdapterBuildContext:
    return RuntimeAdapterBuildContext(
        allocation_id="allocation-1",
        run_id="run-1",
        stage_execution_id="stage-execution-1",
        logical_agent_name="builder",
        request_timeout_seconds=5,
        runtime_config_refs=("debug-config@1",),
        runtime_config_digests=("sha256:" + "a" * 64,),
        run_labels=("debug",),
        agent_labels=("site-a",),
        run_metadata_labels=MappingProxyType(
            {
                "purpose": "eval",
                "eval.id": "eval_01",
                "eval.leg": "a",
                "eval.case": "case_1",
                "eval.note": "left = right/β",
                "debug": "",
            }
        ),
        runtime_adapter_refs=("otlp-http@1",),
        private_bypass_hosts=("artifact.example",),
    )


def telemetry_settings() -> TelemetrySettings:
    return TelemetrySettings(
        adapter="otlp-http@1",
        endpoint=ENDPOINT,
        headers={"Authorization": f"Bearer {HEADER_SECRET}"},
        captureContent=False,
        flushTimeoutSeconds=1,
    )


def configured_allocation() -> AllocationSpec:
    spec = allocation_spec(tools=["read_artifact"])
    return spec.model_copy(
        update={
            "runtime_settings": spec.runtime_settings.model_copy(
                update={"telemetry": telemetry_settings()}
            ),
            "resolved_runtime_config_provenance": (
                spec.resolved_runtime_config_provenance.model_copy(
                    update={"runtime_adapters": ["otlp-http@1"]}
                )
            ),
        }
    )


async def allocation_service(
    tmp_path: Path,
    adapter_factory: OTLPHTTPAdapterFactory,
) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-otlp-test")
    await state.mark_registered()
    registry = FactoryRegistry(
        worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        toolsets={"run-artifacts@1": RunArtifactsToolsetFactory()},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path / "work")},
        runtime_adapters={"otlp-http@1": adapter_factory},
    )
    capabilities = CapabilitySnapshot.create(
        runtimes=["adk@1"],
        toolsets={"run-artifacts@1": ["read_artifact"]},
        sandbox_profiles=["local-workdir@1"],
        runtime_adapters=["otlp-http@1"],
    )
    return state, AllocationService(
        state,
        registry,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=lambda _: None,
    )


def _attributes(values: Any) -> dict[str, Any]:
    return {value.key: _attribute_value(value.value) for value in values}


def _attribute_value(value: Any) -> Any:
    selected = value.WhichOneof("value")
    if selected == "array_value":
        return [_attribute_value(item) for item in value.array_value.values]
    return getattr(value, selected)


def _metrics() -> Any:
    from contractor_runtime.adapters import RuntimeAdapterMetricsState

    return RuntimeAdapterMetricsState()
