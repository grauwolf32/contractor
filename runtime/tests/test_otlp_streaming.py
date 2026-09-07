from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from test_otlp_adapter import (
    HEADER_SECRET,
    adapter_context,
    allocation_service,
    configured_allocation,
    telemetry_settings,
)

from contractor_runtime.adapters.content import MAX_CONTENT_BYTES
from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory, OTLPInstrumentation
from contractor_runtime.contracts import (
    API_VERSION,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    TelemetryExportSettings,
    TelemetrySettings,
)
from contractor_runtime.state import ProcessState


def _settings(batch_size_bytes: int = 8 * 1024 * 1024) -> TelemetrySettings:
    # Tiny batches keep transport race tests cheap; public validation is tested
    # separately with the supported 1 MiB minimum.
    export = TelemetryExportSettings.defaults().model_copy(
        update={"batch_size_bytes": batch_size_bytes}
    )
    return telemetry_settings().model_copy(update={"capture_content": True, "export": export})


def _span(instrumentation: OTLPInstrumentation, number: int, *, size: int = 2048) -> None:
    span = instrumentation.start_span(
        "contractor.worker.model", attributes={"model.alias": f"model-{number}"}
    )
    span.set_content("input", '"' + "x" * (size - 2) + '"')
    span.end(outcome="succeeded")


def _span_ids(payloads: list[bytes]) -> list[str]:
    return [
        span.span_id.hex()
        for payload in payloads
        for resource in ExportTraceServiceRequest.FromString(payload).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


def test_configured_export_limits_and_attempts_control_background_delivery() -> None:
    async def scenario() -> None:
        payloads: list[bytes] = []

        async def collector(request: httpx.Request) -> httpx.Response:
            payloads.append(await request.aread())
            return httpx.Response(503 if len(payloads) < 3 else 200)

        # Validate the actual private wire boundary, with non-default values.
        settings = TelemetrySettings.model_validate_json(
            '{"adapter":"otlp-http@1","endpoint":"https://collector.example/v1/traces",'
            '"headers":{},"captureContent":true,"flushTimeoutSeconds":10,'
            '"export":{"batchSizeBytes":1048576,"maxAttempts":3,'
            '"maxPendingSpans":16,"maxPendingBytes":2097152}}'
        )
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings
        )
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        try:
            for number in range(5):
                _span(instrumentation, number, size=MAX_CONTENT_BYTES)
            expected = _span_ids([instrumentation.export_request()])
            assert adapter._export_task is not None
            assert await asyncio.wait_for(adapter._export_task, timeout=1)
            assert len(payloads) == 3
            assert payloads[0] == payloads[1] == payloads[2]
            await adapter.flush()
            assert _span_ids(payloads[2:]) == expected
            assert all(len(payload) <= 1048576 for payload in payloads)
            assert adapter.metrics.failed_operations == 2
        finally:
            await adapter.close()

    asyncio.run(scenario())


def test_byte_capacity_flushes_even_when_next_span_cannot_fit() -> None:
    async def scenario() -> None:
        payloads: list[bytes] = []

        async def collector(request: httpx.Request) -> httpx.Response:
            payloads.append(await request.aread())
            return httpx.Response(200)

        export = TelemetryExportSettings(
            batchSizeBytes=1048576, maxPendingBytes=1048576, maxPendingSpans=16, maxAttempts=1
        )
        settings = telemetry_settings().model_copy(
            update={"capture_content": True, "export": export}
        )
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings
        )
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        try:
            for number in range(4):
                _span(instrumentation, number, size=MAX_CONTENT_BYTES)
            assert instrumentation.pending_bytes < export.batch_size_bytes
            assert adapter.metrics.last_error_code == "queue_overflow"
            assert adapter._export_task is not None
            assert await asyncio.wait_for(adapter._export_task, timeout=1)
            assert len(_span_ids(payloads)) == 3
            assert instrumentation.pending_spans == 0
        finally:
            await adapter.close()

    asyncio.run(scenario())


def test_eight_mib_triggers_background_batches_and_final_flush_preserves_new_spans() -> None:
    assert TelemetryExportSettings.defaults().batch_size_bytes == 8 * 1024 * 1024

    async def scenario() -> None:
        entered = asyncio.Event()
        proceed = asyncio.Event()
        payloads: list[bytes] = []
        active = 0
        maximum_active = 0

        async def collector(request: httpx.Request) -> httpx.Response:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                payloads.append(await request.aread())
                assert request.headers["Authorization"] == f"Bearer {HEADER_SECRET}"
                if len(payloads) == 1:
                    entered.set()
                    await proceed.wait()
                return httpx.Response(200)
            finally:
                active -= 1

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), _settings()
        )
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        try:
            for number in range(31):
                _span(instrumentation, number, size=MAX_CONTENT_BYTES)
            await asyncio.sleep(0)
            assert payloads == []
            for number in range(31, 33):
                _span(instrumentation, number, size=MAX_CONTENT_BYTES)
            await asyncio.wait_for(entered.wait(), timeout=1)

            # The Worker can keep recording while the first POST is blocked.
            for number in range(33, 70):
                _span(instrumentation, number, size=MAX_CONTENT_BYTES)
            expected = _span_ids([instrumentation.export_request()])
            assert len(expected) == 70  # Includes the in-flight prefix.
            assert len(payloads) == 1

            proceed.set()
            assert adapter._export_task is not None
            assert await asyncio.wait_for(adapter._export_task, timeout=2)
            assert len(payloads) >= 2  # Several batches before finalization.
            assert 0 < instrumentation.pending_spans < 31
            assert (
                instrumentation.pending_bytes < TelemetryExportSettings.defaults().batch_size_bytes
            )

            await adapter.flush()
            assert instrumentation.pending_spans == 0
            assert _span_ids(payloads) == expected
            assert maximum_active == 1
            assert all(
                len(payload) <= TelemetryExportSettings.defaults().batch_size_bytes
                for payload in payloads
            )
            assert adapter.metrics.failed_operations == 0
        finally:
            proceed.set()
            await adapter.close()

    asyncio.run(scenario())


def test_final_flush_joins_active_sender_and_drains_tail() -> None:

    async def scenario() -> None:
        entered = asyncio.Event()
        proceed = asyncio.Event()
        payloads: list[bytes] = []

        async def collector(request: httpx.Request) -> httpx.Response:
            payloads.append(await request.aread())
            entered.set()
            await proceed.wait()
            return httpx.Response(200)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), _settings(4096)
        )
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        try:
            _span(instrumentation, 0)
            _span(instrumentation, 1)
            await asyncio.wait_for(entered.wait(), timeout=1)
            _span(instrumentation, 2, size=10)
            expected = _span_ids([instrumentation.export_request()])
            flush = asyncio.create_task(adapter.flush())
            await asyncio.sleep(0)
            assert not flush.done()
            assert len(payloads) == 1
            proceed.set()
            await asyncio.wait_for(flush, timeout=1)
            assert _span_ids(payloads) == expected
            assert instrumentation.pending_spans == 0
            await adapter.flush()
            assert _span_ids(payloads) == expected
        finally:
            proceed.set()
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("failure", ["status", "disconnect", "timeout"])
def test_second_attempt_recovers_with_identical_payload_and_preserves_new_spans(
    background: bool, failure: str
) -> None:

    async def scenario() -> None:
        retry_entered = asyncio.Event()
        proceed = asyncio.Event()
        payloads: list[bytes] = []

        async def collector(request: httpx.Request) -> httpx.Response:
            payloads.append(await request.aread())
            if len(payloads) == 1:
                if failure == "disconnect":
                    raise httpx.ConnectError("provider-secret-canary")
                if failure == "timeout":
                    await asyncio.Event().wait()
                return httpx.Response(503)
            if len(payloads) == 2:
                retry_entered.set()
                await proceed.wait()
            return httpx.Response(200)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), _settings(4096 if background else 16 * 1024)
        )
        adapter._delivery_timeout = 0.1
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        try:
            _span(instrumentation, 0)
            _span(instrumentation, 1)
            if background:
                completion = adapter._export_task
            else:
                assert adapter._export_task is None
                completion = asyncio.create_task(adapter.flush())
            assert completion is not None
            await asyncio.wait_for(retry_entered.wait(), timeout=1)
            assert payloads[0] == payloads[1]
            _span(instrumentation, 2, size=10)
            expected = _span_ids([instrumentation.export_request()])
            assert len(expected) == 3  # The retry remains charged to the queue.
            proceed.set()
            result = await asyncio.wait_for(completion, timeout=1)
            if background:
                assert result is True
            await adapter.flush()
            assert _span_ids(payloads[1:]) == expected
            assert instrumentation.pending_spans == 0
            assert adapter.metrics.operations == len(payloads)
            assert adapter.metrics.failed_operations == 1
        finally:
            proceed.set()
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["status", "disconnect", "timeout"])
def test_background_failure_drops_only_attempted_batch_and_can_send_again(failure: str) -> None:

    async def scenario() -> None:
        payloads: list[bytes] = []

        async def collector(request: httpx.Request) -> httpx.Response:
            payloads.append(await request.aread())
            if len(payloads) <= 2:
                if failure == "disconnect":
                    raise httpx.ConnectError("provider-secret-canary")
                if failure == "timeout":
                    await asyncio.Event().wait()
                return httpx.Response(503, content=b"provider-secret-canary")
            return httpx.Response(200)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), _settings(4096)
        )
        adapter._delivery_timeout = 0.02
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        try:
            _span(instrumentation, 0)
            _span(instrumentation, 1)
            assert adapter._export_task is not None
            assert not await asyncio.wait_for(adapter._export_task, timeout=1)
            assert len(payloads) == 2
            assert payloads[0] == payloads[1]
            assert instrumentation.pending_spans == 1
            assert adapter.metrics.failed_operations == 2
            assert adapter.metrics.last_error_code == "delivery_failed"
            assert "provider-secret-canary" not in repr(adapter.metrics)

            # New arrivals must restart an idle sender after a failed batch.
            _span(instrumentation, 2)
            assert await asyncio.wait_for(adapter._export_task, timeout=1)
            await adapter.flush()
            ids = _span_ids(payloads[1:])
            assert len(ids) == len(set(ids)) == 3
            assert instrumentation.pending_spans == 0
        finally:
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_inflight_spans_stay_bounded_and_close_cancels_sender(
    limit: str,
) -> None:

    async def scenario() -> None:
        entered = asyncio.Event()
        stopped = asyncio.Event()
        requests = 0

        async def collector(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return httpx.Response(200)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), _settings(4096)
        )
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        if limit == "count":
            instrumentation._max_pending_spans = 3
        _span(instrumentation, 0)
        _span(instrumentation, 1)
        await asyncio.wait_for(entered.wait(), timeout=1)
        _span(instrumentation, 2)
        accepted_bytes = instrumentation.pending_bytes
        if limit == "bytes":
            instrumentation._max_pending_bytes = accepted_bytes
        _span(instrumentation, 3)
        assert instrumentation.pending_spans == 3
        assert instrumentation.pending_bytes == accepted_bytes
        assert adapter.metrics.last_error_code == "queue_overflow"
        task = adapter._export_task
        await asyncio.wait_for(adapter.close(), timeout=1)
        assert stopped.is_set()
        assert task is not None and task.cancelled()
        assert instrumentation.pending_spans == 0
        assert adapter._client is None
        assert not adapter._headers
        _span(instrumentation, 4)
        await adapter.flush()
        await adapter.close()
        assert requests == 1

    asyncio.run(scenario())


def test_metadata_only_spans_export_when_count_limit_is_reached() -> None:
    async def scenario() -> None:
        payloads: list[bytes] = []

        async def collector(request: httpx.Request) -> httpx.Response:
            payloads.append(await request.aread())
            return httpx.Response(200)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), telemetry_settings()
        )
        instrumentation = adapter.handles.instrumentation
        assert isinstance(instrumentation, OTLPInstrumentation)
        try:
            for number in range(TelemetryExportSettings.defaults().max_pending_spans):
                _span(instrumentation, number)
            assert (
                instrumentation.pending_bytes < TelemetryExportSettings.defaults().batch_size_bytes
            )
            assert adapter._export_task is not None
            assert await asyncio.wait_for(adapter._export_task, timeout=1)
            assert len(_span_ids(payloads)) == TelemetryExportSettings.defaults().max_pending_spans
            assert instrumentation.pending_spans == 0
            assert adapter.metrics.failed_operations == 0
        finally:
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("first_attempt_fails", [False, True])
def test_allocation_deadline_cancels_background_post_and_allows_release(
    tmp_path: Path, first_attempt_fails: bool
) -> None:

    async def scenario() -> None:
        entered = asyncio.Event()
        stopped = asyncio.Event()
        requests = 0

        async def collector(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            if first_attempt_fails and requests == 1:
                return httpx.Response(503)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return httpx.Response(200)

        state, service = await allocation_service(
            tmp_path, OTLPHTTPAdapterFactory(httpx.MockTransport(collector))
        )
        spec = configured_allocation()
        spec.runtime_settings.telemetry = _settings(4096)
        spec.runtime_settings.telemetry.export.max_pending_spans = 1
        await service.prepare(spec)
        assert service._context is not None
        instrumentation = service._context.adapter_host.handles.instrumentation
        assert instrumentation is not None
        instrumentation.start_span("contractor.worker.model").end(outcome="succeeded")
        await asyncio.wait_for(entered.wait(), timeout=1)
        request = FinalizeAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=spec.allocation_id,
            finalizationId="finalize-background-otlp",
            deadline=datetime.now(UTC) + timedelta(milliseconds=100),
        )
        response = await service.finalize(request)
        assert stopped.is_set()
        assert response.report.worker.complete
        metrics = response.report.runtime.adapters["otlp-http@1"]
        assert metrics.flush_attempted and not metrics.flush_succeeded
        assert metrics.last_error_code == "flush_timeout"
        assert await service.finalize(request) == response
        assert requests == (2 if first_attempt_fails else 1)
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())
