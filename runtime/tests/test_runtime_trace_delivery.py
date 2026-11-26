"""Stop-budget and cancellation boundaries around the existing final exporter."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from fakes.allocation import allocation_context, allocation_owner
from fakes.clock import advance_clock, run_with_advanceable_clock
from test_otlp_adapter import allocation_service, configured_allocation

from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.contracts import API_VERSION, FinalizeAllocationRequest
from contractor_runtime.state import ProcessState


@pytest.mark.parametrize("collector_hangs", [False, True])
def test_cleanup_timeout_keeps_flush_inside_original_stop_deadline(tmp_path, collector_hangs):
    async def scenario():
        entered, sending = asyncio.Event(), asyncio.Event()
        received = []
        cancelled = asyncio.Event()

        async def collector(request):
            received.append(await request.aread())
            sending.set()
            if collector_hangs:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return httpx.Response(200)

        state, service = await allocation_service(
            tmp_path, OTLPHTTPAdapterFactory(httpx.MockTransport(collector))
        )
        spec = configured_allocation()
        spec.runtime_settings.telemetry.flush_timeout_seconds = 10
        await service.prepare(spec)
        context = allocation_context(service)
        instrumentation = context.adapter_host.handles.instrumentation
        instrumentation.start_span("contractor.worker.error").end(outcome="failed")
        loop = asyncio.get_running_loop()
        start = datetime.now(UTC)
        monotonic_start = loop.time()
        allocation_owner(service)._now = lambda: (
            start + timedelta(seconds=loop.time() - monotonic_start)
        )
        exits = []
        service._exit_process = lambda code: exits.append((code, allocation_owner(service)._now()))

        async def hanging_http():
            entered.set()
            await asyncio.Event().wait()

        context.http_transport = SimpleNamespace(close=hanging_http)
        deadline = start + timedelta(seconds=10)
        stop = asyncio.create_task(
            service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="trace-deadline",
                    deadline=deadline,
                )
            )
        )
        await entered.wait()
        advance_clock(5.1)
        await sending.wait()
        assert not exits
        if collector_hangs:
            advance_clock(2.5)
        with pytest.raises(TimeoutError):
            await stop
        assert len(received) == 1
        assert exits and exits[0][0] == 70 and exits[0][1] < deadline
        assert (await state.snapshot("allocation-1")).process_state is ProcessState.FENCED
        assert context.adapter_host.report_metrics()["otlp-http@1"].flush_succeeded is (
            not collector_hangs
        )
        if collector_hangs:
            assert cancelled.is_set()
        context.http_transport = None
        await context.worker.abort(deadline)
        await context.adapter_host.rollback(deadline=deadline, now=allocation_owner(service)._now)

    run_with_advanceable_clock(scenario())


def test_repeated_caller_cancellation_cannot_skip_emergency_flush(tmp_path):
    async def scenario():
        sending, finish = asyncio.Event(), asyncio.Event()
        requests = []

        async def collector(request):
            requests.append(await request.aread())
            sending.set()
            await finish.wait()
            return httpx.Response(200)

        state, service = await allocation_service(
            tmp_path, OTLPHTTPAdapterFactory(httpx.MockTransport(collector))
        )
        spec = configured_allocation()
        await service.prepare(spec)
        context = allocation_context(service)
        context.adapter_host.handles.instrumentation.start_span("contractor.worker.error").end(
            outcome="failed"
        )
        entered = asyncio.Event()

        async def hanging_http():
            entered.set()
            await asyncio.Event().wait()

        context.http_transport = SimpleNamespace(close=hanging_http)
        exits = []
        service._exit_process = exits.append
        deadline = datetime.now(UTC) + timedelta(seconds=10)
        stop = asyncio.create_task(
            service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="cancel-trace-flush",
                    deadline=deadline,
                )
            )
        )
        await entered.wait()
        stop.cancel()
        await sending.wait()
        stop.cancel()
        await asyncio.sleep(0)
        stop.cancel()
        await asyncio.sleep(0)
        assert not stop.done() and not exits
        assert (await state.snapshot("allocation-1")).process_state is ProcessState.FENCED
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await stop
        assert exits == [70] and len(requests) == 1
        assert context.adapter_host.report_metrics()["otlp-http@1"].flush_succeeded
        # The emergency path did not close adapters under still-owned producers.
        assert not context.adapter_host.closed
        context.http_transport = None
        await context.worker.abort(deadline)
        await context.adapter_host.rollback(deadline=deadline, now=lambda: datetime.now(UTC))

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [400, 503])
def test_export_rejection_does_not_replace_cleanup_failure(tmp_path, caplog, status):
    async def scenario():
        calls = []

        async def collector(request):
            calls.append(await request.aread())
            return httpx.Response(status, content=b"collector-response-secret-canary")

        state, service = await allocation_service(
            tmp_path, OTLPHTTPAdapterFactory(httpx.MockTransport(collector))
        )
        spec = configured_allocation()
        await service.prepare(spec)
        context = allocation_context(service)
        context.adapter_host.handles.instrumentation.start_span("contractor.worker.error").end(
            outcome="failed"
        )
        failure = RuntimeError("original-cleanup-secret-canary")

        async def fail_http():
            raise failure

        context.http_transport = SimpleNamespace(close=fail_http)
        exits = []
        service._exit_process = exits.append
        deadline = datetime.now(UTC) + timedelta(seconds=5)
        with pytest.raises(RuntimeError) as caught:
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="rejected-trace-flush",
                    deadline=deadline,
                )
            )
        assert caught.value is failure
        assert exits == [70] and len(calls) == (1 if status == 400 else 2)
        assert (await state.snapshot("allocation-1")).process_state is ProcessState.FENCED
        metrics = context.adapter_host.report_metrics()["otlp-http@1"]
        assert metrics.flush_attempted and not metrics.flush_succeeded
        assert "phase=http exception=RuntimeError" in caplog.text
        assert "secret-canary" not in caplog.text
        context.http_transport = None
        await context.worker.abort(deadline)
        await context.adapter_host.rollback(deadline=deadline, now=lambda: datetime.now(UTC))

    asyncio.run(scenario())
