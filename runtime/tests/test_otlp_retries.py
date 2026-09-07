"""Exporter-level retry classification, absolute budgets and loss accounting."""

from __future__ import annotations

import asyncio
import json
import time
from email.utils import formatdate
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
from test_otlp_adapter import adapter_context, telemetry_settings

from contractor_runtime.adapters import otlp_retry
from contractor_runtime.adapters.host import RuntimeAdapterMetricsState
from contractor_runtime.adapters.otlp_http import OTLPDeliveryError, OTLPHTTPAdapterFactory
from contractor_runtime.contracts import (
    DroppedSpanCounts,
    RuntimeReport,
    TelemetryExportSettings,
    TelemetryRetrySettings,
)


def settings(*, attempts: int = 3, initial: int = 10, maximum: int = 40):
    export = TelemetryExportSettings.defaults().model_copy(
        update={
            "max_attempts": attempts,
            "retry": TelemetryRetrySettings(
                initialBackoffMilliseconds=initial, maxBackoffMilliseconds=maximum
            ),
        }
    )
    return telemetry_settings().model_copy(update={"export": export, "flush_timeout_seconds": 3})


def enqueue(adapter, count: int = 3) -> None:
    for _ in range(count):
        adapter.handles.instrumentation.start_span("contractor.worker.tool", attributes={}).end(
            outcome="succeeded"
        )


@pytest.mark.parametrize("status", [302, 400, 401, 403, 404, 413, 418, 500, 501, 505])
def test_permanent_http_failures_never_retry(status: int) -> None:
    async def scenario() -> None:
        requests = []

        async def collector(request):
            requests.append(request)
            return httpx.Response(status, content=b"sensitive-provider-body")

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        with pytest.raises(OTLPDeliveryError):
            await adapter.flush()
        await adapter.close()
        assert len(requests) == 1
        assert adapter.metrics.dropped_spans.non_retryable == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3
        assert "sensitive-provider-body" not in repr(adapter.metrics)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [429, 502, 503, 504, "connect", "read"])
def test_transient_failure_retries_exact_payload(failure) -> None:
    async def scenario() -> None:
        payloads = []

        async def collector(request):
            payloads.append(await request.aread())
            if len(payloads) < 3:
                if failure == "connect":
                    raise httpx.ConnectError("secret connection error")
                if failure == "read":
                    raise httpx.ReadTimeout("secret timeout")
                return httpx.Response(failure)
            return httpx.Response(200)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        await adapter.flush()
        await adapter.close()
        assert len(payloads) == 3 and payloads[0] == payloads[1] == payloads[2]
        assert adapter.metrics.failed_operations == 2
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("encoding", ["protobuf", "json"])
@pytest.mark.parametrize("rejected", [0, 1, 3])
def test_partial_success_is_terminal_and_counts_only_rejected(encoding: str, rejected: int) -> None:
    async def scenario() -> None:
        requests = []

        async def collector(request):
            requests.append(request)
            if encoding == "json":
                return httpx.Response(
                    200,
                    json={
                        "partialSuccess": {
                            "rejectedSpans": str(rejected),
                            "errorMessage": "provider-secret",
                        }
                    },
                )
            response = ExportTraceServiceResponse()
            response.partial_success.rejected_spans = rejected
            response.partial_success.error_message = "provider-secret"
            return httpx.Response(200, content=response.SerializeToString())

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        if rejected:
            with pytest.raises(OTLPDeliveryError):
                await adapter.flush()
        else:
            await adapter.flush()
        await adapter.close()
        assert len(requests) == 1
        assert adapter.metrics.dropped_spans.collector_rejected == rejected
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == rejected
        assert "provider-secret" not in str(adapter.metrics.snapshot().model_dump())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"invalid protobuf"),
        httpx.Response(200, json={"partialSuccess": {"rejectedSpans": 4}}),
        httpx.Response(200, json={"partialSuccess": {"rejectedSpans": -1}}),
        httpx.Response(200, json={"partialSuccess": {"rejectedSpans": True}}),
        httpx.Response(200, content=b"x" * (otlp_retry.MAX_RESPONSE_BYTES + 1)),
    ],
)
def test_invalid_ack_is_not_retried(response: httpx.Response) -> None:
    async def scenario() -> None:
        requests = []

        async def collector(request):
            requests.append(request)
            return response

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        with pytest.raises(OTLPDeliveryError):
            await adapter.flush()
        await adapter.close()
        assert len(requests) == 1 and adapter.metrics.dropped_spans.non_retryable == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("header", [None, "", "-1", "1.5", "not a date", "9" * 129])
def test_invalid_retry_after_uses_fallback(header) -> None:
    assert otlp_retry.retry_after_seconds(header, now=0) is None


def test_retry_after_parser_and_backoff_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    assert otlp_retry.retry_after_seconds("12", now=0) == 12
    assert otlp_retry.retry_after_seconds("0", now=0) == 0
    assert otlp_retry.retry_after_seconds(formatdate(100, usegmt=True), now=95) == 5
    assert otlp_retry.retry_after_seconds(formatdate(100, usegmt=True), now=105) == 0
    selected = TelemetryRetrySettings(initialBackoffMilliseconds=17, maxBackoffMilliseconds=45)
    bounds = []

    def uniform(low, high):
        bounds.append((low, high))
        return high

    monkeypatch.setattr(otlp_retry.random, "uniform", uniform)
    assert [otlp_retry.backoff_seconds(selected, n) for n in range(4)] == [
        0.017,
        0.034,
        0.045,
        0.045,
    ]
    assert all(low == high / 2 for low, high in bounds)


@pytest.mark.parametrize("kind", ["seconds", "date"])
def test_exporter_obeys_retry_after(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fixed wall clock keeps the date case short without altering loop time.
    if kind == "date":
        monkeypatch.setattr(otlp_retry.time, "time", lambda: 100.95)

    async def scenario() -> None:
        times = []

        async def collector(request):
            times.append(time.monotonic())
            if len(times) == 1:
                return httpx.Response(
                    503,
                    headers={
                        "Retry-After": "1" if kind == "seconds" else formatdate(101, usegmt=True)
                    },
                )
            return httpx.Response(200)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        await adapter.flush()
        await adapter.close()
        assert len(times) == 2
        assert times[1] - times[0] >= (1 if kind == "seconds" else 0.049)

    asyncio.run(scenario())


@pytest.mark.parametrize("header", ["10", "9" * 128])
def test_retry_after_cannot_extend_deadline(header: str) -> None:
    async def scenario() -> None:
        calls = []

        async def collector(request):
            calls.append(request)
            return httpx.Response(429, headers={"Retry-After": header})

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        adapter._batch_timeout = 0.1
        enqueue(adapter)
        start = time.monotonic()
        with pytest.raises(OTLPDeliveryError):
            await adapter.flush()
        assert time.monotonic() - start < 0.5
        assert len(calls) == 1
        assert adapter.metrics.dropped_spans.deadline_exceeded == 3
        await adapter.close()
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3

    asyncio.run(scenario())


def test_retries_share_one_batch_budget() -> None:
    async def scenario() -> None:
        calls = []

        async def collector(request):
            calls.append(request)
            await asyncio.sleep(0.025)
            return httpx.Response(503)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings(attempts=10, initial=20, maximum=20)
        )
        adapter._batch_timeout = 0.08
        enqueue(adapter)
        start = time.monotonic()
        with pytest.raises(OTLPDeliveryError):
            await adapter.flush()
        assert time.monotonic() - start < 0.5
        assert 1 <= len(calls) < 10
        assert adapter.metrics.dropped_spans.deadline_exceeded == 3
        await adapter.close()

    asyncio.run(scenario())


def test_cancel_backoff_is_prompt_and_never_resends() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        calls = []

        async def collector(request):
            calls.append(request)
            entered.set()
            return httpx.Response(503, headers={"Retry-After": "2"})

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        flushing = asyncio.create_task(adapter.flush())
        await entered.wait()
        await asyncio.sleep(0)
        flushing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        await adapter.close()
        assert len(calls) == 1
        assert adapter.metrics.dropped_spans.cancelled == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3

    asyncio.run(scenario())


def test_overflow_and_shutdown_are_counted_once() -> None:
    async def scenario() -> None:
        configured = settings()
        configured.export.max_pending_spans = 3
        adapter = await OTLPHTTPAdapterFactory(
            httpx.MockTransport(lambda r: httpx.Response(200))
        ).create(adapter_context(), configured)
        enqueue(adapter, 4)
        await adapter.close()
        await adapter.close()
        assert adapter.metrics.dropped_spans.queue_overflow == 1
        assert adapter.metrics.dropped_spans.shutdown == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 4

    asyncio.run(scenario())


def test_drop_counters_saturate_and_reports_are_detached() -> None:
    metrics = RuntimeAdapterMetricsState()
    assert metrics.snapshot().dropped_spans is None
    metrics.record_dropped_spans("queue_overflow", 2**64 - 1)
    snapshot = metrics.snapshot()
    metrics.record_dropped_spans("queue_overflow", 100)
    metrics.record_dropped_spans("shutdown", 3)
    assert metrics.dropped_spans.queue_overflow == 2**64 - 1
    assert snapshot.dropped_spans.shutdown == 0
    for value in [-1, 2**64, 1.5, True, "3"]:
        with pytest.raises(ValueError):
            DroppedSpanCounts(queueOverflow=value)


def test_exhaustion_counts_spans_once_and_uses_configured_backoff(monkeypatch) -> None:
    ceilings = []

    def uniform(low, high):
        ceilings.append(high)
        return high

    monkeypatch.setattr(otlp_retry.random, "uniform", uniform)

    async def scenario() -> None:
        adapter = await OTLPHTTPAdapterFactory(
            httpx.MockTransport(lambda r: httpx.Response(503, headers={"Retry-After": "bad"}))
        ).create(adapter_context(), settings(attempts=4, initial=17, maximum=30))
        enqueue(adapter)
        with pytest.raises(OTLPDeliveryError):
            await adapter.flush()
        await adapter.close()
        assert adapter.metrics.operations == 4
        assert adapter.metrics.dropped_spans.retry_exhausted == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3
        assert ceilings == [0.017, 0.03, 0.03]

    asyncio.run(scenario())


def test_encoding_failure_discards_batch_without_http(monkeypatch) -> None:
    async def scenario() -> None:
        adapter = await OTLPHTTPAdapterFactory(
            httpx.MockTransport(lambda r: pytest.fail("encoding failure must not send"))
        ).create(adapter_context(), settings())
        enqueue(adapter)

        def broken_encoding(**kwargs):
            raise ValueError("secret encoding diagnostic")

        monkeypatch.setattr(adapter._instrumentation, "export_request", broken_encoding)
        with pytest.raises(OTLPDeliveryError):
            await adapter.flush()
        await adapter.close()
        assert adapter.metrics.dropped_spans.encoding_failed == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3
        assert "secret encoding diagnostic" not in repr(adapter.metrics)

    asyncio.run(scenario())


def test_close_counts_inflight_batch_and_new_tail_once() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()

        async def collector(request):
            entered.set()
            await asyncio.Event().wait()

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        flushing = asyncio.create_task(adapter.flush())
        await entered.wait()
        enqueue(adapter, 2)
        await adapter.close()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        await adapter.close()
        assert adapter.metrics.dropped_spans.shutdown == 5
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 5
        assert adapter._instrumentation.pending_spans == 0

    asyncio.run(scenario())


def test_drop_counters_round_trip_in_runtime_report_and_published_schema() -> None:
    metrics = RuntimeAdapterMetricsState()
    metrics.record_dropped_spans("queue_overflow", 2**64 - 1)
    metrics.record_dropped_spans("collector_rejected", 3)
    report = RuntimeReport(complete=True, adapters={"otlp-http@1": metrics.snapshot()})
    encoded = report.model_dump_json(by_alias=True, exclude_none=True)
    assert RuntimeReport.model_validate_json(encoded) == report
    schema = json.loads(
        (Path(__file__).parents[2] / "api/private-v2/runtime-report.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(json.loads(encoded))
    final_schema = json.loads(
        (Path(__file__).parents[2] / "api/v1alpha1/allocation.schema.json").read_text()
    )
    Draft202012Validator(final_schema["$defs"]["runtimeAdapterMetrics"]).validate(
        json.loads(encoded)["adapters"]["otlp-http@1"]
    )


@pytest.mark.parametrize(
    "retry",
    [
        {"initialBackoffMilliseconds": 0, "maxBackoffMilliseconds": 1},
        {"initialBackoffMilliseconds": 10, "maxBackoffMilliseconds": 1},
        {"initialBackoffMilliseconds": True, "maxBackoffMilliseconds": 10},
        {"initialBackoffMilliseconds": 1, "maxBackoffMilliseconds": 60001},
    ],
)
def test_runtime_rejects_invalid_retry_settings(retry) -> None:
    encoded = settings().export.model_dump(by_alias=True)
    encoded["retry"] = retry
    with pytest.raises(ValueError):
        TelemetryExportSettings.model_validate(encoded)
