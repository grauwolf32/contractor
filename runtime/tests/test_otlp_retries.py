"""Exporter-level retry classification, absolute budgets and loss accounting."""

from __future__ import annotations

import asyncio
import gzip
import json
import ssl
import time
import zlib
from email.utils import formatdate
from pathlib import Path

import httpx
import pytest
from fakes.telemetry import SERVER_EXPORT
from jsonschema import Draft202012Validator
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
from referencing import Registry, Resource
from test_http_toolset import CorruptTLSRecordTarget
from test_otlp_adapter import adapter_context, telemetry_settings

from contractor_runtime import http_body
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
    export = SERVER_EXPORT.model_copy(
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


def tls_record_error() -> ssl.SSLError:
    return ssl.SSLError(1, "[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed")


class FailingBody(httpx.AsyncByteStream):
    """A response body that fails after its first bytes, as a reset or TLS record does."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aiter__(self):
        yield b"\x0a"
        raise self._error


TRANSPORT_FAILURES = {
    "tls-send": lambda: tls_record_error(),
    "reset-send": lambda: ConnectionResetError(54, "Connection reset by peer"),
    "tls-body": lambda: tls_record_error(),
    "reset-body": lambda: ConnectionResetError(54, "Connection reset by peer"),
}


def failing_collector(failure: str, payloads: list[bytes], failures: int):
    """A collector whose first failures attempts fail in the named phase."""

    async def collector(request):
        payloads.append(await request.aread())
        if len(payloads) > failures:
            return httpx.Response(200)
        error = TRANSPORT_FAILURES[failure]()
        if failure.endswith("-send"):
            raise error
        return httpx.Response(200, stream=FailingBody(error))

    return collector


@pytest.mark.parametrize("failure", sorted(TRANSPORT_FAILURES))
def test_tls_and_os_transport_failures_retry_exact_payload(failure: str) -> None:
    # httpcore lets OSError such as ssl.SSLError escape unmapped while sending or
    # reading the acknowledgement; it is a temporary transport failure (OBS-R74).
    async def scenario() -> None:
        payloads: list[bytes] = []
        adapter = await OTLPHTTPAdapterFactory(
            httpx.MockTransport(failing_collector(failure, payloads, failures=2))
        ).create(adapter_context(), settings())
        enqueue(adapter)
        await adapter.flush()
        await adapter.close()
        assert len(payloads) == 3 and payloads[0] == payloads[1] == payloads[2]
        assert adapter.metrics.failed_operations == 2
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", sorted(TRANSPORT_FAILURES))
def test_persistent_tls_and_os_transport_failures_exhaust_retries(failure: str) -> None:
    async def scenario() -> None:
        payloads: list[bytes] = []
        adapter = await OTLPHTTPAdapterFactory(
            httpx.MockTransport(failing_collector(failure, payloads, failures=3))
        ).create(adapter_context(), settings())
        enqueue(adapter)
        with pytest.raises(OTLPDeliveryError):
            await adapter.flush()
        await adapter.close()
        assert len(payloads) == 3
        assert adapter.metrics.dropped_spans.retry_exhausted == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_corrupt_tls_record_from_a_real_collector_exhausts_retries(
    tmp_path: Path, phase: str
) -> None:
    # A real TLS collector answers with a record the client cannot decrypt;
    # httpcore raises ssl.SSLError, which the exporter retries until its budget.
    async def scenario() -> None:
        collector = CorruptTLSRecordTarget(tmp_path, phase)
        try:
            adapter = await OTLPHTTPAdapterFactory(collector.transport()).create(
                adapter_context(),
                settings().model_copy(update={"endpoint": "https://target.example/v1/traces"}),
            )
            enqueue(adapter)
            with pytest.raises(OTLPDeliveryError):
                await adapter.flush()
            await adapter.close()
        finally:
            collector.close()
        assert collector.connections == 3
        assert adapter.metrics.dropped_spans.retry_exhausted == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    [
        lambda: httpx.UnsupportedProtocol("unsupported"),
        lambda: httpx.LocalProtocolError("local"),
        lambda: TypeError("implementation"),
        "decoding",
    ],
    ids=["unsupported-protocol", "local-protocol", "type-error", "decoding"],
)
def test_terminal_transport_and_implementation_failures_never_retry(failure) -> None:
    async def scenario() -> None:
        requests = []

        async def collector(request):
            requests.append(request)
            if failure == "decoding":
                # An acknowledgement that fails content decoding is malformed (OBS-R77).
                return httpx.Response(
                    200, headers={"Content-Encoding": "gzip"}, content=b"not gzip"
                )
            raise failure()

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


ACK_ENCODINGS = [
    pytest.param(None, lambda body: body, id="no-encoding"),
    pytest.param("identity", lambda body: body, id="identity"),
    pytest.param("gzip", gzip.compress, id="gzip"),
    pytest.param("deflate", zlib.compress, id="deflate"),
    pytest.param(
        "deflate", lambda body: zlib.compress(body, wbits=-zlib.MAX_WBITS), id="raw-deflate"
    ),
    pytest.param("deflate, gzip", lambda body: gzip.compress(zlib.compress(body)), id="layered"),
]


class AckStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, block: bool = False) -> None:
        self.chunks = chunks
        self.block = block
        self.reads = 0
        self.closes = 0
        self.entered = asyncio.Event()
        self.joined = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk
        if self.block:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.joined = True

    async def aclose(self) -> None:
        self.closes += 1


def acknowledgement(encoding: str, rejected: int = 0, message: str = "provider-secret") -> bytes:
    if encoding == "json":
        return json.dumps(
            {"partialSuccess": {"rejectedSpans": str(rejected), "errorMessage": message}}
        ).encode()
    response = ExportTraceServiceResponse()
    response.partial_success.rejected_spans = rejected
    response.partial_success.error_message = message
    return response.SerializeToString()


@pytest.mark.parametrize(("coding", "encode"), ACK_ENCODINGS)
@pytest.mark.parametrize("encoding", ["protobuf", "json"])
@pytest.mark.parametrize("rejected", [0, 1, 3])
def test_streamed_acknowledgements_preserve_partial_success(coding, encode, encoding, rejected):
    async def scenario() -> None:
        wire = encode(acknowledgement(encoding, rejected))
        stream = AckStream([wire[i : i + 7] for i in range(0, len(wire), 7)])
        headers = {"content-type": f"application/{encoding}"}
        if coding is not None:
            headers["content-encoding"] = coding
        calls = []

        def collector(request):
            calls.append(request)
            return httpx.Response(200, headers=headers, stream=stream)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        try:
            if rejected:
                with pytest.raises(OTLPDeliveryError):
                    await adapter.flush()
            else:
                await adapter.flush()
        finally:
            await adapter.close()
        assert len(calls) == 1 and stream.closes == 1
        assert adapter.metrics.dropped_spans.collector_rejected == rejected
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == rejected
        assert "provider-secret" not in repr(adapter.metrics)

    asyncio.run(scenario())


@pytest.mark.parametrize(("coding", "encode"), ACK_ENCODINGS)
def test_streamed_acknowledgement_accepts_exact_decoded_limit(coding, encode):
    async def scenario() -> None:
        body = b"{}" + b" " * (otlp_retry.MAX_RESPONSE_BYTES - 2)
        wire = encode(body)
        stream = AckStream([wire])
        headers = {"content-type": "application/json"}
        if coding is not None:
            headers["content-encoding"] = coding
        adapter = await OTLPHTTPAdapterFactory(
            httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, stream=stream))
        ).create(adapter_context(), settings())
        enqueue(adapter)
        try:
            await adapter.flush()
        finally:
            await adapter.close()
        assert stream.closes == 1 and stream.reads == 1
        assert adapter.metrics.failed_operations == 0
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(("coding", "encode"), ACK_ENCODINGS[2:])
@pytest.mark.parametrize("encoding", ["protobuf", "json"])
def test_streamed_compression_bomb_stops_decoding_at_limit(monkeypatch, coding, encode, encoding):
    # Observe the real zlib calls in both HTTPX and the shared reader: the old
    # aiter_bytes path requests unbounded output and produces the whole 4 MiB.
    wire = encode(acknowledgement(encoding, message="provider-secret" + "x" * (4 * 1024 * 1024)))
    assert len(wire) < otlp_retry.MAX_RESPONSE_BYTES
    decompressobj = zlib.decompressobj
    produced = []

    class DecoderProbe:
        def __init__(self, *args, **kwargs):
            self.decoder = decompressobj(*args, **kwargs)

        def decompress(self, data, max_length=0):
            output = self.decoder.decompress(data, max_length)
            produced.append((max_length, len(output)))
            return output

        def flush(self):
            return self.decoder.flush()

    monkeypatch.setattr(http_body.zlib, "decompressobj", DecoderProbe)

    async def scenario() -> None:
        stream = AckStream([wire, b"must not be read"])
        calls = []

        def collector(request):
            calls.append(request)
            return httpx.Response(
                200,
                headers={"content-encoding": coding, "content-type": f"application/{encoding}"},
                stream=stream,
            )

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        try:
            with pytest.raises(OTLPDeliveryError):
                await adapter.flush()
        finally:
            await adapter.close()
        assert len(calls) == 1 and stream.closes == 1 and stream.reads == 1
        assert adapter.metrics.failed_operations == 1
        assert adapter.metrics.dropped_spans.non_retryable == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3
        assert "provider-secret" not in repr(adapter.metrics)

    asyncio.run(scenario())
    assert produced
    assert all(0 < bound <= otlp_retry.MAX_RESPONSE_BYTES + 1 for bound, _ in produced)
    assert max(size for _, size in produced) <= otlp_retry.MAX_RESPONSE_BYTES + 1


@pytest.mark.parametrize("coding", ["identity", "gzip"])
def test_streamed_received_overflow_is_terminal_even_with_tiny_decoded_body(coding):
    async def scenario() -> None:
        prefix = gzip.compress(b"{}") if coding == "gzip" else b"{}"
        padding = b"\0" if coding == "gzip" else b" "
        stream = AckStream(
            [prefix + padding * (otlp_retry.MAX_RESPONSE_BYTES - len(prefix)), padding, padding]
        )
        calls = []

        def collector(request):
            calls.append(request)
            return httpx.Response(
                200,
                headers={"content-encoding": coding, "content-type": "application/json"},
                stream=stream,
            )

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        try:
            with pytest.raises(OTLPDeliveryError):
                await adapter.flush()
        finally:
            await adapter.close()
        assert len(calls) == 1 and stream.closes == 1 and stream.reads == 2
        assert adapter.metrics.dropped_spans.non_retryable == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("coding", ["gzip", "deflate", "deflate, gzip"])
def test_streamed_corrupt_compression_is_terminal(coding):
    async def scenario() -> None:
        stream = AckStream([b"provider-secret", b"must not be read"])
        calls = []

        def collector(request):
            calls.append(request)
            return httpx.Response(200, headers={"content-encoding": coding}, stream=stream)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        try:
            with pytest.raises(OTLPDeliveryError):
                await adapter.flush()
        finally:
            await adapter.close()
        assert len(calls) == 1 and stream.closes == 1 and stream.reads == 1
        assert adapter.metrics.dropped_spans.non_retryable == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3
        assert "provider-secret" not in repr(adapter.metrics)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (500, False), (429, True), (502, True), (503, True), (504, True)],
)
def test_streamed_status_classification_precedes_acknowledgement_decoding(status, retryable):
    async def scenario() -> None:
        streams = [AckStream([b"provider-secret: invalid gzip"]), AckStream([gzip.compress(b"")])]
        calls = []

        def collector(request):
            index = len(calls)
            calls.append(request)
            return httpx.Response(
                status if index == 0 else 200,
                headers={"content-encoding": "gzip", "retry-after": "0"},
                stream=streams[index],
            )

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        enqueue(adapter)
        try:
            if retryable:
                await adapter.flush()
            else:
                with pytest.raises(OTLPDeliveryError):
                    await adapter.flush()
        finally:
            await adapter.close()
        assert streams[0].reads == 0 and streams[0].closes == 1
        assert len(calls) == (2 if retryable else 1)
        if retryable:
            assert streams[1].reads == 1 and streams[1].closes == 1
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == (0 if retryable else 3)
        assert "provider-secret" not in repr(adapter.metrics)

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["cancel", "shutdown", "batch-deadline", "flush-deadline"])
def test_streamed_compressed_acknowledgement_interruption_joins_and_closes(interruption):
    async def scenario() -> None:
        stream = AckStream([gzip.compress(b"{}")[0:2]], block=True)
        calls = []

        def collector(request):
            calls.append(request)
            return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=stream)

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings()
        )
        if interruption == "batch-deadline":
            adapter._batch_timeout = 0.05
        if interruption == "flush-deadline":
            adapter._flush_timeout = 0.05
        enqueue(adapter)
        flushing = asyncio.create_task(adapter.flush())
        try:
            await asyncio.wait_for(stream.entered.wait(), timeout=1)
            if interruption == "cancel":
                flushing.cancel()
            elif interruption == "shutdown":
                await adapter.close()
            expected_error = {
                "cancel": asyncio.CancelledError,
                "shutdown": asyncio.CancelledError,
                "batch-deadline": OTLPDeliveryError,
                "flush-deadline": TimeoutError,
            }[interruption]
            with pytest.raises(expected_error):
                await asyncio.wait_for(flushing, timeout=1)
            if interruption == "flush-deadline":
                # The exporter's flush deadline must fire; the outer test
                # watchdog would cancel the task instead of completing it.
                assert not flushing.cancelled()
                assert isinstance(flushing.exception(), TimeoutError)
        finally:
            await adapter.close()
        assert len(calls) == 1 and stream.reads == 1 and stream.closes == 1 and stream.joined
        reason = {
            "cancel": "cancelled",
            "shutdown": "shutdown",
            "batch-deadline": "deadline_exceeded",
            "flush-deadline": "cancelled",
        }[interruption]
        assert getattr(adapter.metrics.dropped_spans, reason) == 3
        assert sum(adapter.metrics.dropped_spans.model_dump().values()) == 3
        assert adapter._instrumentation.pending_spans == 0

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


def test_shared_drop_counters_round_trip_in_runtime_report_and_published_schema() -> None:
    metrics = RuntimeAdapterMetricsState()
    metrics.record_dropped_spans("queue_overflow", 2**64 - 1)
    metrics.record_dropped_spans("collector_rejected", 3)
    report = RuntimeReport(complete=True, adapters={"otlp-http@1": metrics.snapshot()})
    encoded = report.model_dump_json(by_alias=True, exclude_none=True)
    assert RuntimeReport.model_validate_json(encoded) == report
    schema_root = Path(__file__).parents[2] / "api/v1alpha1"
    schemas = {
        path.name: json.loads(path.read_bytes()) for path in schema_root.glob("*.schema.json")
    }
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    runtime = json.loads(encoded)
    Draft202012Validator(schemas["runtime-report.schema.json"], registry=registry).validate(runtime)
    final = json.loads(
        (
            Path(__file__).parents[2] / "api/testdata/v1alpha1/valid/allocation-final-response.json"
        ).read_bytes()
    )
    final["report"]["runtime"] = runtime
    Draft202012Validator(schemas["allocation.schema.json"], registry=registry).validate(final)


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
