"""Bounded content-free ``otlp-http@1`` Worker trace exporter."""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import httpx
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

from contractor_runtime import __version__
from contractor_runtime.adapters.host import (
    AdapterFactoryError,
    AdapterHandles,
    AdapterSettings,
    RuntimeAdapterBuildContext,
    RuntimeAdapterMetricsState,
)
from contractor_runtime.adapters.instrumentation import (
    RuntimeInstrumentation,
    RuntimeSpan,
    ScalarAttribute,
    TelemetryAttribute,
)
from contractor_runtime.contracts import RuntimeAdapterRef, TelemetrySettingsV2

MAX_PENDING_SPANS = 2048
MAX_PENDING_BYTES = 2 * 1024 * 1024
MAX_SPAN_ATTRIBUTES = 64
MAX_STRING_ATTRIBUTE_BYTES = 256
# One flattened RuntimeConfig chain may contain default + 32 Run + 32 Agent refs.
MAX_SEQUENCE_VALUES = 65
MAX_ATTRIBUTE_KEY_BYTES = 128
MAX_SPAN_NAME_BYTES = 128
MAX_RESPONSE_BYTES = 64 * 1024

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_SPAN_NAMES = frozenset(
    {
        "contractor.worker.a2a_task",
        "contractor.worker.error",
        "contractor.worker.model",
        "contractor.worker.tool",
    }
)
_ALLOWED_SPAN_ATTRIBUTES = frozenset(
    {
        "operation.kind",
        "model.alias",
        "tool.name",
        "outcome",
        "duration.ms",
        "error.type",
        "tokens.input",
        "tokens.output",
        "tokens.total",
        "tokens.cached_input",
        "counts.model_calls",
        "counts.tool_calls",
    }
)
_SAFE_OUTCOMES = frozenset({"cancelled", "failed", "rejected", "succeeded", "unavailable"})


class OTLPDeliveryError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("OTLP trace delivery failed")


class OTLPCloseError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("OTLP exporter close failed")


@dataclass(frozen=True, slots=True, repr=False)
class _QueuedSpan:
    encoded: bytes
    size: int


class _OTLPSpan(RuntimeSpan):
    def __init__(
        self,
        instrumentation: OTLPInstrumentation,
        name: str,
        attributes: Mapping[str, TelemetryAttribute],
        *,
        started_unix_ns: int,
        started_monotonic_ns: int,
    ) -> None:
        self._instrumentation = instrumentation
        self._name = name
        self._attributes = dict(attributes)
        self._started_unix_ns = started_unix_ns
        self._started_monotonic_ns = started_monotonic_ns
        self._ended = False

    def end(
        self,
        *,
        outcome: str,
        attributes: Mapping[str, TelemetryAttribute] | None = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        finished_monotonic_ns = self._instrumentation.monotonic_ns()
        duration_ns = max(0, finished_monotonic_ns - self._started_monotonic_ns)
        merged = dict(self._attributes)
        if attributes is not None:
            merged.update(attributes)
        merged["outcome"] = outcome if outcome in _SAFE_OUTCOMES else "failed"
        merged["duration.ms"] = duration_ns // 1_000_000
        self._instrumentation.enqueue(
            self._name,
            started_unix_ns=self._started_unix_ns,
            finished_unix_ns=self._started_unix_ns + duration_ns,
            attributes=merged,
        )
        self._attributes.clear()

    def __repr__(self) -> str:
        return f"_OTLPSpan(name={self._name!r}, ended={self._ended!r})"


class OTLPInstrumentation(RuntimeInstrumentation):
    """Synchronous span hook that only appends bounded encoded records."""

    def __init__(
        self,
        metrics: RuntimeAdapterMetricsState,
        resource_attributes: Mapping[str, TelemetryAttribute],
        *,
        secret_values: Sequence[str],
        wall_time_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        max_pending_spans: int = MAX_PENDING_SPANS,
        max_pending_bytes: int = MAX_PENDING_BYTES,
    ) -> None:
        self._metrics = metrics
        self._secret_values = tuple(value for value in secret_values if value)
        self._wall_time_ns = wall_time_ns
        self.monotonic_ns = monotonic_ns
        self._max_pending_spans = max_pending_spans
        self._max_pending_bytes = max_pending_bytes
        self._resource = Resource(
            attributes=_key_values(
                resource_attributes,
                allowed_keys=frozenset(resource_attributes),
                secrets=self._secret_values,
            )
        )
        self._resource_size = len(self._resource.SerializeToString())
        self._queue: list[_QueuedSpan] = []
        self._pending_bytes = self._resource_size
        self._closed = False

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, TelemetryAttribute] | None = None,
    ) -> RuntimeSpan:
        selected_name = name if name in _ALLOWED_SPAN_NAMES else "contractor.worker.error"
        selected = _safe_attributes(attributes or {}, self._secret_values)
        return _OTLPSpan(
            self,
            selected_name,
            selected,
            started_unix_ns=self._wall_time_ns(),
            started_monotonic_ns=self.monotonic_ns(),
        )

    def enqueue(
        self,
        name: str,
        *,
        started_unix_ns: int,
        finished_unix_ns: int,
        attributes: Mapping[str, TelemetryAttribute],
    ) -> None:
        if self._closed:
            return
        try:
            safe_attributes = _safe_attributes(attributes, self._secret_values)
            outcome = safe_attributes.get("outcome", "failed")
            span = Span(
                trace_id=os.urandom(16),
                span_id=os.urandom(8),
                name=_truncate_utf8(name, MAX_SPAN_NAME_BYTES),
                kind=Span.SPAN_KIND_INTERNAL,
                start_time_unix_nano=max(0, started_unix_ns),
                end_time_unix_nano=max(started_unix_ns, finished_unix_ns),
                attributes=_key_values(
                    safe_attributes,
                    allowed_keys=_ALLOWED_SPAN_ATTRIBUTES,
                    secrets=self._secret_values,
                ),
                status=Status(
                    code=(
                        Status.STATUS_CODE_OK
                        if outcome == "succeeded"
                        else Status.STATUS_CODE_ERROR
                    )
                ),
            )
            encoded = span.SerializeToString()
        except Exception:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            return
        size = len(encoded)
        if (
            len(self._queue) >= self._max_pending_spans
            or self._pending_bytes + size > self._max_pending_bytes
        ):
            self._metrics.record_operation(succeeded=False, error_code="queue_overflow")
            return
        self._queue.append(_QueuedSpan(encoded=encoded, size=size))
        self._pending_bytes += size

    def export_request(self) -> bytes:
        resource_spans = ResourceSpans(resource=self._resource)
        scope_spans = ScopeSpans()
        scope_spans.scope.name = "contractor.runtime.worker"
        scope_spans.scope.version = __version__
        for item in self._queue:
            scope_spans.spans.add().ParseFromString(item.encoded)
        resource_spans.scope_spans.append(scope_spans)
        return ExportTraceServiceRequest(resource_spans=[resource_spans]).SerializeToString()

    def clear(self) -> None:
        self._queue.clear()
        self._pending_bytes = self._resource_size

    def close(self) -> None:
        self.clear()
        self._secret_values = ()
        self._closed = True

    @property
    def pending_spans(self) -> int:
        return len(self._queue)

    @property
    def pending_bytes(self) -> int:
        return self._pending_bytes

    def __repr__(self) -> str:
        return (
            "OTLPInstrumentation("
            f"pending_spans={len(self._queue)}, pending_bytes={self._pending_bytes}, "
            f"closed={self._closed!r})"
        )


class OTLPHTTPAdapterFactory:
    ref = "otlp-http@1"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def probe(self) -> bool:
        try:
            ExportTraceServiceRequest().SerializeToString()
            client = httpx.AsyncClient(
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
            )
            await client.aclose()
        except Exception:
            return False
        return True

    async def create(
        self,
        context: RuntimeAdapterBuildContext,
        settings: AdapterSettings,
    ) -> OTLPHTTPAdapter:
        if not isinstance(settings, TelemetrySettingsV2):
            raise AdapterFactoryError(retryable=False)
        try:
            return OTLPHTTPAdapter(context, settings, transport=self._transport)
        except Exception:
            raise AdapterFactoryError(retryable=False) from None

    def __repr__(self) -> str:
        return "OTLPHTTPAdapterFactory(ref='otlp-http@1')"


class OTLPHTTPAdapter:
    ref: RuntimeAdapterRef = "otlp-http@1"

    def __init__(
        self,
        context: RuntimeAdapterBuildContext,
        settings: TelemetrySettingsV2,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.metrics = RuntimeAdapterMetricsState()
        secret_headers = {
            name: value.get_secret_value() for name, value in settings.headers.items()
        }
        secret_values = (*secret_headers.values(), settings.endpoint)
        resource_attributes: dict[str, TelemetryAttribute] = {
            "service.name": "contractor-runtime-worker",
            "service.version": __version__,
            "contractor.run.id": context.run_id,
            "contractor.stage_execution.id": context.stage_execution_id,
            "contractor.allocation.id": context.allocation_id,
            "contractor.worker.name": context.logical_agent_name,
            "contractor.runtime.config_refs": context.runtime_config_refs,
            "contractor.runtime.config_digests": context.runtime_config_digests,
            "contractor.run.labels": context.run_labels,
            "contractor.agent.labels": context.agent_labels,
            "contractor.runtime.adapter_refs": context.runtime_adapter_refs,
        }
        self._instrumentation = OTLPInstrumentation(
            self.metrics,
            resource_attributes,
            secret_values=secret_values,
        )
        self.handles = AdapterHandles(instrumentation=self._instrumentation)
        self._endpoint = settings.endpoint
        self._headers = httpx.Headers(secret_headers)
        self._headers["Content-Type"] = "application/x-protobuf"
        self._headers["Accept"] = "application/x-protobuf"
        timeout = float(min(context.request_timeout_seconds, settings.flush_timeout_seconds))
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )
        self._closed = False

    async def flush(self) -> None:
        client = self._client
        if self._closed or client is None or self._instrumentation.pending_spans == 0:
            return
        try:
            payload = self._instrumentation.export_request()
        except Exception:
            self.metrics.record_operation(succeeded=False, error_code="request_failed")
            raise OTLPDeliveryError from None
        failed = False
        try:
            async with client.stream(
                "POST",
                self._endpoint,
                headers=self._headers,
                content=payload,
            ) as response:
                if not 200 <= response.status_code < 300 or not await _accepted_response(response):
                    failed = True
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
        if failed:
            self.metrics.record_operation(succeeded=False, error_code="delivery_failed")
            raise OTLPDeliveryError from None
        self.metrics.record_operation(succeeded=True)
        self._instrumentation.clear()

    async def close(self) -> None:
        if self._closed:
            return
        client = self._client
        failed = False
        try:
            if client is not None:
                await client.aclose()
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
        finally:
            self._client = None
            self._headers.clear()
            self._endpoint = ""
            self._instrumentation.close()
            self.handles = AdapterHandles()
            self._closed = True
        if failed:
            raise OTLPCloseError from None

    def __repr__(self) -> str:
        return (
            "OTLPHTTPAdapter("
            f"ref={self.ref!r}, pending_spans={self._instrumentation.pending_spans}, "
            f"closed={self._closed!r})"
        )


def _safe_attributes(
    source: Mapping[str, TelemetryAttribute],
    secrets: Sequence[str],
) -> dict[str, TelemetryAttribute]:
    result: dict[str, TelemetryAttribute] = {}
    for key, value in source.items():
        if len(result) >= MAX_SPAN_ATTRIBUTES or key not in _ALLOWED_SPAN_ATTRIBUTES:
            continue
        selected = _safe_attribute_value(key, value, secrets)
        if selected is not None:
            result[key] = selected
    return result


async def _accepted_response(response: httpx.Response) -> bool:
    body = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                return False
            body.extend(chunk)
        decoded = ExportTraceServiceResponse.FromString(bytes(body))
    except (DecodeError, ValueError):
        return False
    partial = decoded.partial_success
    return partial.rejected_spans == 0 and not partial.error_message


def _safe_attribute_value(
    key: str,
    value: TelemetryAttribute,
    secrets: Sequence[str],
) -> TelemetryAttribute | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(-(2**63), min(2**63 - 1, value))
    if isinstance(value, str):
        if any(secret and secret in value for secret in secrets):
            return None
        selected = _truncate_utf8(value, MAX_STRING_ATTRIBUTE_BYTES)
        if key == "error.type" and _SAFE_IDENTIFIER.fullmatch(selected) is None:
            return "UnknownError"
        return selected
    if isinstance(value, Sequence):
        selected_values: list[ScalarAttribute] = []
        for item in value[:MAX_SEQUENCE_VALUES]:
            selected = _safe_attribute_value(key, item, secrets)
            if isinstance(selected, (str, int, bool)):
                selected_values.append(selected)
        return tuple(selected_values)
    return None


def _key_values(
    source: Mapping[str, TelemetryAttribute],
    *,
    allowed_keys: frozenset[str],
    secrets: Sequence[str],
) -> list[KeyValue]:
    result: list[KeyValue] = []
    for key, value in source.items():
        if len(result) >= MAX_SPAN_ATTRIBUTES or key not in allowed_keys:
            continue
        if len(key.encode("utf-8")) > MAX_ATTRIBUTE_KEY_BYTES:
            continue
        selected = _safe_attribute_value(key, value, secrets)
        if selected is None:
            continue
        result.append(KeyValue(key=key, value=_any_value(selected)))
    return result


def _any_value(value: TelemetryAttribute) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, str):
        return AnyValue(string_value=value)
    return AnyValue(array_value=ArrayValue(values=[_any_value(item) for item in value]))


def _truncate_utf8(value: str, maximum: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value
    return encoded[:maximum].decode("utf-8", errors="ignore")
