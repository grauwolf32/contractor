"""Bounded ``otlp-http@1`` exporter with explicit trusted-sink content capture."""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

from contractor_runtime import __version__
from contractor_runtime.adapters.content import MAX_CONTENT_BYTES
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
from contractor_runtime.adapters.otlp_retry import (
    DeliveryResponse,
    backoff_seconds,
    classify_response,
)
from contractor_runtime.contracts import (
    MAX_RUN_METADATA_LABEL_VALUE_BYTES,
    MAX_RUN_METADATA_LABELS,
    RUN_METADATA_LABEL_KEY_PATTERN,
    DroppedSpanCounts,
    RuntimeAdapterRef,
    TelemetryExportSettings,
    TelemetryRetrySettings,
    TelemetrySettingsV2,
)

MAX_SPAN_ATTRIBUTES = 64
MAX_STRING_ATTRIBUTE_BYTES = 256
# One flattened RuntimeConfig chain may contain default + 32 Run + 32 Agent refs.
MAX_SEQUENCE_VALUES = 65
MAX_ATTRIBUTE_KEY_BYTES = 128
MAX_SPAN_NAME_BYTES = 128
RUN_METADATA_LABEL_ATTRIBUTE_PREFIX = "contractor.run.label."

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
        self.capture_content = instrumentation.capture_content
        self._content: dict[str, str] = {}

    def set_content(self, field: str, value: str) -> None:
        if (
            self.capture_content
            and not self._ended
            and field in {"input", "output"}
            and len(value.encode("utf-8")) <= MAX_CONTENT_BYTES
        ):
            self._content["langfuse.observation." + field] = value

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
            content=self._content,
        )
        self._attributes.clear()
        self._content.clear()

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
        run_metadata_labels: Mapping[str, str] | None = None,
        wall_time_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        max_pending_spans: int | None = None,
        max_pending_bytes: int | None = None,
        capture_content: bool = False,
        on_enqueue: Callable[[], None] | None = None,
    ) -> None:
        self.capture_content = capture_content
        self._metrics = metrics
        self._secret_values = tuple(value for value in secret_values if value)
        self._wall_time_ns = wall_time_ns
        self.monotonic_ns = monotonic_ns
        defaults = TelemetryExportSettings.defaults()
        self._max_pending_spans = (
            max_pending_spans if max_pending_spans is not None else defaults.max_pending_spans
        )
        self._max_pending_bytes = (
            max_pending_bytes if max_pending_bytes is not None else defaults.max_pending_bytes
        )
        self._on_enqueue = on_enqueue
        self._trace_id = os.urandom(16)
        self._run_metadata_attributes = MappingProxyType(
            _run_metadata_attributes(run_metadata_labels or {}, self._secret_values)
        )
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
        self.queue_full = False
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
        content: Mapping[str, str] | None = None,
    ) -> None:
        if self._closed:
            self._metrics.record_dropped_spans("shutdown", 1)
            return
        try:
            safe_attributes = _safe_attributes(attributes, self._secret_values)
            if name == "contractor.worker.a2a_task":
                safe_attributes.update(self._run_metadata_attributes)
            outcome = safe_attributes.get("outcome", "failed")
            allowed_keys = _ALLOWED_SPAN_ATTRIBUTES
            if name == "contractor.worker.a2a_task":
                allowed_keys = allowed_keys | frozenset(self._run_metadata_attributes)
            span = Span(
                trace_id=self._trace_id,
                span_id=os.urandom(8),
                name=_truncate_utf8(name, MAX_SPAN_NAME_BYTES),
                kind=Span.SPAN_KIND_INTERNAL,
                start_time_unix_nano=max(0, started_unix_ns),
                end_time_unix_nano=max(started_unix_ns, finished_unix_ns),
                attributes=_key_values(
                    safe_attributes,
                    allowed_keys=allowed_keys,
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
            if name == "contractor.worker.model":
                span.attributes.add(
                    key="langfuse.observation.type", value=AnyValue(string_value="generation")
                )
                if "model.alias" in safe_attributes:
                    span.attributes.add(
                        key="gen_ai.request.model",
                        value=AnyValue(string_value=str(safe_attributes["model.alias"])),
                    )
            if self.capture_content:
                for key, value in (content or {}).items():
                    if (
                        key in {"langfuse.observation.input", "langfuse.observation.output"}
                        and len(value.encode("utf-8")) <= MAX_CONTENT_BYTES
                    ):
                        # Content goes directly to the explicitly trusted sink,
                        # never through metadata sanitization or log fields.
                        span.attributes.add(key=key, value=AnyValue(string_value=value))
            encoded = span.SerializeToString()
        except Exception:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            self._metrics.record_dropped_spans("encoding_failed", 1)
            return
        size = len(encoded)
        if (
            len(self._queue) >= self._max_pending_spans
            or self._pending_bytes + size > self._max_pending_bytes
        ):
            self._metrics.record_dropped_spans("queue_overflow", 1)
            self._metrics.record_operation(succeeded=False, error_code="queue_overflow")
            self.queue_full = True
            if self._on_enqueue is not None:
                self._on_enqueue()
            return
        self._queue.append(_QueuedSpan(encoded=encoded, size=size))
        self._pending_bytes += size
        if self._on_enqueue is not None:
            self._on_enqueue()

    def export_request(self, *, span_count: int | None = None) -> bytes:
        resource_spans = ResourceSpans(resource=self._resource)
        scope_spans = ScopeSpans()
        scope_spans.scope.name = "contractor.runtime.worker"
        scope_spans.scope.version = __version__
        for item in self._queue[:span_count]:
            scope_spans.spans.add().ParseFromString(item.encoded)
        resource_spans.scope_spans.append(scope_spans)
        return ExportTraceServiceRequest(resource_spans=[resource_spans]).SerializeToString()

    def batch_span_count(self, maximum_bytes: int) -> int:
        size = self._resource_size
        count = 0
        for item in self._queue:
            if count and size + item.size > maximum_bytes:
                break
            size += item.size
            count += 1
        return count

    def discard_prefix(self, count: int) -> None:
        # The sending prefix stays charged to both queue limits during I/O.
        # Appends during delivery belong to the next batch and must survive.
        self._pending_bytes -= sum(item.size for item in self._queue[:count])
        del self._queue[:count]
        self.queue_full = False

    def clear(self) -> None:
        self._queue.clear()
        self._pending_bytes = self._resource_size
        self.queue_full = False

    def close(self) -> None:
        self._metrics.record_dropped_spans("shutdown", self.pending_spans)
        self.clear()
        self._on_enqueue = None
        self._secret_values = ()
        self._trace_id = b""
        self._run_metadata_attributes = MappingProxyType({})
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
        self.metrics = RuntimeAdapterMetricsState(dropped_spans=DroppedSpanCounts())
        self._export_settings = (settings.export or TelemetryExportSettings.defaults()).model_copy(
            deep=True
        )
        self._retry_settings = (
            self._export_settings.retry or TelemetryRetrySettings.defaults()
        ).model_copy(deep=True)
        self._batch_timeout = float(settings.flush_timeout_seconds)
        self._flush_timeout = float(settings.flush_timeout_seconds)
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
            run_metadata_labels=context.run_metadata_labels,
            capture_content=settings.capture_content,
            on_enqueue=self._schedule_export,
            max_pending_spans=self._export_settings.max_pending_spans,
            max_pending_bytes=self._export_settings.max_pending_bytes,
        )
        self.handles = AdapterHandles(instrumentation=self._instrumentation)
        self._endpoint = settings.endpoint
        self._headers = httpx.Headers(secret_headers)
        self._headers["Content-Type"] = "application/x-protobuf"
        self._headers["Accept"] = "application/x-protobuf"
        timeout = float(min(context.request_timeout_seconds, settings.flush_timeout_seconds))
        self._delivery_timeout = timeout
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )
        self._closed = False
        self._closing = False
        self._draining = False
        self._export_task: asyncio.Task[bool] | None = None

    def _schedule_export(self) -> None:
        if (
            self._closed
            or self._closing
            or self._draining
            or not self._instrumentation.pending_spans
            or not self._batch_ready()
        ):
            return
        if self._export_task is None or self._export_task.done():
            self._start_export()

    def _batch_ready(self) -> bool:
        return (
            self._instrumentation.pending_bytes >= self._export_settings.batch_size_bytes
            or self._instrumentation.pending_spans >= self._export_settings.max_pending_spans
            or self._instrumentation.queue_full
        )

    def _start_export(self) -> asyncio.Task[bool]:
        task = asyncio.get_running_loop().create_task(
            self._export_batches(), name="allocation-otlp-export"
        )
        self._export_task = task
        return task

    async def _export_batches(self) -> bool:
        succeeded = True
        while (
            not self._closing
            and self._instrumentation.pending_spans
            and (self._draining or self._batch_ready())
        ):
            try:
                await self._send_batch()
            except OTLPDeliveryError:
                # After all attempts fail, release this batch so subsequent
                # spans can progress without an unbounded retry loop.
                succeeded = False
        return succeeded

    async def flush(self) -> None:
        if self._closed or self._closing:
            return
        # Join the existing sender and let it drain even a sub-threshold tail.
        # Cancellation from the lifecycle deadline also cancels its active POST.
        self._draining = True
        task = self._export_task
        if task is None or task.done():
            if self._instrumentation.pending_spans == 0:
                return
            task = self._start_export()
        async with asyncio.timeout(self._flush_timeout):
            if not await task:
                raise OTLPDeliveryError

    async def _send_batch(self) -> None:
        client = self._client
        if client is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._batch_timeout
        batch_size = self._export_settings.batch_size_bytes
        count = self._instrumentation.batch_span_count(batch_size)
        dropped = count
        reason = "encoding_failed"
        try:
            try:
                payload = self._instrumentation.export_request(span_count=count)
                while len(payload) > batch_size and count > 1:
                    count -= 1
                    payload = self._instrumentation.export_request(span_count=count)
                dropped = count
                if len(payload) > batch_size:
                    raise ValueError("OTLP batch exceeds its bound")
            except Exception:
                dropped = count
                self.metrics.record_operation(succeeded=False, error_code="request_failed")
                raise OTLPDeliveryError from None
            reason = "retry_exhausted"
            async with asyncio.timeout_at(deadline):
                for attempt in range(self._export_settings.max_attempts):
                    if loop.time() >= deadline:
                        reason = "deadline_exceeded"
                        raise OTLPDeliveryError
                    result = await self._attempt(client, payload, count)
                    if result.accepted:
                        dropped = result.rejected_spans
                        reason = "collector_rejected"
                        if dropped:
                            raise OTLPDeliveryError
                        return
                    if not result.retryable:
                        reason = "non_retryable"
                        raise OTLPDeliveryError
                    if attempt + 1 == self._export_settings.max_attempts:
                        break
                    delay = result.retry_after
                    if delay is None:
                        delay = backoff_seconds(self._retry_settings, attempt)
                    # A collector delay cannot be shortened to squeeze in another
                    # request. Give up this batch when the delay cannot fit.
                    if delay >= deadline - loop.time():
                        reason = "deadline_exceeded"
                        raise OTLPDeliveryError
                    await asyncio.sleep(delay)
            raise OTLPDeliveryError
        except TimeoutError:
            reason = "deadline_exceeded"
            raise OTLPDeliveryError from None
        except asyncio.CancelledError:
            reason = "shutdown" if self._closing else "cancelled"
            raise
        finally:
            # Attempts reuse exactly this prefix; concurrent arrivals remain queued.
            self.metrics.record_dropped_spans(reason, dropped)
            self._instrumentation.discard_prefix(count)

    async def _attempt(
        self, client: httpx.AsyncClient, payload: bytes, count: int
    ) -> DeliveryResponse:
        try:
            async with asyncio.timeout(self._delivery_timeout):
                async with client.stream(
                    "POST", self._endpoint, headers=self._headers, content=payload
                ) as response:
                    result = await classify_response(response, span_count=count)
        except asyncio.CancelledError:
            self.metrics.record_operation(succeeded=False, error_code="delivery_failed")
            raise
        except (httpx.UnsupportedProtocol, httpx.LocalProtocolError):
            result = DeliveryResponse()
        except (httpx.TransportError, TimeoutError):
            result = DeliveryResponse(retryable=True)
        except Exception:
            result = DeliveryResponse()
        succeeded = result.accepted and result.rejected_spans == 0
        self.metrics.record_operation(
            succeeded=succeeded, error_code=None if succeeded else "delivery_failed"
        )
        return result

    async def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        client = self._client
        failed = False
        try:
            if self._export_task is not None:
                self._export_task.cancel()
                await asyncio.gather(self._export_task, return_exceptions=True)
            if client is not None:
                await client.aclose()
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
        finally:
            self._export_task = None
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


def _run_metadata_attributes(source: Mapping[str, str], secrets: Sequence[str]) -> dict[str, str]:
    if len(source) > MAX_RUN_METADATA_LABELS:
        return {}
    result: dict[str, str] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return {}
        attribute_key = RUN_METADATA_LABEL_ATTRIBUTE_PREFIX + key
        if (
            not key
            or len(attribute_key.encode("utf-8")) > MAX_ATTRIBUTE_KEY_BYTES
            or RUN_METADATA_LABEL_KEY_PATTERN.fullmatch(key) is None
            or key.startswith("contractor.")
            or "\0" in value
            or len(value.encode("utf-8")) > MAX_RUN_METADATA_LABEL_VALUE_BYTES
        ):
            return {}
        if any(secret and secret in value for secret in secrets):
            continue
        result[attribute_key] = value
    return result


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
