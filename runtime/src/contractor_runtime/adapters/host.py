"""One bounded lifecycle owner for allocation-scoped infrastructure adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from contractor_runtime.adapters.instrumentation import RuntimeInstrumentation
from contractor_runtime.contracts import (
    AllocationSpecV2,
    CaidoSettingsV2,
    DroppedSpanCounts,
    HTTPProxySettingsV2,
    RuntimeAdapterMetricsV2,
    RuntimeAdapterRef,
    RuntimeSettingsV2,
    TelemetrySettingsV2,
)

UINT64_MAX = 2**64 - 1
SAFE_ERROR_CODES = frozenset(
    {
        "close_failed",
        "delivery_failed",
        "flush_failed",
        "flush_timeout",
        "queue_overflow",
        "request_failed",
    }
)


@dataclass(frozen=True, slots=True, repr=False)
class AdapterHandles:
    """Explicit private channels made available to Worker and Tool factories."""

    model_http: Any | None = field(default=None, repr=False)
    tool_http: Any | None = field(default=None, repr=False)
    tool_subprocess: Any | None = field(default=None, repr=False)
    caido_graphql: Any | None = field(default=None, repr=False)
    instrumentation: RuntimeInstrumentation | None = field(default=None, repr=False)

    def merge(self, other: AdapterHandles) -> AdapterHandles:
        values: dict[str, Any | None] = {}
        for name in (
            "model_http",
            "tool_http",
            "tool_subprocess",
            "caido_graphql",
            "instrumentation",
        ):
            current = getattr(self, name)
            incoming = getattr(other, name)
            if current is not None and incoming is not None:
                raise ValueError("Runtime adapters provide the same allocation channel")
            values[name] = current if current is not None else incoming
        return AdapterHandles(**values)

    def for_tool_channels(self, channels: set[str] | frozenset[str]) -> AdapterHandles:
        return AdapterHandles(
            tool_http=(self.tool_http if "runtime-http-client" in channels else None),
            tool_subprocess=(
                self.tool_subprocess if "runtime-subprocess-launcher" in channels else None
            ),
            caido_graphql=(self.caido_graphql if "caido-graphql-client" in channels else None),
        )

    def for_worker(self) -> AdapterHandles:
        return AdapterHandles(
            model_http=self.model_http,
            instrumentation=self.instrumentation,
        )

    @property
    def enabled_channels(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "model_http",
                "tool_http",
                "tool_subprocess",
                "caido_graphql",
                "instrumentation",
            )
            if getattr(self, name) is not None
        )

    def __repr__(self) -> str:
        return f"AdapterHandles(enabled_channels={self.enabled_channels!r})"


EMPTY_ADAPTER_HANDLES = AdapterHandles()


@dataclass(frozen=True, slots=True)
class RuntimeAdapterBuildContext:
    """Safe allocation identity; secret-bearing settings are passed separately."""

    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    request_timeout_seconds: int
    runtime_config_refs: tuple[str, ...]
    runtime_config_digests: tuple[str, ...]
    run_labels: tuple[str, ...]
    agent_labels: tuple[str, ...]
    run_metadata_labels: Mapping[str, str]
    runtime_adapter_refs: tuple[str, ...]
    private_bypass_hosts: tuple[str, ...]


class AdapterFactoryError(RuntimeError):
    """A classified, detail-free adapter construction failure."""

    def __init__(self, *, retryable: bool) -> None:
        super().__init__("Runtime adapter construction failed")
        self.retryable = retryable


class AdapterPreparationError(RuntimeError):
    """Safe host preparation failure consumed by the Allocation boundary."""

    def __init__(self, *, retryable: bool, cleanup_confirmed: bool) -> None:
        super().__init__("Runtime adapter preparation failed")
        self.retryable = retryable
        self.cleanup_confirmed = cleanup_confirmed


class AdapterTeardownError(RuntimeError):
    """At least one adapter close could not be confirmed within the deadline."""

    def __init__(self) -> None:
        super().__init__("Runtime adapter close could not be confirmed")


@dataclass(slots=True, repr=False)
class RuntimeAdapterMetricsState:
    """Saturating, safe adapter telemetry retained after the adapter is erased."""

    operations: int = 0
    failed_operations: int = 0
    flush_attempted: bool | None = None
    flush_succeeded: bool | None = None
    last_error_code: str | None = None
    dropped_spans: DroppedSpanCounts | None = None

    def record_dropped_spans(self, reason: str, count: int) -> None:
        if reason not in DroppedSpanCounts.model_fields or type(count) is not int or count < 0:
            raise ValueError("invalid discarded span count")
        if self.dropped_spans is None:
            self.dropped_spans = DroppedSpanCounts()
        value = getattr(self.dropped_spans, reason)
        setattr(self.dropped_spans, reason, min(UINT64_MAX, value + count))

    def record_operation(self, *, succeeded: bool, error_code: str | None = None) -> None:
        if error_code is not None and error_code not in SAFE_ERROR_CODES:
            raise ValueError("Runtime adapter error code is not allowlisted")
        self.operations = _saturating_increment(self.operations)
        if not succeeded:
            self.failed_operations = _saturating_increment(self.failed_operations)
            self.last_error_code = error_code

    def record_flush(self, *, succeeded: bool, error_code: str | None = None) -> None:
        self.flush_attempted = True
        self.flush_succeeded = succeeded
        self.record_operation(succeeded=succeeded, error_code=error_code)

    def snapshot(self) -> RuntimeAdapterMetricsV2:
        operations = max(0, min(UINT64_MAX, self.operations))
        failures = max(0, min(operations, self.failed_operations))
        flush_attempted = self.flush_attempted
        flush_succeeded = self.flush_succeeded
        if (flush_attempted is None) != (flush_succeeded is None):
            flush_attempted = None
            flush_succeeded = None
        elif flush_attempted is False and flush_succeeded:
            flush_succeeded = False
        error_code = self.last_error_code if self.last_error_code in SAFE_ERROR_CODES else None
        return RuntimeAdapterMetricsV2(
            operations=operations,
            failedOperations=failures,
            flushAttempted=flush_attempted,
            flushSucceeded=flush_succeeded,
            lastErrorCode=error_code,
            droppedSpans=(
                self.dropped_spans.model_copy(deep=True) if self.dropped_spans is not None else None
            ),
        )

    def __repr__(self) -> str:
        return (
            "RuntimeAdapterMetricsState("
            f"operations={self.operations}, failed_operations={self.failed_operations}, "
            f"flush_attempted={self.flush_attempted!r}, "
            f"flush_succeeded={self.flush_succeeded!r}, "
            f"last_error_code={self.last_error_code!r})"
        )


class AllocationAdapter(Protocol):
    ref: RuntimeAdapterRef
    handles: AdapterHandles
    metrics: RuntimeAdapterMetricsState

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


AdapterSettings = TelemetrySettingsV2 | HTTPProxySettingsV2 | CaidoSettingsV2


class RuntimeAdapterFactory(Protocol):
    ref: str

    async def probe(self) -> bool: ...

    async def create(
        self,
        context: RuntimeAdapterBuildContext,
        settings: AdapterSettings,
    ) -> AllocationAdapter: ...


@dataclass(slots=True, repr=False)
class _HostedAdapter:
    adapter: AllocationAdapter = field(repr=False)
    flush_timeout_seconds: float | None


class AllocationAdapterHost:
    """Constructs, injects, flushes and erases one allocation's adapters."""

    def __init__(
        self,
        hosted: Mapping[RuntimeAdapterRef, _HostedAdapter] | None = None,
        *,
        handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
    ) -> None:
        self._hosted = dict(hosted or {})
        self._handles = handles
        self._metrics: dict[RuntimeAdapterRef, RuntimeAdapterMetricsState] = {
            ref: item.adapter.metrics for ref, item in self._hosted.items()
        }
        self._closed = not self._hosted

    @classmethod
    async def create(
        cls,
        spec: AllocationSpecV2,
        factories: Mapping[str, RuntimeAdapterFactory],
        *,
        deadline: datetime,
        now: Callable[[], datetime],
        private_bypass_hosts: Sequence[str] = (),
    ) -> AllocationAdapterHost:
        selected = _selected_settings(spec.runtime_settings)
        hosted: dict[RuntimeAdapterRef, _HostedAdapter] = {}
        handles = EMPTY_ADAPTER_HANDLES
        empty_metadata_labels: Mapping[str, str] = MappingProxyType({})
        run_metadata_labels: Mapping[str, str] = MappingProxyType(dict(spec.run_metadata_labels))
        context = RuntimeAdapterBuildContext(
            allocation_id=spec.allocation_id,
            run_id=spec.run_id,
            stage_execution_id=spec.stage_execution_id,
            logical_agent_name=spec.logical_agent_name,
            request_timeout_seconds=spec.runtime_settings.request_timeout_seconds,
            runtime_config_refs=_runtime_config_refs(spec),
            runtime_config_digests=_runtime_config_digests(spec),
            run_labels=tuple(
                item.label for item in spec.resolved_runtime_config_provenance.run_labels
            ),
            agent_labels=tuple(
                item.label for item in spec.resolved_runtime_config_provenance.agent_labels
            ),
            run_metadata_labels=empty_metadata_labels,
            runtime_adapter_refs=tuple(spec.resolved_runtime_config_provenance.runtime_adapters),
            private_bypass_hosts=_private_bypass_hosts(
                spec.runtime_settings,
                private_bypass_hosts,
            ),
        )
        retryable = False
        construction_unconfirmed = False
        try:
            for ref in sorted(selected):
                settings = selected[ref]
                factory = factories.get(ref)
                if factory is None:
                    raise AdapterFactoryError(retryable=False)
                remaining = (deadline - now()).total_seconds()
                if remaining <= 0:
                    construction_unconfirmed = True
                    raise AdapterFactoryError(retryable=True)
                create_task = asyncio.create_task(
                    factory.create(
                        (
                            replace(
                                context,
                                run_metadata_labels=run_metadata_labels,
                            )
                            if isinstance(settings, TelemetrySettingsV2)
                            else context
                        ),
                        settings,
                    ),
                    name=f"adapter-create-{ref}-{spec.allocation_id}",
                )
                try:
                    done, _ = await asyncio.wait({create_task}, timeout=remaining)
                except asyncio.CancelledError:
                    create_task.cancel()
                    create_task.add_done_callback(_consume_background_task)
                    construction_unconfirmed = True
                    raise
                if not done:
                    create_task.cancel()
                    create_task.add_done_callback(_consume_background_task)
                    construction_unconfirmed = True
                    raise AdapterFactoryError(retryable=True)
                flush_timeout = (
                    float(settings.flush_timeout_seconds)
                    if isinstance(settings, TelemetrySettingsV2)
                    else None
                )
                try:
                    adapter = await create_task
                except asyncio.CancelledError:
                    if create_task.cancelled():
                        raise AdapterFactoryError(retryable=True) from None
                    raise
                hosted[ref] = _HostedAdapter(
                    adapter=adapter,
                    flush_timeout_seconds=flush_timeout,
                )
                if str(adapter.ref) != ref or not isinstance(
                    adapter.metrics, RuntimeAdapterMetricsState
                ):
                    raise AdapterFactoryError(retryable=False)
                _validate_typed_handles(settings, adapter.handles)
                handles = handles.merge(adapter.handles)
        except asyncio.CancelledError:
            cleanup_confirmed = await _close_hosted(hosted, deadline=deadline, now=now)
            if not cleanup_confirmed or construction_unconfirmed:
                raise AdapterPreparationError(retryable=True, cleanup_confirmed=False) from None
            raise
        except AdapterFactoryError as error:
            retryable = error.retryable
        except Exception:
            retryable = False

        if len(hosted) != len(selected):
            cleanup_confirmed = await _close_hosted(hosted, deadline=deadline, now=now)
            raise AdapterPreparationError(
                retryable=retryable,
                cleanup_confirmed=cleanup_confirmed and not construction_unconfirmed,
            ) from None
        return cls(hosted, handles=handles)

    @classmethod
    def empty(cls) -> AllocationAdapterHost:
        return cls()

    @property
    def handles(self) -> AdapterHandles:
        return self._handles

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(str(ref) for ref in self._metrics))

    @property
    def closed(self) -> bool:
        return self._closed

    def report_metrics(self) -> dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2]:
        return {ref: self._metrics[ref].snapshot() for ref in sorted(self._metrics)}

    async def terminate(
        self,
        *,
        deadline: datetime,
        now: Callable[[], datetime],
    ) -> None:
        if self._closed:
            return
        close_failed = False
        for ref in reversed(tuple(self._hosted)):
            hosted = self._hosted[ref]
            if (
                hosted.flush_timeout_seconds is not None
                and hosted.adapter.metrics.flush_attempted is None
            ):
                remaining = (deadline - now()).total_seconds()
                # Flush is best-effort. Reserve half of the remaining outer
                # deadline for the mandatory close that erases credentials.
                timeout = min(hosted.flush_timeout_seconds, max(0.0, remaining / 2))
                outcome = await _bounded_call(hosted.adapter.flush(), timeout=timeout)
                if outcome == "succeeded":
                    hosted.adapter.metrics.record_flush(succeeded=True)
                elif outcome == "timeout":
                    hosted.adapter.metrics.record_flush(succeeded=False, error_code="flush_timeout")
                elif outcome == "cancel_unconfirmed":
                    hosted.adapter.metrics.record_flush(succeeded=False, error_code="flush_timeout")
                    close_failed = True
                else:
                    hosted.adapter.metrics.record_flush(succeeded=False, error_code="flush_failed")

            remaining = (deadline - now()).total_seconds()
            outcome = await _bounded_call(hosted.adapter.close(), timeout=max(0.0, remaining))
            if outcome == "succeeded":
                del self._hosted[ref]
            else:
                hosted.adapter.metrics.record_operation(succeeded=False, error_code="close_failed")
                close_failed = True

        if close_failed:
            self._handles = EMPTY_ADAPTER_HANDLES
            raise AdapterTeardownError
        self._handles = EMPTY_ADAPTER_HANDLES
        self._closed = True

    async def rollback(
        self,
        *,
        deadline: datetime,
        now: Callable[[], datetime],
    ) -> None:
        """Close without flush because no Worker execution became active."""

        if self._closed:
            return
        if not await _close_hosted(self._hosted, deadline=deadline, now=now):
            self._handles = EMPTY_ADAPTER_HANDLES
            raise AdapterTeardownError
        self._handles = EMPTY_ADAPTER_HANDLES
        self._closed = True

    def __repr__(self) -> str:
        return f"AllocationAdapterHost(refs={self.refs!r}, closed={self._closed!r})"


def _selected_settings(settings: RuntimeSettingsV2) -> dict[str, AdapterSettings]:
    selected: dict[str, AdapterSettings] = {}
    if settings.telemetry is not None:
        selected[settings.telemetry.adapter] = settings.telemetry
    if settings.http_proxy is not None:
        selected[settings.http_proxy.adapter] = settings.http_proxy
    if settings.caido is not None:
        selected[settings.caido.adapter] = settings.caido
    return selected


def _runtime_config_refs(spec: AllocationSpecV2) -> tuple[str, ...]:
    provenance = spec.resolved_runtime_config_provenance
    bindings = (provenance.default, *provenance.run_labels, *provenance.agent_labels)
    return tuple(f"{binding.config.name}@{binding.config.version}" for binding in bindings)


def _runtime_config_digests(spec: AllocationSpecV2) -> tuple[str, ...]:
    provenance = spec.resolved_runtime_config_provenance
    bindings = (provenance.default, *provenance.run_labels, *provenance.agent_labels)
    return tuple(binding.config.digest for binding in bindings)


def _private_bypass_hosts(
    settings: RuntimeSettingsV2,
    additional: Sequence[str],
) -> tuple[str, ...]:
    parsed = urlsplit(settings.artifact_api_url)
    values = {"127.0.0.1", "::1", "localhost", *additional}
    if parsed.hostname is not None:
        values.add(parsed.hostname)
    if parsed.netloc:
        values.add(parsed.netloc)
    return tuple(sorted(values))


def _validate_typed_handles(settings: AdapterSettings, handles: AdapterHandles) -> None:
    if isinstance(settings, TelemetrySettingsV2):
        if (
            any(
                value is not None
                for value in (
                    handles.model_http,
                    handles.tool_http,
                    handles.tool_subprocess,
                    handles.caido_graphql,
                )
            )
            or handles.instrumentation is None
        ):
            raise AdapterFactoryError(retryable=False)
        return
    if isinstance(settings, CaidoSettingsV2):
        if (
            any(
                value is not None
                for value in (
                    handles.model_http,
                    handles.tool_http,
                    handles.tool_subprocess,
                    handles.instrumentation,
                )
            )
            or handles.caido_graphql is None
        ):
            raise AdapterFactoryError(retryable=False)
        return
    expected = {
        {
            "llm-gateway": "model_http",
            "tool-http": "tool_http",
            "tool-subprocess": "tool_subprocess",
        }[target]
        for target in settings.targets
    }
    actual = {
        name
        for name in ("model_http", "tool_http", "tool_subprocess")
        if getattr(handles, name) is not None
    }
    if (
        handles.instrumentation is not None
        or handles.caido_graphql is not None
        or actual != expected
    ):
        raise AdapterFactoryError(retryable=False)


async def _close_hosted(
    hosted: Mapping[RuntimeAdapterRef, _HostedAdapter],
    *,
    deadline: datetime,
    now: Callable[[], datetime],
) -> bool:
    confirmed = True
    for ref in reversed(tuple(hosted)):
        remaining = (deadline - now()).total_seconds()
        outcome = await _bounded_call(hosted[ref].adapter.close(), timeout=max(0.0, remaining))
        if outcome != "succeeded":
            confirmed = False
    if isinstance(hosted, dict) and confirmed:
        hosted.clear()
    return confirmed


async def _bounded_call(operation: Awaitable[None], *, timeout: float) -> str:
    if timeout <= 0:
        if hasattr(operation, "close"):
            operation.close()  # type: ignore[union-attr]
        return "timeout"
    task = asyncio.create_task(operation)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_consume_background_task)
        raise
    if not done:
        task.cancel()
        try:
            cancelled, _ = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.add_done_callback(_consume_background_task)
            raise
        if not cancelled:
            task.add_done_callback(_consume_background_task)
            return "cancel_unconfirmed"
        _consume_background_task(task)
        return "timeout"
    try:
        await task
    except asyncio.CancelledError:
        raise
    except Exception:
        return "failed"
    return "succeeded"


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


def _saturating_increment(value: int) -> int:
    return min(UINT64_MAX, value + 1)
