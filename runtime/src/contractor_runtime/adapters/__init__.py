"""Allocation-scoped infrastructure adapter contracts and lifecycle host."""

from contractor_runtime.adapters.host import (
    AdapterFactoryError,
    AdapterHandles,
    AdapterPreparationError,
    AdapterTeardownError,
    AllocationAdapter,
    AllocationAdapterHost,
    RuntimeAdapterBuildContext,
    RuntimeAdapterFactory,
    RuntimeAdapterMetricsState,
)
from contractor_runtime.adapters.instrumentation import (
    RuntimeInstrumentation,
    RuntimeSpan,
    TelemetryAttribute,
)

__all__ = [
    "AdapterFactoryError",
    "AdapterHandles",
    "AdapterPreparationError",
    "AdapterTeardownError",
    "AllocationAdapter",
    "AllocationAdapterHost",
    "RuntimeAdapterBuildContext",
    "RuntimeAdapterFactory",
    "RuntimeAdapterMetricsState",
    "RuntimeInstrumentation",
    "RuntimeSpan",
    "TelemetryAttribute",
]
