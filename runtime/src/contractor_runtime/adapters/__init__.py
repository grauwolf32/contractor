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
]
