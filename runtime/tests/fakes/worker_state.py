"""Allocation fixture pairing revisioned State with its private resources."""

from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.worker.state import WorkerStateStore as RuntimeWorkerStateStore


class WorkerStateStore(RuntimeWorkerStateStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resources = ToolResources(metrics=self.metrics)


WorkerState = WorkerStateStore
