"""Inspect the one allocation owned by a single-allocation test fixture."""

from dataclasses import dataclass

from contractor_runtime.allocation.service import AllocationService
from contractor_runtime.state import ProcessState


def allocation_owner(service: AllocationService):
    assert len(service._allocations) == 1
    return next(iter(service._allocations.values()))


def allocation_context(service: AllocationService):
    return allocation_owner(service)._context if service._allocations else None


@dataclass(frozen=True, slots=True)
class AllocationSnapshot:
    allocation_id: str
    stage_execution_id: str
    process_state: ProcessState
    workspace: str
    tool_names: tuple[str, ...]
    has_runtime_settings: bool
    has_worker: bool
    runtime_adapter_refs: tuple[str, ...]
    has_project_workspace: bool


async def allocation_snapshot(service: AllocationService):
    context = allocation_context(service)
    if context is None:
        return None
    owner = allocation_owner(service)
    async with owner._lock:
        state = await owner._state.snapshot(context.allocation_id)
        return AllocationSnapshot(
            allocation_id=context.allocation_id,
            stage_execution_id=context.stage_execution_id,
            process_state=state.process_state,
            workspace=str(context.workspace.path),
            tool_names=tuple(sorted(context.tools)),
            has_runtime_settings=context.runtime_settings is not None,
            has_worker=context.worker is not None,
            runtime_adapter_refs=context.adapter_host.refs,
            has_project_workspace=context.project_workspace is not None,
        )
