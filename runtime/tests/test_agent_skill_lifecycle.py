from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.model import scripted_model
from fakes.spec import allocation_spec
from test_agent_skill_toolset import FakeSkillClient, resolved_value, skill_package

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    TerminationError,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.state import ProcessState, RuntimeState


@pytest.mark.parametrize("termination", ["finalize", "abort", "lease"])
def test_skill_state_is_erased_before_slot_can_return_idle(
    tmp_path: Path, termination: str
) -> None:
    async def scenario() -> None:
        service, state, spec = await skilled_service(tmp_path)
        await service.prepare(spec)
        snapshot = await service.snapshot()
        assert snapshot is not None
        workspace = Path(snapshot.workspace)
        extraction = workspace / ".agent-skills"
        assert extraction.is_dir()

        if termination == "finalize":
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalization-1",
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
        elif termination == "abort":
            await service.abort(
                AbortAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    abortId="abort-1",
                    reason=TerminationError(
                        code="test_abort", message="Abort test allocation", retryable=False
                    ),
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
        else:
            await service.expire_control_lease(2)

        assert not extraction.exists()
        service_snapshot = await service.snapshot()
        assert service_snapshot is not None
        assert not service_snapshot.has_worker
        assert (await state.snapshot()).process_state in {
            ProcessState.DRAINING,
            ProcessState.FENCED,
        }

        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert not workspace.exists()

    asyncio.run(scenario())


def test_skill_preparation_failure_maps_exact_code_and_rolls_back(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        selected = selected.model_copy(update={"package_digest": "sha256:" + "0" * 64})
        client = FakeSkillClient({"demo": value})
        factories = built_in_factories(
            tmp_path,
            lambda _allocation, _settings: client,  # type: ignore[arg-type]
            lambda _: scripted_model([]),
        )
        capabilities = capabilities_for(factories)
        state = RuntimeState(instance_id="runtime-skills", capabilities=capabilities)
        await state.mark_registered()
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
            force_exit=lambda _: None,
        )
        spec = allocation_spec(skills=[selected])

        with pytest.raises(AllocationError) as captured:
            await service.prepare(spec)
        assert captured.value.code == "skill_digest_mismatch"
        assert captured.value.retryable is False
        assert await service.snapshot() is None
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_skill_deletion_failure_fences_and_forces_process_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        exit_codes: list[int] = []
        service, state, spec = await skilled_service(tmp_path, exit_codes=exit_codes)
        await service.prepare(spec)
        context = service._context
        assert context is not None
        runtime = context.worker
        owner = runtime._agent_skills  # type: ignore[attr-defined]
        assert owner is not None
        extraction = owner.extraction_root

        def fail_remove(_: Path) -> None:
            raise OSError("synthetic private deletion failure")

        owner._remove_tree = fail_remove
        with pytest.raises(AllocationError) as captured:
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalization-1",
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
        assert captured.value.code == "worker_stop_unconfirmed"
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert exit_codes == [70]
        assert extraction.exists()
        assert "synthetic private deletion failure" not in captured.value.message

    asyncio.run(scenario())


async def skilled_service(
    tmp_path: Path, *, exit_codes: list[int] | None = None
) -> tuple[AllocationService, RuntimeState, object]:
    payload = skill_package()
    selected, value = resolved_value(payload)
    assert selected.package_digest == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    client = FakeSkillClient({"demo": value})
    factories = built_in_factories(
        tmp_path,
        lambda _allocation, _settings: client,  # type: ignore[arg-type]
        lambda _: scripted_model([]),
    )
    capabilities = capabilities_for(factories)
    state = RuntimeState(instance_id="runtime-skills", capabilities=capabilities)
    await state.mark_registered()
    service = AllocationService(
        state,
        factories,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=(exit_codes.append if exit_codes is not None else lambda _: None),
    )
    return service, state, allocation_spec(skills=[selected])


def capabilities_for(factories: object) -> CapabilitySnapshot:
    return CapabilitySnapshot.create(
        runtimes=factories.worker_runtimes,  # type: ignore[attr-defined]
        toolsets={
            ref: factory.exported_tools
            for ref, factory in factories.toolsets.items()  # type: ignore[attr-defined]
        },
        sandbox_profiles=factories.sandbox_profiles,  # type: ignore[attr-defined]
        runtime_adapters=factories.runtime_adapters,  # type: ignore[attr-defined]
    )
