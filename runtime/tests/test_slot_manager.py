from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes.spec import allocation_spec
from test_agent_skill_toolset import resolved_value, skill_package

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.factories import (
    FactoryRegistry,
    StubADKWorkerRuntimeFactory,
    built_in_factories,
)
from contractor_runtime.state import ProcessState, RuntimeState


def test_non_skill_runtime_rejects_selected_skills_before_workspace(tmp_path: Path) -> None:
    async def scenario() -> None:
        selected, _ = resolved_value(skill_package())
        builtins = built_in_factories(tmp_path)
        factories = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets=builtins.toolsets,
            sandbox_profiles=builtins.sandbox_profiles,
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=["adk@1"],
            toolsets={ref: item.exported_tools for ref, item in factories.toolsets.items()},
            sandbox_profiles=factories.sandbox_profiles,
        )
        state = RuntimeState(instance_id="runtime-no-skills", capabilities=capabilities)
        await state.mark_registered()
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
            force_exit=lambda _: None,
        )

        with pytest.raises(AllocationError) as captured:
            await service.prepare(allocation_spec(skills=[selected]))
        assert captured.value.code == "skill_runtime_unsupported"
        assert captured.value.retryable is False
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert await service.snapshot() is None
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())
