"""Test-only Worker runtime with the allocation lifecycle surface."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from contractor_runtime.contracts import AgentStateSnapshot
from contractor_runtime.factories import WorkerBuildContext


class StubADKWorkerRuntimeFactory:
    ref = "adk@1"
    supports_agent_skills = False

    async def probe(self) -> bool:
        return True

    async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
        return StubWorkerRuntime(context)


class StubWorkerRuntime:
    def __init__(self, context: WorkerBuildContext) -> None:
        endpoint = (
            f"{context.a2a_base_url.rstrip('/')}/private/v1/allocations/{context.allocation_id}/a2a"
        )
        self._agent_card: dict[str, Any] = {
            "name": context.logical_agent_name,
            "description": context.description,
            "url": endpoint,
            "protocolVersion": "1.0",
            "version": context.card_version,
            "capabilities": {},
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json"],
            "skills": [],
        }
        self._worker_state = context.state
        self.stopped = False

    @property
    def agent_card(self) -> Mapping[str, Any]:
        return dict(self._agent_card)

    async def agent_state_snapshot(self) -> AgentStateSnapshot:
        return await self._worker_state.agent_state_snapshot()

    async def finalize(self, deadline: datetime) -> None:
        self.stopped = True

    async def abort(self, deadline: datetime) -> None:
        self.stopped = True
