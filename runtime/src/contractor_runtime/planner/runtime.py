"""Bind the transport-neutral Planner loop to an allocation's existing A2A lifecycle."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, datetime

from contractor_runtime.a2a_server import (
    agent_card_dict,
    build_agent_card,
    build_worker_a2a_application,
)
from contractor_runtime.contracts import (
    API_VERSION,
    WorkerCompletion,
    WorkerFailure,
    WorkerObservations,
    WorkerResult,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.digests import verify_model_policy_digest
from contractor_runtime.llm.cost import CostCollector
from contractor_runtime.llm.factory import gateway_model, with_gateway_recovery
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.planner.host import PlannerStoppedError
from contractor_runtime.planner.loop import PlannerLoop
from contractor_runtime.planner.server_host import ServerPlannerHost


class PlannerRuntimeFactory:
    ref = "planner@1"
    supports_agent_skills = False
    supports_worker_completion = False

    def __init__(self, artifact_client_factory, model_factory=None) -> None:
        self._clients = artifact_client_factory
        self._models = model_factory or gateway_model

    async def probe(self) -> bool:
        return self._clients is not None

    async def create(self, context):
        cost = CostCollector()
        context = with_gateway_recovery(replace(context, cost_observer=cost.observe), self._clients)
        client = self._clients(context.allocation_id, context.runtime_settings)
        return PlannerRuntime(
            context,
            self._models(context),
            client.transport,
            self._models,
            cost,
        )


class PlannerRuntime:
    def __init__(self, context, model, transport, model_factory, cost) -> None:
        self.allocation_id = context.allocation_id
        self._context, self._model, self._transport = context, model, transport
        self._model_factory, self._cost = model_factory, cost
        self._active = None
        self._accepting = True
        self._started = False
        self._invocation_id = uuid.uuid4().hex
        card = build_agent_card(
            allocation_id=self.allocation_id,
            endpoint=f"{context.a2a_base_url}/private/v1/allocations/{self.allocation_id}/a2a",
            logical_agent_name=context.logical_agent_name,
            description=context.description,
            version=context.card_version,
        )
        self.agent_card = agent_card_dict(card)
        self.a2a_application = build_worker_a2a_application(self, card)

    async def agent_state_snapshot(self):
        return await self._context.state.agent_state_snapshot()

    async def failure_completion(self, code, message, *, retryable=False):
        return WorkerCompletion(
            apiVersion=API_VERSION,
            invocationId=self._invocation_id,
            stateRevision=1,
            failure=WorkerFailure(code=code, message=message, retryable=retryable),
        )

    async def invoke(self, request):
        if not self._accepting or self._started or request.planner is None:
            return await self.failure_completion(
                "invalid_worker_request", "Planner invocation is unavailable"
            )
        if request.planner.compaction is not None:
            try:
                verify_model_policy_digest(request.planner.compaction.policy)
            except ValueError:
                return await self.failure_completion(
                    "invalid_stage_content", "Planner compactor policy digest is invalid"
                )
        self._started = True
        self._active = asyncio.current_task()
        compactor = None
        try:
            remaining = max(0, (request.deadline - datetime.now(UTC)).total_seconds())
            async with asyncio.timeout(remaining):
                host = ServerPlannerHost(
                    self.allocation_id, self._transport, deadline=request.deadline
                )
                if request.planner.compaction is not None:
                    compactor = self._model_factory(
                        replace(
                            self._context,
                            model_policy=request.planner.compaction.policy,
                            gateway_recovery=None,
                        )
                    )
                loop = PlannerLoop(
                    self._model,
                    host,
                    self._context.model_policy,
                    compactor=compactor,
                    cost_snapshot=lambda: self._cost.snapshot(0),
                )
                await loop.run(request.planner, request.instructions)
            return WorkerCompletion(
                apiVersion=API_VERSION,
                invocationId=self._invocation_id,
                stateRevision=1,
                result=WorkerResult(
                    subtaskId=request.subtask_id,
                    result="Planner invocation finished",
                    observations=WorkerObservations(profile="lean@1", tools={}, truncated=False),
                    artifacts={},
                    summarized=False,
                ),
            )
        except PlannerStoppedError as error:
            return await self.failure_completion(error.code, str(error), retryable=error.retryable)
        except TimeoutError:
            return await self.failure_completion(
                "planner_deadline_exceeded", "Planner execution deadline expired", retryable=True
            )
        finally:
            try:
                if isinstance(compactor, OpenAICompatibleGatewayLlm):
                    await compactor.close()
            finally:
                self._active = None

    def cancel_active(self, owner) -> None:
        if self._active is owner:
            owner.cancel()

    async def finalize(self, deadline) -> None:
        self._accepting = False
        active = self._active
        if active is not None:
            active.cancel()
            async with asyncio.timeout(max(0, (deadline - datetime.now(UTC)).total_seconds())):
                await join_owned_task(active)
        if isinstance(self._model, OpenAICompatibleGatewayLlm):
            await self._model.close()

    async def abort(self, deadline) -> None:
        await self.finalize(deadline)
