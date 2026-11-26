"""Default LLM construction for a resolved Worker context."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from google.adk.models.base_llm import BaseLlm

from contractor_runtime.llm.client import build_gateway_client
from contractor_runtime.llm.openai import (
    OpenAICompatibleGatewayLlm,
)
from contractor_runtime.llm.recovery import GatewayRecoveryClient

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext


class ModelFactory(Protocol):
    def __call__(self, context: WorkerBuildContext) -> BaseLlm: ...


def gateway_model(context: WorkerBuildContext) -> BaseLlm:
    policy = context.model_policy
    client = build_gateway_client(context)
    client.cost_observer = context.cost_observer or context.state.metrics.llm_cost.observe
    return OpenAICompatibleGatewayLlm(
        model=policy.model,
        supports_images=policy.supports_images,
        client_handle=client,
    )


def with_gateway_recovery(context, artifact_client_factory):
    """Bind the existing allocation recovery transport before model construction."""
    if not context.runtime_settings.llm_recovery:
        return context
    if artifact_client_factory is None:
        raise RuntimeError("Model recovery requires the allocation control transport")
    client = artifact_client_factory(context.allocation_id, context.runtime_settings)
    return replace(
        context,
        gateway_recovery=GatewayRecoveryClient(
            client.allocation_id,
            client.transport,
            on_retry=context.state.metrics.record_recovery_authority_retry,
        ),
    )
