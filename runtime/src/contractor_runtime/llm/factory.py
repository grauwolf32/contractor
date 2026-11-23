"""Default LLM construction for a resolved Worker context."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from google.adk.models.base_llm import BaseLlm

from contractor_runtime.llm.client import build_gateway_client
from contractor_runtime.llm.openai import (
    OpenAICompatibleGatewayLlm,
)

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext


class ModelFactory(Protocol):
    def __call__(self, context: WorkerBuildContext) -> BaseLlm: ...


def gateway_model(context: WorkerBuildContext) -> BaseLlm:
    policy = context.model_policy
    client = build_gateway_client(context)
    client.cost_observer = context.state.metrics.llm_cost.observe
    return OpenAICompatibleGatewayLlm(
        model=policy.model,
        supports_images=policy.supports_images,
        client_handle=client,
    )
