from importlib.metadata import version

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from contractor_runtime.a2a_server import build_agent_card, build_worker_a2a_application
from contractor_runtime.adk_runtime import GatewayLiteLlm
from contractor_runtime.contracts import StageContentRequest, StageContentResult


def test_pinned_adk_and_a2a_dependencies_construct_used_classes() -> None:
    assert version("google-adk") == "2.8.0"
    assert version("a2a-sdk") == "1.1.2"
    assert version("litellm") == "1.98.0"

    model = GatewayLiteLlm(
        model="openai/smoke-model",
        api_base="https://llm.example/v1",
        api_key="temporary-smoke-token",
    )
    assert isinstance(model, LiteLlm)
    agent = LlmAgent(
        name="smoke_worker",
        model=model,
        instruction="Return the requested result.",
        output_schema=StageContentResult,
    )
    runner = Runner(
        app_name="smoke",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    assert runner.agent is agent

    card = build_agent_card(
        allocation_id="allocation-smoke",
        endpoint="https://runtime.example/private/v1/allocations/allocation-smoke/a2a",
        logical_agent_name="smoke",
        description="Dependency smoke Worker",
        version="1",
    )
    application = build_worker_a2a_application(SmokeWorker(), card)
    assert isinstance(card, AgentCard)
    assert application is not None
    assert DefaultRequestHandler is not None
    model.clear_credentials()


class SmokeWorker:
    allocation_id = "allocation-smoke"

    async def invoke(self, request: StageContentRequest) -> StageContentResult:
        raise AssertionError(f"smoke Worker must not execute: {request!r}")

    def cancel_active(self) -> None:
        return None
