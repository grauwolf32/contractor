import asyncio
import importlib.util
import json
import sys
from importlib.metadata import version

import httpx
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contractor_runtime.a2a_server import build_agent_card, build_worker_a2a_application
from contractor_runtime.contracts import StageContentRequest, WorkerCompletion, WorkerModelResult
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm


def test_pinned_adk_and_a2a_dependencies_construct_used_classes() -> None:
    assert version("google-adk") == "2.8.0"
    assert version("a2a-sdk") == "1.1.2"
    assert importlib.util.find_spec("openai") is None
    assert importlib.util.find_spec("litellm") is None

    model = OpenAICompatibleGatewayLlm(
        model="smoke-model",
        client_handle=new_gateway_client(
            base_url="https://llm.example/v1",
            api_key="temporary-smoke-token",
            timeout_seconds=120,
        ),
    )
    assert isinstance(model, BaseLlm)
    agent = LlmAgent(
        name="smoke_worker",
        model=model,
        instruction="Return the requested result.",
        output_schema=WorkerModelResult,
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
    asyncio.run(model.close())


def test_adk_gateway_invocation_does_not_load_optional_provider_sdks() -> None:
    async def scenario() -> None:
        async def gateway(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["model"] == "smoke-model"
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(
                200,
                json={
                    "model": "smoke-model",
                    "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            model = OpenAICompatibleGatewayLlm(
                model="smoke-model",
                client_handle=new_gateway_client(
                    base_url="https://gateway.example/v1",
                    api_key="smoke-token",
                    timeout_seconds=1,
                    http_client=http,
                ),
            )
            runner = Runner(
                app_name="dependency-smoke",
                agent=LlmAgent(name="worker", model=model, instruction="Return ok"),
                session_service=InMemorySessionService(),
                auto_create_session=True,
            )
            try:
                events = [
                    event
                    async for event in runner.run_async(
                        user_id="smoke-user",
                        session_id="smoke-session",
                        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
                    )
                ]
                final = next(event for event in events if event.is_final_response())
                assert final.content.parts[0].text == "ok"
                assert final.usage_metadata.total_token_count == 2
            finally:
                await model.close()
            for prefix in ("openai", "litellm", "anthropic"):
                assert not any(
                    name == prefix or name.startswith(prefix + ".") for name in sys.modules
                )

    asyncio.run(scenario())


class SmokeWorker:
    allocation_id = "allocation-smoke"

    async def invoke(self, request: StageContentRequest) -> WorkerCompletion:
        raise AssertionError(f"smoke Worker must not execute: {request!r}")

    async def failure_completion(
        self, code: str, message: str, *, retryable: bool = False
    ) -> WorkerCompletion:
        raise AssertionError(f"smoke Worker must not reject: {code}, {message}, {retryable}")

    def cancel_active(self, owner: asyncio.Task[object]) -> None:
        del owner
