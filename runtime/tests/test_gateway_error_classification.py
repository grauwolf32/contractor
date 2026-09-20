"""Regression fixtures from LM Studio unload responses observed via LiteLLM."""

import asyncio

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest

from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.errors import classify_response
from contractor_runtime.llm.openai import GatewayModelError, OpenAICompatibleGatewayLlm
from contractor_runtime.telemetry.metrics import MetricsState


@pytest.mark.parametrize(
    "message", ["Model is unloaded.", "Model unloaded by user or API request."]
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_model_unload_is_transient_even_when_provider_uses_http_400(message, wrapped):
    body = {"error": message}
    if wrapped:
        body = {
            "error": {
                "message": (
                    "litellm.BadRequestError: OpenAIException - Error code: 400 - "
                    + repr(body)
                    + ". Received Model Group=worker-model"
                ),
                "code": "400",
            }
        }
    failure = classify_response(httpx.Response(400, json=body))
    assert (failure.code, failure.retryable, failure.status) == ("model_unavailable", True, 400)


@pytest.mark.parametrize(
    "status,body",
    [
        (400, {"error": "Invalid tools: Model is unloaded."}),
        (400, {"error": {"code": "context_length_exceeded"}}),
        (429, {"error": {"code": "insufficient_quota"}}),
        (429, {"error": {"code": "budget_exceeded"}}),
        (401, {"error": "Model is unloaded."}),
        (403, {"error": "Model is unloaded."}),
    ],
)
def test_semantic_failures_cannot_be_overridden_by_retry_hint(status, body):
    failure = classify_response(httpx.Response(status, json=body))
    assert not failure.retryable
    if isinstance(body["error"], dict) or status in {401, 403}:
        hinted = classify_response(
            httpx.Response(status, json=body, headers={"x-should-retry": "true"})
        )
        assert not hinted.retryable


def test_classification_reaches_adapter_and_metrics_without_provider_body():
    async def scenario():
        async def gateway(request):
            return httpx.Response(400, json={"error": "Model is unloaded."})

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            client = new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key="secret-canary",
                timeout_seconds=1,
                http_client=http,
            )
            client.max_retries = 0
            model = OpenAICompatibleGatewayLlm(model="worker-model", client_handle=client)
            with pytest.raises(GatewayModelError) as caught:
                async for _ in model.generate_content_async(LlmRequest()):
                    pass
            failure = caught.value
            assert failure.retryable
            assert failure.failure.code == "model_unavailable"
            assert failure.__context__ is None
            assert "secret-canary" not in repr(vars(failure))
            assert "Model is unloaded" not in repr(vars(failure))
            metrics = MetricsState()
            metrics.record_model_error(failure)
            assert "model_unavailable" in repr(metrics.snapshot())
            permanent = GatewayModelError("BadRequestError", retryable=False)
            metrics.record_model_error(permanent)
            assert metrics.errors[-1].retryable is False

    asyncio.run(scenario())
