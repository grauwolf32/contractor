"""Gateway failure classification against the shared cross-language fixture."""

import asyncio
import json

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from paths import REPOSITORY_ROOT

from contractor_runtime.contracts import GatewayFailureSignatures
from contractor_runtime.contracts.settings import default_gateway_failure_signatures
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.errors import classify_response
from contractor_runtime.llm.openai import GatewayModelError, OpenAICompatibleGatewayLlm
from contractor_runtime.telemetry.metrics import MetricsState

FIXTURE = json.loads(
    (REPOSITORY_ROOT / "api/testdata/v1alpha1/gateway-failure-classification-cases.json").read_text(
        encoding="utf-8"
    )
)
SIGNATURE_SETS = {
    "declared": GatewayFailureSignatures.model_validate(FIXTURE["declared"]),
    "default": default_gateway_failure_signatures(),
    "empty": GatewayFailureSignatures(),
}


def _response(case: dict) -> httpx.Response:
    headers = case.get("headers", {})
    if "bodyText" in case:
        return httpx.Response(case["status"], text=case["bodyText"], headers=headers)
    return httpx.Response(case["status"], json=case["body"], headers=headers)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[case["name"] for case in FIXTURE["cases"]])
def test_classification_matches_shared_fixture(case: dict) -> None:
    failure = classify_response(_response(case), SIGNATURE_SETS[case["signatures"]])
    assert (failure.code, failure.retryable, failure.status) == (
        case["code"],
        case["retryable"],
        case["status"],
    )


def test_fixture_exercises_every_signature_set_and_both_verdicts() -> None:
    sets = {case["signatures"] for case in FIXTURE["cases"]}
    assert sets == set(SIGNATURE_SETS)
    verdicts = {(case["signatures"], case["retryable"]) for case in FIXTURE["cases"]}
    assert all((name, True) in verdicts and (name, False) in verdicts for name in SIGNATURE_SETS)


def test_classification_reaches_adapter_and_metrics_without_provider_body():
    async def scenario():
        async def gateway(request):
            return httpx.Response(404, json={"error": "model 'worker' not found"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            client = new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key="secret-canary",
                timeout_seconds=1,
                http_client=http,
                failure_signatures=SIGNATURE_SETS["declared"],
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
            assert "not found" not in repr(vars(failure))
            metrics = MetricsState()
            metrics.record_model_error(failure)
            assert "model_unavailable" in repr(metrics.snapshot())
            permanent = GatewayModelError("BadRequestError", retryable=False)
            metrics.record_model_error(permanent)
            assert metrics.errors[-1].retryable is False

            # The same response under the protocol default is a permanent rejection.
            plain = new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key="secret-canary",
                timeout_seconds=1,
                http_client=http,
            )
            plain.max_retries = 0
            model = OpenAICompatibleGatewayLlm(model="worker-model", client_handle=plain)
            with pytest.raises(GatewayModelError) as rejected:
                async for _ in model.generate_content_async(LlmRequest()):
                    pass
            assert not rejected.value.retryable
            assert rejected.value.failure.code == "gateway_request_rejected"

    asyncio.run(scenario())
