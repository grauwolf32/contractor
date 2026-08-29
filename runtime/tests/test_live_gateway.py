"""Opt-in smoke test for a real OpenAI-compatible LLM Gateway."""

from __future__ import annotations

import asyncio
import os

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from contractor_runtime.adk_runtime import GatewayLiteLlm
from contractor_runtime.metrics import MetricsState

GATEWAY_URL = os.getenv("CONTRACTOR_LIVE_LLM_GATEWAY_URL")
GATEWAY_TOKEN = os.getenv("CONTRACTOR_LIVE_LLM_GATEWAY_TOKEN")
GATEWAY_MODEL = os.getenv("CONTRACTOR_LIVE_LLM_MODEL", "worker-model")


@pytest.mark.skipif(
    not GATEWAY_URL or not GATEWAY_TOKEN,
    reason="live LLM Gateway environment is not configured",
)
def test_live_gateway_returns_content_and_usage_for_worker_metrics() -> None:
    async def scenario() -> None:
        assert GATEWAY_URL is not None
        assert GATEWAY_TOKEN is not None
        model = GatewayLiteLlm(
            model=f"openai/{GATEWAY_MODEL}",
            api_base=GATEWAY_URL,
            api_key=GATEWAY_TOKEN,
            timeout=180.0,
        )
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text='Reply with exactly {"ok":true}.')],
                )
            ],
            config=types.GenerateContentConfig(max_output_tokens=256, temperature=0),
        )
        metrics = MetricsState()
        metrics.record_model_call()
        texts: list[str] = []
        try:
            async for response in model.generate_content_async(request):
                if response.usage_metadata is not None:
                    metrics.record_model_usage(response.usage_metadata)
                if response.content is not None:
                    texts.extend(
                        part.text for part in response.content.parts or [] if part.text is not None
                    )
        finally:
            model.clear_credentials()

        assert any('"ok":true' in text.replace(" ", "") for text in texts)
        assert metrics.counters["llm_calls"] == 1
        assert metrics.counters.get("total_tokens", 0) > 0
        report = metrics.build_report(report_id="live-gateway-smoke", duration_ms=1)
        assert report.metrics.total_tokens is not None
        assert GATEWAY_TOKEN not in report.model_dump_json(by_alias=True)

    asyncio.run(scenario())
