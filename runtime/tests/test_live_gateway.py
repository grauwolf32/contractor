"""Opt-in smoke test for a real OpenAI-compatible LLM Gateway."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from contractor_runtime.adk_runtime import AdkWorkerRuntime, AdkWorkerRuntimeFactory, GatewayLiteLlm
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import (
    API_VERSION,
    ModelPolicyRef,
    ResolvedModelPolicy,
    RuntimeSettings,
    StageContentRequest,
    WorkerModelResult,
    WorkerSessionMode,
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.metrics import MetricsState
from contractor_runtime.summarizer import TerminalSummarizer
from contractor_runtime.workspace import AllocationWorkspace

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


@pytest.mark.skipif(
    not GATEWAY_URL or not GATEWAY_TOKEN,
    reason="live LLM Gateway environment is not configured",
)
def test_live_terminal_summarizer_returns_one_strict_worker_result() -> None:
    async def scenario() -> None:
        assert GATEWAY_URL is not None
        assert GATEWAY_TOKEN is not None
        model = GatewayLiteLlm(
            model=f"openai/{GATEWAY_MODEL}",
            api_base=GATEWAY_URL,
            api_key=GATEWAY_TOKEN,
            timeout=180.0,
            num_retries=0,
        )
        policy = ResolvedModelPolicy(
            ref=ModelPolicyRef(
                policyId="live_terminal_summarizer",
                version="1",
                digest="sha256:" + "1" * 64,
            ),
            model=GATEWAY_MODEL,
            contextWindowTokens=131_072,
            maxOutputTokens=2048,
            maxModelCalls=1,
            temperature=0,
        )
        summarizer = TerminalSummarizer(model=model, policy=policy)
        prompt = (
            "Contractor terminal Worker summary input (JSON):\n"
            '{"observations":{"profile":"lean@1","tools":{},"truncated":false},'
            '"task":{"artifacts":{},"instructions":"Return a concise factual result.",'
            '"objective":"Confirm the live terminal summarizer contract",'
            '"parameters":{},"subtaskId":"live-summary-1"},'
            '"transcript":[],"transcriptTruncated":false}'
        )

        candidate = await summarizer.run(prompt=prompt, invocation_id="live-summary")
        result = WorkerModelResult.model_validate_json(candidate)

        assert result.subtask_id == "live-summary-1"
        assert result.result.strip()
        assert summarizer.usage.model_calls == 1
        assert summarizer.usage.token_usage_unavailable in {0, 1}

    asyncio.run(scenario())


@pytest.mark.skipif(
    not GATEWAY_URL or not GATEWAY_TOKEN,
    reason="live LLM Gateway environment is not configured",
)
def test_live_worker_result_finalizer_separates_tools_and_schema(tmp_path: Path) -> None:
    async def inspect_marker(marker: str) -> dict[str, str]:
        """Return a harmless marker when inspection is useful."""

        return {"marker": marker}

    async def scenario() -> None:
        assert GATEWAY_URL is not None
        assert GATEWAY_TOKEN is not None
        workspace_path = tmp_path / "allocation-live-result-finalizer"
        workspace_path.mkdir()
        policy = ResolvedModelPolicy(
            ref=ModelPolicyRef(
                policyId="live_worker_result_finalizer",
                version="1",
                digest="sha256:" + "2" * 64,
            ),
            model=GATEWAY_MODEL,
            maxOutputTokens=2048,
            maxModelCalls=4,
            maxToolCalls=2,
            maxTotalTokens=32_768,
            temperature=0,
        )
        context = WorkerBuildContext(
            allocation_id="live-result-finalizer",
            run_id="live-run",
            stage_execution_id="live-stage-execution",
            logical_agent_name="live-worker",
            namespace="live-worker",
            worker_session_mode=WorkerSessionMode.ISOLATED,
            description="Live result-finalizer compatibility probe",
            instruction=(
                "Follow the task exactly. The available tool is optional. "
                "Return a concise terminal answer."
            ),
            card_version="1",
            model_policy=policy,
            workspace=AllocationWorkspace(root=tmp_path, path=workspace_path),
            tools={"inspect_marker": inspect_marker},  # type: ignore[dict-item]
            state=WorkerState(),
            a2a_base_url="https://runtime.invalid",
            runtime_settings=RuntimeSettings(
                llmGatewayUrl=GATEWAY_URL,
                llmGatewayToken=GATEWAY_TOKEN,
                artifactApiUrl="https://control.invalid/private/v1",
                requestTimeoutSeconds=180,
            ),
        )
        runtime = await AdkWorkerRuntimeFactory().create(context)
        assert isinstance(runtime, AdkWorkerRuntime)
        assert runtime._agent is not None
        assert runtime._agent.output_schema is None
        assert runtime._agent.tools
        assert runtime._result_finalizer is not None
        try:
            completion = await runtime.invoke(
                StageContentRequest(
                    apiVersion=API_VERSION,
                    subtaskId="live-finalizer-1",
                    objective="Confirm the separated result-finalizer path",
                    instructions=(
                        "Return the phrase LIVE_RESULT_FINALIZER_OK in the terminal answer."
                    ),
                    parameters={},
                    artifacts={},
                    resultArtifacts={},
                )
            )
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=5))

        assert completion.failure is None
        assert completion.result is not None
        assert completion.result.subtask_id == "live-finalizer-1"
        assert "LIVE_RESULT_FINALIZER_OK" in completion.result.result
        assert completion.result.summarized is False
        assert completion.result.observations.tools.get("inspect_marker") is None or (
            completion.result.observations.tools["inspect_marker"].calls == 1
        )

    asyncio.run(scenario())
