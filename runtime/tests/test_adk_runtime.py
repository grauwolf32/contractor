from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes.model import json_result, scripted_model, tool_call
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest

from contractor_runtime.adk_runtime import (
    AdkWorkerRuntime,
    AdkWorkerRuntimeFactory,
    GatewayLiteLlm,
    GatewayModelError,
)
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    ArtifactRef,
    ArtifactWriteResult,
    ModelPolicyRef,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    RuntimeSettings,
    SandboxProfileRef,
    StageContentRequest,
    ToolsetRef,
    ToolsetSelection,
    WorkerRuntimeRef,
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-adk-gateway-token"


def test_adk_worker_executes_selected_tools_and_validates_exact_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                    call_id="read-1",
                ),
                tool_call(
                    "write_artifact",
                    {
                        "namespace": "builder",
                        "name": "report",
                        "media_type": "application/json",
                        "data_base64": base64.b64encode(b"{}").decode(),
                        "expected_revision": None,
                    },
                    call_id="write-1",
                ),
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Report created",
                        "artifacts": {
                            "report": {
                                "namespace": "builder",
                                "name": "report",
                                "revision": "write-r1",
                            }
                        },
                    }
                ),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        assert result.artifacts["report"].revision == "write-r1"
        assert client.calls == ["read_artifact", "write_artifact"]
        assert state.metrics.counters == {
            "input_tokens": 21,
            "llm_calls": 3,
            "output_tokens": 9,
            "outcomes.succeeded": 1,
            "tool_calls": 2,
            "tool_calls.read_artifact": 1,
            "tool_calls.write_artifact": 1,
            "total_tokens": 30,
        }
        assert all(request["maxOutputTokens"] == 4096 for request in model.requests)
        assert all(request["temperature"] == 0.1 for request in model.requests)
        assert all(
            request["toolNames"] == ["read_artifact", "write_artifact"]
            for request in model.requests
        )
        session = await runtime._session_service.get_session(
            app_name=runtime._app_name,
            user_id=runtime._user_id,
            session_id=runtime._session_id,
        )
        assert session is not None
        assert session.state["metrics"]["finalOutcome"] == "succeeded"
        assert session.state["metrics"]["counters"]["tool_calls"] == 2
        serialized = repr(runtime) + repr(state.metrics) + result.model_dump_json(by_alias=True)
        assert SECRET not in serialized
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_rejects_unknown_fields_and_unobserved_artifact_refs(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        invalid = scripted_model(
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Looks valid",
                        "artifacts": {},
                        "invented": True,
                    }
                )
            ]
        )
        first = await create_runtime(tmp_path / "invalid", WorkerState(), {}, invalid)
        invalid_result = await first.invoke(stage_request())
        assert invalid_result.error is not None
        assert invalid_result.error.code == "invalid_worker_result"
        await first.abort(datetime.now(UTC) + timedelta(seconds=1))

        invented = scripted_model(
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Invented ref",
                        "artifacts": {
                            "report": {
                                "namespace": "builder",
                                "name": "report",
                                "revision": "guessed-r1",
                            }
                        },
                    }
                )
            ]
        )
        second = await create_runtime(tmp_path / "invented", WorkerState(), {}, invented)
        invented_result = await second.invoke(stage_request())
        assert invented_result.error is not None
        assert invented_result.error.code == "unverified_artifact_ref"
        await second.abort(datetime.now(UTC) + timedelta(seconds=1))

        secret_bearing = scripted_model(
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": SECRET,
                        "artifacts": {},
                    }
                )
            ]
        )
        secret_state = WorkerState()
        third = await create_runtime(tmp_path / "secret", secret_state, {}, secret_bearing)
        secret_result = await third.invoke(stage_request())
        assert secret_result.error is not None
        assert secret_result.error.code == "unsafe_worker_result"
        assert SECRET not in secret_result.model_dump_json(by_alias=True)
        assert SECRET not in repr(secret_state.metrics.snapshot())
        await third.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_abort_cancels_long_running_adk_invocation(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = scripted_model(
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Too late",
                        "artifacts": {},
                    }
                )
            ],
            block=True,
        )
        state = WorkerState()
        runtime = await create_runtime(tmp_path, state, {}, model)
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(model.started.wait(), timeout=1)

        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert state.metrics.final_outcome == "cancelled"
        assert state.metrics.counters["llm_calls"] == 1
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_gateway_adapter_discards_secret_bearing_provider_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def leaking_provider(_model: LiteLlm, _request: LlmRequest, stream: bool = False) -> Any:
        del stream
        if False:
            yield None
        raise RuntimeError(f"provider rejected Authorization: Bearer {SECRET}")

    monkeypatch.setattr(LiteLlm, "generate_content_async", leaking_provider)
    model = GatewayLiteLlm(
        model="openai/worker-model",
        api_base="https://llm.example/v1",
        api_key=SECRET,
    )

    async def scenario() -> None:
        with pytest.raises(GatewayModelError) as captured:
            async for _response in model.generate_content_async(LlmRequest()):
                pass
        assert SECRET not in repr(captured.value)
        assert captured.value.__context__ is None

    asyncio.run(scenario())
    model.clear_credentials()


async def create_runtime(
    tmp_path: Path,
    state: WorkerState,
    tools: dict[str, object],
    model: object,
) -> AdkWorkerRuntime:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = build_context(tmp_path, state, tools)
    factory = AdkWorkerRuntimeFactory(lambda _: model)  # type: ignore[arg-type,return-value]
    runtime = await factory.create(context)
    assert isinstance(runtime, AdkWorkerRuntime)
    return runtime


async def selected_tools(
    tmp_path: Path, client: FakeArtifactClient, state: WorkerState
) -> dict[str, object]:
    settings = runtime_settings()
    factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)  # type: ignore[arg-type]
    result = await factory.create_selected(
        selected=["read_artifact", "write_artifact"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="builder",
        runtime_settings=settings,
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        state=state,
    )
    return dict(result)


def build_context(
    tmp_path: Path, state: WorkerState, tools: dict[str, object]
) -> WorkerBuildContext:
    return WorkerBuildContext(
        allocation_id="allocation-1",
        run_id="run-1",
        stage_execution_id="stage-execution-1",
        logical_agent_name="builder",
        namespace="builder",
        agent_template=agent_template(),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        tools=tools,  # type: ignore[arg-type]
        state=state,
        a2a_base_url="https://runtime.example",
        runtime_settings=runtime_settings(),
    )


def agent_template() -> ResolvedAgentTemplate:
    return ResolvedAgentTemplate(
        ref=AgentTemplateRef(
            templateId="artifact_builder", version="1", digest="sha256:" + "0" * 64
        ),
        description="Build artifacts",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/worker.md", digest="sha256:" + "1" * 64, text="Use tools."
        ),
        modelPolicy=ResolvedModelPolicy(
            ref=ModelPolicyRef(policyId="worker", version="1", digest="sha256:" + "2" * 64),
            model="worker-model",
            maxOutputTokens=4096,
            temperature=0.1,
        ),
        toolsets=[
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="run-artifacts", version="1"),
                tools=["read_artifact", "write_artifact"],
            )
        ],
        sandboxProfile=SandboxProfileRef(sandboxProfileId="local-workdir", version="1"),
    )


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken=SECRET,
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )


def stage_request() -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        objective="Build a report",
        instructions="Read the input and write the report.",
        parameters={"format": "json"},
        artifacts={"source": ArtifactRef(namespace="inputs", name="source", revision="input-r1")},
    )


class FakeArtifactClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._known: dict[tuple[str, str, str], ArtifactRef] = {}

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.calls.append("read_artifact")
        exact = ArtifactRef(namespace=ref.namespace, name=ref.name, revision="read-r1")
        self._remember(exact)
        return ArtifactValue(artifact=exact, media_type="text/plain", data=b"source")

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        del expected_revision
        self.calls.append("write_artifact")
        exact = ArtifactRef(namespace=target.namespace, name=target.name, revision="write-r1")
        self._remember(exact)
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=exact,
            mediaType=media_type,
            size=len(data),
        )

    def _remember(self, ref: ArtifactRef) -> None:
        assert ref.revision is not None
        self._known[(ref.namespace, ref.name, ref.revision)] = ref
