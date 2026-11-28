from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fakes.model import scripted_model, text_result, tool_call
from fakes.worker_state import WorkerState
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.skill_toolset import SkillDiscoveryMode

import contractor_runtime.agent_skills.runtime as skill_runtime
from contractor_runtime.agent_skills import MEDIA_TYPE
from contractor_runtime.agent_skills.runtime import (
    BINARY_ASSET_STATUS,
    DISCLOSURE_ENVELOPE_BYTES,
    EXACT_SKILL_TOOL_NAMES,
    MAXIMUM_DISCLOSURE_BYTES,
    MAXIMUM_PACKAGES,
    AgentSkillPreparationError,
    DisclosureBudget,
    prepare_agent_skills,
    probe_native_agent_skills,
)
from contractor_runtime.artifacts import ArtifactTransportError, ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    ArtifactRef,
    ModelPolicyRef,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    ResolvedSkill,
    RuntimeSettings,
    SandboxProfileRef,
    StageContentRequest,
    WorkerRuntimeRef,
    WorkerSessionMode,
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory
from contractor_runtime.worker.runtime import AdkWorkerRuntime
from contractor_runtime.workspace import AllocationWorkspace


def test_worker_uses_exact_native_script_free_skill_surface(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        client = FakeSkillClient({"demo": value})
        model = scripted_model(
            [
                tool_call("list_skills", {}, call_id="list"),
                tool_call("load_skill", {"skill_name": "demo"}, call_id="load"),
                tool_call(
                    "load_skill_resource",
                    {"skill_name": "demo", "file_path": "references/guide.md"},
                    call_id="resource",
                ),
                text_result("Used the selected skill"),
            ]
        )
        state = WorkerState()
        context = build_context(tmp_path, state, selected)
        factory = AdkWorkerRuntimeFactory(lambda _: model, lambda _allocation, _settings: client)  # type: ignore[arg-type,return-value]

        runtime = await factory.create(context)
        assert isinstance(runtime, AdkWorkerRuntime)
        extraction_root = tmp_path / ".agent-skills"
        assert (extraction_root / "demo" / "SKILL.md").is_file()
        assert client.reads == [selected.artifact]
        assert "Follow the exact demo procedure" not in repr(runtime._agent_skills)
        assert str(tmp_path) not in repr(runtime._agent_skills)

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert result.result.result == "Used the selected skill"
        assert len(model.requests) == 5
        for request in model.requests[:-1]:
            assert request["toolNames"] == sorted(EXACT_SKILL_TOOL_NAMES)
            instruction = request["systemInstruction"]
            assert isinstance(instruction, str)
            assert "load_skill" in instruction
            assert all(
                forbidden not in instruction
                for forbidden in (
                    "run_skill_script",
                    "search_skills",
                    "AgentTemplate",
                    "agent template",
                )
            )
        assert model.requests[-1]["toolNames"] == []
        assert model.requests[-1]["hasResponseSchema"] is True
        assert state.metrics.counters["tool_calls"] == 3
        assert [call.tool for call in state.metrics.tool_calls] == list(EXACT_SKILL_TOOL_NAMES)
        assert all(
            call.result_size_bytes and call.result_size_bytes > 0
            for call in state.metrics.tool_calls
        )
        budget = state.metrics.build_report(
            report_id="worker-skill-report", duration_ms=1
        ).metrics.worker_budget
        assert budget is not None
        assert budget.observed_tool_calls == 3
        serialized_metrics = json.dumps(state.metrics.snapshot(), sort_keys=True)
        assert "Follow the exact demo procedure" not in serialized_metrics
        assert "Reference contents" not in serialized_metrics

        session_id = runtime._session_lifecycle.shared_session_id
        assert session_id is not None
        session = await runtime._session_service.get_session(
            app_name=runtime._app_name,
            user_id=runtime._user_id,
            session_id=session_id,
        )
        assert session is not None
        assert session.state["_adk_activated_skill_contractor_worker"] == ["demo"]
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))
        assert not extraction_root.exists()
        assert runtime._agent_skills is None
        assert runtime._context is None

    asyncio.run(scenario())


def test_script_source_disclosure_is_exact_bounded_and_never_executes(tmp_path: Path) -> None:
    async def scenario() -> None:
        marker = tmp_path / "executed"
        source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
        selected, value = resolved_value(skill_package(script=source))
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="allocation-1",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda _allocation, _settings: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}
        assert tuple(tools) == EXACT_SKILL_TOOL_NAMES
        assert prepared.native_toolset._code_executor is None
        assert prepared.native_toolset._env is None
        context = FakeToolContext()
        loaded = await tools["load_skill"].run_async(
            args={"skill_name": "demo"}, tool_context=context
        )
        assert loaded["skill_name"] == "demo"
        charge = prepared.charge_for("load_skill_resource", "demo", "scripts/check.py")
        prepared.disclosure = DisclosureBudget(limit=charge)
        result = await tools["load_skill_resource"].run_async(
            args={"skill_name": "demo", "file_path": "scripts/check.py"},
            tool_context=context,
        )
        assert result == {
            "skill_name": "demo",
            "file_path": "scripts/check.py",
            "content": source.decode(),
        }
        assert prepared.disclosure.used == charge
        blocked = await tools["load_skill_resource"].run_async(
            args={"skill_name": "demo", "file_path": "scripts/check.py"},
            tool_context=context,
        )
        assert blocked["error_code"] == "SKILL_DISCLOSURE_LIMIT"
        assert not marker.exists()
        assert source.decode() not in json.dumps(state.metrics.snapshot())
        assert (prepared.extraction_root / "demo/scripts/check.py").read_bytes() == source
        native_skill = prepared.skills[0]
        await prepared.close()
        assert not native_skill.resources.scripts
        assert not prepared.extraction_root.exists()
        assert not marker.exists()

    asyncio.run(scenario())


def test_worker_loads_binary_skill_asset_through_text_gateway(tmp_path: Path) -> None:
    async def scenario() -> None:
        selected, value = resolved_value(skill_package())
        requests: list[dict[str, Any]] = []
        replies = [
            {
                "tool_calls": [
                    {
                        "id": "load",
                        "type": "function",
                        "function": {
                            "name": "load_skill",
                            "arguments": '{"skill_name":"demo"}',
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "resource",
                        "type": "function",
                        "function": {
                            "name": "load_skill_resource",
                            "arguments": '{"skill_name":"demo","file_path":"assets/pixel.bin"}',
                        },
                    }
                ]
            },
            {"content": "Used the selected skill"},
            {"content": '{"subtaskId":"0","result":"Used the selected skill"}'},
        ]

        async def gateway(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            requests.append(json.loads(request.content))
            index = len(requests) - 1
            assert index < len(replies)
            return httpx.Response(
                200,
                json={
                    "model": "worker-model",
                    "choices": [
                        {
                            "finish_reason": "tool_calls" if index < 2 else "stop",
                            "message": replies[index],
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(gateway), trust_env=False
        ) as http:
            model = OpenAICompatibleGatewayLlm(
                model="worker-model",
                client_handle=new_gateway_client(
                    base_url="https://gateway.example/v1",
                    api_key="test-token",
                    timeout_seconds=1,
                    http_client=http,
                ),
            )
            state = WorkerState()
            runtime = await AdkWorkerRuntimeFactory(
                lambda _: model,
                lambda _allocation, _settings: FakeSkillClient({"demo": value}),
            ).create(build_context(tmp_path, state, selected))
            completion = await runtime.invoke(stage_request())
            assert completion.failure is None
            assert completion.result is not None
            assert completion.result.result == "Used the selected skill"
            assert len(requests) == 4
            resource_reply = json.dumps(requests[2], sort_keys=True)
            assert BINARY_ASSET_STATUS in resource_reply
            assert "/wD+" not in json.dumps(requests)
            assert state.metrics.counters["tool_calls"] == 2
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("media", "skill_media_type_invalid"),
        ("digest", "skill_digest_mismatch"),
        ("archive", "skill_archive_invalid"),
    ],
)
def test_preparation_revalidates_private_artifact(
    tmp_path: Path, case: str, expected_code: str
) -> None:
    async def scenario() -> None:
        payload = b"not-a-zip" if case == "archive" else skill_package()
        selected, value = resolved_value(payload)
        if case == "media":
            value = ArtifactValue(
                artifact=value.artifact,
                media_type="application/zip",
                data=value.data,
                binding_created_at=value.binding_created_at,
                revision_created_at=value.revision_created_at,
            )
        if case == "digest":
            selected = selected.model_copy(update={"package_digest": "sha256:" + "0" * 64})
        workspace = allocation_workspace(tmp_path)
        with pytest.raises(AgentSkillPreparationError) as captured:
            await prepare_agent_skills(
                [selected],
                allocation_id="allocation-1",
                runtime_settings=runtime_settings(),
                workspace=workspace,
                artifact_client_factory=lambda _allocation, _settings: FakeSkillClient(
                    {"demo": value}
                ),
            )
        assert captured.value.code == expected_code
        assert captured.value.retryable is False
        assert not (tmp_path / ".agent-skills").exists()

    asyncio.run(scenario())


def test_transient_private_read_is_retryable_and_leaves_no_extraction(tmp_path: Path) -> None:
    class UnavailableClient:
        async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
            del ref
            raise ArtifactTransportError("unavailable")

    async def scenario() -> None:
        payload = skill_package()
        selected, _ = resolved_value(payload)
        with pytest.raises(AgentSkillPreparationError) as captured:
            await prepare_agent_skills(
                [selected],
                allocation_id="allocation-1",
                runtime_settings=runtime_settings(),
                workspace=allocation_workspace(tmp_path),
                artifact_client_factory=lambda _allocation, _settings: UnavailableClient(),  # type: ignore[arg-type,return-value]
            )
        assert captured.value.code == "skill_artifact_unavailable"
        assert captured.value.retryable is True
        assert not (tmp_path / ".agent-skills").exists()

    asyncio.run(scenario())


def test_disclosure_reservation_is_exact_non_refunding_and_pre_dispatch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="allocation-1",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda _allocation, _settings: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None
        charge = prepared.charge_for("load_skill", "demo")
        prepared.disclosure = DisclosureBudget(
            limit=MAXIMUM_DISCLOSURE_BYTES,
            used=MAXIMUM_DISCLOSURE_BYTES - charge,
        )
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}
        declarations = json.dumps(
            [tool._get_declaration().model_dump(mode="json") for tool in tools.values()],
            sort_keys=True,
        )
        assert all(
            forbidden not in declarations for forbidden in ("run_skill_script", "search_skills")
        )
        context = FakeToolContext()

        loaded = await tools["load_skill"].run_async(
            args={"skill_name": "demo"},
            tool_context=context,
        )
        assert loaded["skill_name"] == "demo"
        assert prepared.disclosure.used == MAXIMUM_DISCLOSURE_BYTES
        activated = list(context.state["_adk_activated_skill_contractor_worker"])

        rejected = await tools["load_skill"].run_async(
            args={"skill_name": "demo"},
            tool_context=context,
        )
        assert rejected["error_code"] == "SKILL_DISCLOSURE_LIMIT"
        assert context.state["_adk_activated_skill_contractor_worker"] == activated

        # A native failure happens after reservation and therefore cannot refund it.
        prepared.disclosure = DisclosureBudget()
        failing = ExplodingNativeTool()
        tools["list_skills"]._native = failing
        list_charge = prepared.charge_for("list_skills")
        failed = await tools["list_skills"].run_async(
            args={},
            tool_context=context,
        )
        assert failed["error_code"] == "SKILL_TOOL_ERROR"
        assert failing.calls == 1
        assert prepared.disclosure.used == list_charge
        await prepared.close()

    asyncio.run(scenario())


def test_cancelled_skill_calls_are_recorded(tmp_path: Path) -> None:
    async def scenario() -> None:
        selected, value = resolved_value(skill_package())
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="allocation-1",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda _allocation, _settings: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}

        # Cancelled while the native ADK tool runs.
        native = BlockingNativeTool()
        tools["list_skills"]._native = native
        running = asyncio.create_task(
            tools["list_skills"].run_async(args={}, tool_context=FakeToolContext())
        )
        await native.started.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        # Cancelled while waiting for the disclosure reservation.
        disclosure = BlockingDisclosure()
        prepared.disclosure = disclosure
        waiting = asyncio.create_task(
            tools["load_skill"].run_async(
                args={"skill_name": "demo"}, tool_context=FakeToolContext()
            )
        )
        await disclosure.started.wait()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        listed, loaded = state.metrics.tool_calls
        for call, name in ((listed, "list_skills"), (loaded, "load_skill")):
            assert call.tool == name
            assert call.error is not None
            assert call.error.code == "tool_call_cancelled"
            assert call.error.retryable is True
        assert loaded.arguments == {"skill_name": "demo"}
        assert state.metrics.counters["tool_errors"] == 2
        await prepared.close()

    asyncio.run(scenario())


def test_concurrent_disclosure_reservations_stop_at_the_exact_allocation_limit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="allocation-1",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda _allocation, _settings: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None
        charge = prepared.charge_for("list_skills")
        prepared.disclosure = DisclosureBudget(
            used=MAXIMUM_DISCLOSURE_BYTES - 3 * charge,
        )
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}
        results = await asyncio.gather(
            *(
                tools["list_skills"].run_async(
                    args={},
                    tool_context=FakeToolContext(f"invocation-{index}"),
                )
                for index in range(12)
            )
        )
        rejected = [
            result
            for result in results
            if isinstance(result, dict) and result.get("error_code") == "SKILL_DISCLOSURE_LIMIT"
        ]
        assert len(rejected) == 9
        assert prepared.disclosure.used == MAXIMUM_DISCLOSURE_BYTES
        assert state.metrics.counters["tool_calls"] == 12
        assert state.metrics.counters["tool_errors"] == 9
        await prepared.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "raw",
    [
        "../../recognizable-private-path",
        "references/../recognizable-private-path",
        "references/%2e%2e/recognizable-private-path",
        "references//recognizable-private-path",
        r"references\recognizable-private-path",
        "/references/recognizable-private-path",
        "references/приватный-path",
        "references/recognizable-private-path\x00",
    ],
)
def test_invalid_model_arguments_are_not_retained_or_dispatched(tmp_path: Path, raw: str) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="allocation-1",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda _allocation, _settings: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}
        result = await tools["load_skill_resource"].run_async(
            args={"skill_name": "demo", "file_path": raw},
            tool_context=FakeToolContext(),
        )
        assert result["error_code"] == "INVALID_ARGUMENTS"
        snapshot = json.dumps(state.metrics.snapshot(), sort_keys=True)
        assert raw not in snapshot
        assert state.metrics.tool_calls[0].arguments == {"arguments_valid": False}
        assert prepared.disclosure.used == 0
        await prepared.close()

    asyncio.run(scenario())


def test_binary_resource_returns_bounded_text_and_charges_only_text(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="allocation-1",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda _allocation, _settings: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}
        result = await tools["load_skill_resource"].run_async(
            args={"skill_name": "demo", "file_path": "assets/pixel.bin"},
            tool_context=FakeToolContext("invocation-binary"),
        )
        assert result == {
            "skill_name": "demo",
            "file_path": "assets/pixel.bin",
            "status": BINARY_ASSET_STATUS,
        }
        visible_size = len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
        charge = prepared.charge_for("load_skill_resource", "demo", "assets/pixel.bin")
        assert charge == DISCLOSURE_ENVELOPE_BYTES + visible_size
        assert prepared.disclosure.used == charge
        assert state.metrics.tool_calls[0].result_size_bytes == visible_size
        await prepared.close()

    asyncio.run(scenario())


def test_oversized_native_binary_result_is_suppressed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="allocation-1",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda _allocation, _settings: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}
        charge = prepared.charge_for("load_skill_resource", "demo", "assets/pixel.bin")
        tools["load_skill_resource"]._native = OversizedBinaryNativeTool(charge)
        context = FakeToolContext("invocation-oversized")
        result = await tools["load_skill_resource"].run_async(
            args={"skill_name": "demo", "file_path": "assets/pixel.bin"},
            tool_context=context,
        )
        assert result["error_code"] == "SKILL_DISCLOSURE_ESTIMATE_INVALID"
        retained = json.dumps(state.metrics.snapshot(), sort_keys=True)
        assert "oversized-native-result-canary" not in retained
        await prepared.close()

    asyncio.run(scenario())


def test_native_agent_skill_probe_checks_pinned_surface() -> None:
    assert asyncio.run(probe_native_agent_skills()) is True


@pytest.mark.parametrize(
    "changed", ["registry", "executor", "environment", "tools", "toolsets", "surface"]
)
def test_native_probe_rejects_expanded_or_executing_surfaces(
    monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    original = skill_runtime.SkillToolset

    class UninitializedEnvironment:
        # ADK tracks environment cleanup with weak keys. Keep the rejected
        # surface realistic without initializing or executing anything.
        is_initialized = False

    class UnsupportedToolset(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            attributes = {
                "registry": ("_registry", object()),
                "executor": ("_code_executor", object()),
                "environment": ("_env", UninitializedEnvironment()),
                "tools": (
                    "_provided_tools_by_name",
                    {"unexpected": BaseTool(name="unexpected", description="unsupported")},
                ),
                "toolsets": ("_provided_toolsets", [original(skills=[])]),
            }
            if changed in attributes:
                name, value = attributes[changed]
                setattr(self, name, value)

        async def get_tools(self, *args, **kwargs):
            tools = await super().get_tools(*args, **kwargs)
            if changed == "surface":
                tools.append(BaseTool(name="run_skill_script", description="unsupported"))
            return tools

    monkeypatch.setattr(skill_runtime, "SkillToolset", UnsupportedToolset)
    assert asyncio.run(probe_native_agent_skills()) is False


def test_all_32_skills_use_lazy_native_dispatch_and_are_erased(tmp_path: Path) -> None:
    async def scenario() -> None:
        assert MAXIMUM_PACKAGES == 32
        selected, values, responses = [], {}, [tool_call("list_skills", {}, call_id="all-skills")]
        for index in range(MAXIMUM_PACKAGES):
            name = f"skill-{index:02d}"
            pin, value = resolved_value(
                skill_package(name=name, description="я" * 512, instructions=f"Guidance {index}"),
                name=name,
            )
            selected.append(pin)
            values[name] = value
            responses.append(tool_call("load_skill", {"skill_name": name}, call_id=f"load-{index}"))
        responses.append(text_result("All selected guidance loaded"))
        state = WorkerState()
        context = build_context(tmp_path, state, selected[0])
        context = replace(
            context,
            resolved_skills=tuple(selected),
            model_policy=context.model_policy.model_copy(
                update={"max_model_calls": 40, "max_tool_calls": 40}
            ),
        )
        client = FakeSkillClient(values)
        model = scripted_model(responses)
        runtime = await AdkWorkerRuntimeFactory(lambda _: model, lambda *_: client).create(context)
        owner = runtime._agent_skills
        assert owner is not None and len(owner.skills) == MAXIMUM_PACKAGES
        assert owner.native_toolset._discovery_mode is SkillDiscoveryMode.LAZY
        assert owner.native_toolset._lifecycle_enabled is False
        native_skills = list(owner.skills)
        result = await runtime.invoke(stage_request())
        assert result.failure is None and result.result is not None
        assert len(client.reads) == MAXIMUM_PACKAGES
        assert len(model.requests) == MAXIMUM_PACKAGES + 3
        assert all(
            request["toolNames"] == sorted(EXACT_SKILL_TOOL_NAMES)
            for request in model.requests[:-1]
        )
        session_id = runtime._session_lifecycle.shared_session_id
        session = await runtime._session_service.get_session(
            app_name=runtime._app_name, user_id=runtime._user_id, session_id=session_id
        )
        assert session is not None
        assert session.state["_adk_activated_skill_contractor_worker"] == list(values)
        assert owner.disclosure.used == owner.charge_for("list_skills") + sum(
            owner.charge_for("load_skill", name) for name in values
        )
        assert owner.disclosure.used < MAXIMUM_DISCLOSURE_BYTES
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=5))
        assert all(skill.instructions == "" for skill in native_skills)
        assert owner.skills == [] and owner.native_tools == {}
        assert not owner.extraction_root.exists()
        reads = []

        def forbidden_client(*_args):
            reads.append(True)
            raise AssertionError("33 Skills must fail before client creation")

        with pytest.raises(AgentSkillPreparationError) as rejected:
            await prepare_agent_skills(
                [*selected, selected[-1]],
                allocation_id="too-many-skills",
                runtime_settings=runtime_settings(),
                workspace=allocation_workspace(tmp_path),
                artifact_client_factory=forbidden_client,
            )
        assert rejected.value.code == "skill_runtime_unsupported" and reads == []

    asyncio.run(scenario())


def test_multibyte_disclosure_reaches_16_mib_without_seed_or_refund(tmp_path: Path) -> None:
    async def scenario() -> None:
        charge = 1 << 20
        envelope = DISCLOSURE_ENVELOPE_BYTES + len(
            json.dumps(
                {"skill_name": "demo", "file_path": "references/guide.md", "content": ""},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        text = "я" * 1000 + "x" * (charge - envelope - 2000)
        over_path = "references/extra.md"
        over = text + "x" * (len("references/guide.md") - len(over_path) + 1)
        selected, value = resolved_value(
            skill_package(reference=text.encode(), extra_resources=((over_path, over.encode()),))
        )
        prepared = await prepare_agent_skills(
            [selected],
            allocation_id="maximum-disclosure",
            runtime_settings=runtime_settings(),
            workspace=allocation_workspace(tmp_path),
            artifact_client_factory=lambda *_: FakeSkillClient({"demo": value}),
        )
        assert prepared is not None and prepared.disclosure.used == 0
        assert prepared.disclosure.limit == MAXIMUM_DISCLOSURE_BYTES
        assert prepared.charge_for("load_skill_resource", "demo", "references/guide.md") == charge
        assert prepared.charge_for("load_skill_resource", "demo", over_path) == charge + 1
        state = WorkerState()
        adapter = prepared.build_adapter(metrics=state.metrics)
        tools = {tool.name: tool for tool in await adapter.get_tools()}
        tool = tools["load_skill_resource"]
        native = tool._native
        dispatched = []
        original = native.run_async

        async def dispatch(*, args, tool_context):
            dispatched.append(args["file_path"])
            return await original(args=args, tool_context=tool_context)

        native.run_async = dispatch
        context = FakeToolContext()
        arguments = {"skill_name": "demo", "file_path": "references/guide.md"}
        for _ in range(15):
            result = await tool.run_async(args=arguments, tool_context=context)
            assert result["content"] == text
        assert prepared.disclosure.used == MAXIMUM_DISCLOSURE_BYTES - charge
        rejected = await tool.run_async(
            args={"skill_name": "demo", "file_path": over_path}, tool_context=context
        )
        assert rejected["error_code"] == "SKILL_DISCLOSURE_LIMIT"
        assert "content" not in rejected and len(dispatched) == 15
        result = await tool.run_async(args=arguments, tool_context=context)
        assert result["content"] == text
        assert prepared.disclosure.used == MAXIMUM_DISCLOSURE_BYTES
        rejected = await tool.run_async(args=arguments, tool_context=context)
        assert rejected["error_code"] == "SKILL_DISCLOSURE_LIMIT"
        assert len(dispatched) == 16 and over_path not in dispatched
        assert state.metrics.counters["tool_calls"] == 18
        assert state.metrics.counters["tool_errors"] == 2
        assert text[:1000] not in json.dumps(state.metrics.snapshot(), ensure_ascii=False)
        await prepared.close()
        assert not prepared.extraction_root.exists()

    asyncio.run(scenario())


class FakeSkillClient:
    def __init__(self, values: dict[str, ArtifactValue]) -> None:
        self.values = values
        self.reads: list[ArtifactRef] = []

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.reads.append(ref)
        return self.values[ref.name]


class FakeToolContext:
    agent_name = "contractor_worker"

    def __init__(self, invocation_id: str = "invocation-1") -> None:
        self.invocation_id = invocation_id
        self.state: dict[str, Any] = {}


class ExplodingNativeTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="list_skills", description="fails")
        self.calls = 0

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        del args, tool_context
        self.calls += 1
        raise RuntimeError("private native failure")


class BlockingNativeTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="list_skills", description="blocks")
        self.started = asyncio.Event()

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        del args, tool_context
        self.started.set()
        await asyncio.Event().wait()


class BlockingDisclosure:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def reserve(self, charge: int) -> bool:
        del charge
        self.started.set()
        await asyncio.Event().wait()
        return True


class OversizedBinaryNativeTool(BaseTool):
    def __init__(self, charge: int) -> None:
        super().__init__(name="load_skill_resource", description="oversized")
        self._charge = charge

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        del args, tool_context
        return {
            "skill_name": "demo",
            "file_path": "assets/pixel.bin",
            "status": "oversized-native-result-canary" + "x" * self._charge,
        }


def skill_package(
    *,
    script: bytes | None = None,
    name: str = "demo",
    description: str = "Demonstrate a bounded workflow.",
    instructions: str = "Follow the exact demo procedure.",
    reference: bytes = b"Reference contents.",
    extra_resources: tuple[tuple[str, bytes], ...] = (),
) -> bytes:
    entries = [
        (
            "SKILL.md",
            (
                f"---\nname: {name}\ndescription: {description}\n"
                f"metadata:\n  author: contractor\n---\n{instructions}\n"
            ).encode(),
        ),
        ("references/guide.md", reference),
        ("assets/pixel.bin", b"\xff\x00\xfe"),
    ]
    entries.extend(extra_resources)
    if script is not None:
        entries.append(("scripts/check.py", script))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return output.getvalue()


def resolved_value(payload: bytes, *, name: str = "demo") -> tuple[ResolvedSkill, ArtifactValue]:
    artifact = ArtifactRef(namespace="skills", name=name, revision="skill-r1")
    selected = ResolvedSkill(
        name=name,
        artifact=artifact,
        packageDigest=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
    now = datetime.now(UTC)
    value = ArtifactValue(
        artifact=artifact,
        media_type=MEDIA_TYPE,
        data=payload,
        binding_created_at=now,
        revision_created_at=now,
    )
    return selected, value


def build_context(
    tmp_path: Path, state: WorkerState, selected: ResolvedSkill
) -> WorkerBuildContext:
    template = ResolvedAgentTemplate(
        ref=AgentTemplateRef(templateId="skilled-worker", version="1", digest="sha256:" + "0" * 64),
        description="Use one selected Agent Skill",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/skilled.md",
            digest="sha256:" + "1" * 64,
            text="Use the selected guidance and return a strict result.",
        ),
        toolsets=[],
        skills=[ArtifactRef(namespace="skills", name="demo")],
        sandboxProfile=SandboxProfileRef(sandboxProfileId="local-workdir", version="1"),
    )
    return WorkerBuildContext(
        allocation_id="allocation-1",
        run_id="run-1",
        stage_execution_id="stage-execution-1",
        logical_agent_name="builder",
        namespace="builder",
        worker_session_mode=WorkerSessionMode.SHARED,
        description=template.description,
        instruction=template.instructions.text,
        card_version=template.ref.version,
        model_policy=worker_policy(),
        workspace=allocation_workspace(tmp_path),
        tools={},
        state=state,
        resources=state.resources,
        a2a_base_url="https://runtime.example",
        runtime_settings=runtime_settings(),
        resolved_skills=(selected,),
    )


def allocation_workspace(tmp_path: Path) -> AllocationWorkspace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return AllocationWorkspace(root=tmp_path.parent, path=tmp_path)


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken="secret-token",
        artifactApiUrl="https://control.example",
        requestTimeoutSeconds=5,
    )


def stage_request() -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        deadline=datetime.now(UTC) + timedelta(hours=1),
        subtaskId="0",
        objective="Use the selected skill",
        instructions="Complete the task.",
        parameters={},
        artifacts={},
    )


def worker_policy() -> ResolvedModelPolicy:
    return ResolvedModelPolicy(
        ref=ModelPolicyRef(policyId="worker", version="1", digest="sha256:" + "2" * 64),
        model="worker-model",
        maxOutputTokens=4096,
        maxModelCalls=8,
        maxToolCalls=8,
        maxTotalTokens=32768,
        temperature=0.1,
    )
