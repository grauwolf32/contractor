from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_projectfs_zip import settings as workspace_settings
from test_workspace_process_e2e import direct_workspace_spec

from contractor_runtime.adapters.host import AdapterHandles
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot, discover_capabilities
from contractor_runtime.contracts import SandboxProfileRef, ToolsetRef, ToolsetSelection
from contractor_runtime.digests import _agent_template_digest
from contractor_runtime.factories import FactoryRegistry, built_in_factories
from contractor_runtime.sandbox_contracts import (
    EXECUTION_CHANNELS,
    EXECUTION_TOOLS,
    EXECUTION_TOOLSET,
    PODMAN_PROFILE,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
    SandboxErrorCode,
    SandboxIdentity,
    selected_executor,
    validate_sandbox_selection,
)
from contractor_runtime.state import RuntimeState


def sandbox_spec():
    spec = direct_workspace_spec("sandbox-contract")
    spec.agent_template.sandbox_profile = SandboxProfileRef(sandboxProfileId="podman", version="1")
    spec.agent_template.toolsets = [
        ToolsetSelection(
            ref=ToolsetRef(toolsetId="code-execution", version="1"), tools=["exec_command"]
        )
    ]
    spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
    return spec


@pytest.mark.parametrize(
    "mode,storage,profile,execution,valid",
    [
        ("direct", "local", "podman", True, True),
        ("direct", "local", "podman", False, True),
        (None, "local", "podman", False, False),
        ("overlay", "local", "podman", True, False),
        ("direct", "memory", "podman", True, False),
        ("direct", None, "podman", True, False),
        ("direct", "local", "local-workdir", True, False),
        ("overlay", "memory", "local-workdir", False, True),
        ("direct", "memory", "local-workdir", False, True),
    ],
)
def test_exact_sandbox_workspace_and_tool_selection(mode, storage, profile, execution, valid):
    spec = sandbox_spec()
    if mode is None:
        spec.workspace = None
    else:
        spec.workspace.mode = mode
    spec.agent_template.sandbox_profile.sandbox_profile_id = profile
    if not execution:
        spec.agent_template.toolsets = []
    if valid:
        validate_sandbox_selection(spec, storage)
    else:
        with pytest.raises(SandboxContractError, match="sandbox_incompatible"):
            validate_sandbox_selection(spec, storage)


@pytest.mark.parametrize(
    "mode,storage,code",
    [
        (None, "local", "sandbox_incompatible"),
        ("overlay", "local", "sandbox_incompatible"),
        ("direct", "memory", "sandbox_incompatible"),
        ("direct", "local", "unsupported_sandbox_profile"),
    ],
)
def test_prepare_rejects_before_resource_creation(tmp_path: Path, mode, storage, code) -> None:
    async def scenario() -> None:
        factories = built_in_factories(
            tmp_path / "scratch",
            workspace_settings=workspace_settings(
                storage, tmp_path / "project" if storage == "local" else None
            ),
        )
        capability = CapabilitySnapshot.create(
            runtimes=["adk@1"],
            toolsets={},
            sandbox_profiles=["local-workdir@1"],
            workspace=factories.workspace_provider.capability,
        )
        state = RuntimeState(instance_id="sandbox-test", capabilities=capability)
        await state.mark_registered()
        service = AllocationService(
            state, factories, capability, a2a_base_url="https://runtime.test"
        )
        spec = sandbox_spec()
        if mode is None:
            spec.workspace = None
        else:
            spec.workspace.mode = mode
        with pytest.raises(AllocationError) as failure:
            await service.prepare(spec)
        assert failure.value.code == code
        assert service._context is None
        assert not (tmp_path / "scratch").exists()
        assert not (tmp_path / "project").exists()

    asyncio.run(scenario())


def test_execution_contract_is_not_an_installed_factory_or_capability(tmp_path: Path) -> None:
    async def scenario() -> None:
        factories = built_in_factories(tmp_path)
        assert PODMAN_PROFILE not in factories.sandbox_profiles
        assert EXECUTION_TOOLSET not in factories.toolsets
        snapshot = await discover_capabilities(factories)
        assert not snapshot.supports_sandbox(PODMAN_PROFILE)
        assert not snapshot.has_toolset(EXECUTION_TOOLSET)
        assert snapshot.supports_sandbox("local-workdir@1")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"command": ""}, "sandbox_invalid_command"),
        ({"command": "x" * ((64 << 10) + 1)}, "sandbox_invalid_command"),
        ({"command": "я" * (32 << 10) + "!"}, "sandbox_invalid_command"),
        ({"command": "secret\x00"}, "sandbox_invalid_command"),
        ({"command": "\ud800"}, "sandbox_invalid_command"),
        ({"timeout_seconds": True}, "sandbox_invalid_command"),
        ({"timeout_seconds": 0}, "sandbox_invalid_command"),
        ({"timeout_seconds": 3601}, "sandbox_invalid_command"),
        ({"cwd": "/etc"}, "sandbox_invalid_cwd"),
        ({"cwd": "../"}, "sandbox_invalid_cwd"),
        ({"cwd": "C:\\temp"}, "sandbox_invalid_cwd"),
        ({"cwd": "a//b"}, "sandbox_invalid_cwd"),
    ],
)
def test_execution_request_limits_and_safe_errors(changes, code) -> None:
    with pytest.raises(SandboxContractError) as failure:
        ExecutionRequest(**{"command": "secret", **changes})
    assert str(failure.value) == code


def test_execution_request_preserves_shell_input_without_host_execution() -> None:
    command = "echo '$HOME'; $(touch /tmp/should-not-run); python -c 'print(1)'"
    request = ExecutionRequest(command, cwd="src")
    assert request.command == command and request.timeout_seconds == 60
    assert command not in repr(request) and "src" not in repr(request)
    assert ExecutionRequest("я" * (32 << 10)).command
    with pytest.raises(TypeError):
        ExecutionRequest("true", image="host")


def test_nonzero_result_is_completed_and_observation_excludes_private_identity() -> None:
    result = ExecutionResult(ExecutionStatus.COMPLETED, 7, "secret", "", False, True, 12)
    assert result.observation() == {
        "status": "completed",
        "exitCode": 7,
        "stdout": "secret",
        "stderr": "",
        "stdoutTruncated": False,
        "stderrTruncated": True,
        "durationMs": 12,
        "errorCode": None,
    }
    assert "secret" not in repr(result)
    for changes in (
        {"exit_code": None},
        {"duration_ms": -1},
        {"stdout_truncated": 1},
        {"error_code": SandboxErrorCode.OUTCOME_UNKNOWN},
        {"stdout": "x" * ((3 << 20) + 1)},
    ):
        with pytest.raises(ValueError):
            replace(result, **changes)
    failure = ExecutionResult(
        ExecutionStatus.TIMED_OUT, None, "", "", False, False, 60, SandboxErrorCode.TIMEOUT
    )
    assert failure.observation()["errorCode"] == "sandbox_timeout"


def test_full_container_identity_and_narrow_selected_handle() -> None:
    host_handles = AdapterHandles(
        tool_http=object(), tool_subprocess=object(), caido_graphql=object()
    )
    assert host_handles.for_tool_channels({"sandbox-execution"}).enabled_channels == ()
    identity = SandboxIdentity("runtime", "incarnation", "allocation", "creation", "a" * 64)
    assert repr(identity) == "SandboxIdentity()"
    with pytest.raises(ValueError):
        replace(identity, container_id="a" * 12)
    executor = SimpleNamespace(execute=lambda: None)
    assert (
        selected_executor(EXECUTION_TOOLSET, ("exec_command",), EXECUTION_CHANNELS, executor)
        is executor
    )
    for ref, tools, channels, handle in [
        ("filesystem@1", ("read_file",), EXECUTION_CHANNELS, executor),
        (EXECUTION_TOOLSET, (), EXECUTION_CHANNELS, executor),
        (
            EXECUTION_TOOLSET,
            ("exec_command",),
            {"exec_command": frozenset({"runtime-subprocess-launcher"})},
            executor,
        ),
        (EXECUTION_TOOLSET, ("exec_command",), EXECUTION_CHANNELS, None),
    ]:
        with pytest.raises(SandboxContractError):
            selected_executor(ref, tools, channels, handle)


def test_factory_cannot_mix_sandbox_and_host_authority(tmp_path: Path) -> None:
    factories = built_in_factories(tmp_path)
    for ref, channels in [
        (EXECUTION_TOOLSET, frozenset({"runtime-subprocess-launcher"})),
        (EXECUTION_TOOLSET, frozenset({"sandbox-execution", "runtime-subprocess-launcher"})),
        ("other@1", frozenset({"sandbox-execution"})),
    ]:
        factory = SimpleNamespace(
            ref=ref,
            exported_tools=EXECUTION_TOOLS,
            infrastructure_channels={"exec_command": channels},
        )
        with pytest.raises(ValueError, match="isolated Toolset channel"):
            FactoryRegistry(factories.worker_runtimes, {ref: factory}, factories.sandbox_profiles)
