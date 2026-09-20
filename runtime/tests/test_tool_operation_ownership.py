"""Cancelled source/validator work remains owned through allocation cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fakes.model import scripted_model, tool_call
from fakes.spec import allocation_spec
from test_likec4_toolset import BASE_DOCUMENT
from test_likec4_toolset import MemoryArtifactClient as LikeC4ArtifactClient
from test_likec4_toolset import make_tools as make_likec4_tools
from test_openapi_toolset import MemoryArtifactClient as OpenAPIArtifactClient
from test_source_analysis_toolset import ReadOnlyArtifactClient, make_zip

import contractor_runtime.toolsets.likec4.tools as likec4_tools
import contractor_runtime.toolsets.source_analysis.tools as source_tools
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import RuntimeAdapterMetricsState
from contractor_runtime.adapters.http_proxy import ProxySubprocessLauncher
from contractor_runtime.allocation import AllocationService, WorkerState
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    ReleaseAllocationRequest,
    RuntimeSettings,
    StageContentRequest,
    TerminationError,
    ToolsetRef,
    ToolsetSelection,
)
from contractor_runtime.digests import _agent_template_digest
from contractor_runtime.factories import FactoryRegistry
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.toolsets.likec4.tools import LikeC4ToolsetFactory
from contractor_runtime.toolsets.openapi.tools import OpenAPIToolsetFactory
from contractor_runtime.toolsets.source_analysis.tools import SourceAnalysisToolsetFactory
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory
from contractor_runtime.workspace import LocalWorkdirFactory


@pytest.mark.parametrize("phase", ["extract", "install"])
def test_source_abort_and_release_join_file_work_before_reusing_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed(
            "inputs", "source", "application/zip", make_zip({"src/private.py": "source"})
        )
        model = scripted_model(
            [tool_call("open_source_archive", ref.model_dump(), call_id="source-1")]
        )
        factory = SourceAnalysisToolsetFactory(lambda *_: client)
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": AdkWorkerRuntimeFactory(model_factory=lambda _: model)},
            toolsets={"source-analysis@1": factory},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=["adk@1"],
            toolsets={"source-analysis@1": ["open_source_archive"]},
            sandbox_profiles=["local-workdir@1"],
        )
        state = RuntimeState(instance_id="runtime-owned-source")
        await state.mark_registered()
        exits: list[int] = []
        service = AllocationService(
            state,
            registry,
            capabilities,
            a2a_base_url="https://runtime.example",
            force_exit=exits.append,
        )
        spec = allocation_spec()
        spec.agent_template.toolsets = [
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="source-analysis", version="1"),
                tools=["open_source_archive"],
            )
        ]
        spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
        await service.prepare(spec)
        context = service._context
        assert context is not None and context.worker is not None
        assert context.execution is None  # Ordinary local-workdir has no process supervisor.
        workspace = context.workspace.path
        started, resume = threading.Event(), threading.Event()
        owner = source_tools if phase == "extract" else source_tools._SourceArchiveSession
        name = "_read_member_bounded" if phase == "extract" else "_install_staging"
        original = getattr(owner, name)

        def blocked(*args):
            started.set()
            assert resume.wait(5), "file work was never released"
            return original(*args)

        monkeypatch.setattr(owner, name, blocked)
        invocation = asyncio.create_task(
            context.worker.invoke(
                StageContentRequest(
                    apiVersion=API_VERSION,
                    subtaskId="0",
                    objective="Read source",
                    instructions="Open source",
                    parameters={},
                    artifacts={"source": ref},
                    resultArtifacts={},
                )
            )
        )
        abort = None
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            abort = asyncio.create_task(
                service.abort(
                    AbortAllocationRequest(
                        apiVersion=API_VERSION,
                        allocationId=spec.allocation_id,
                        abortId="cancel-source",
                        reason=TerminationError(
                            code="user_cancelled", message="cancelled", retryable=False
                        ),
                        deadline=datetime.now(UTC) + timedelta(seconds=3),
                    )
                )
            )
            await asyncio.sleep(0.05)
            assert not abort.done(), "abort confirmed cleanup while file work was still running"
            resume.set()
            response = await asyncio.wait_for(abort, 3)
            await asyncio.gather(invocation, return_exceptions=True)
            await service.release(
                ReleaseAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                )
            )
            await service.confirm_release(spec.allocation_id)
            assert response.report.worker.complete
            assert (await state.snapshot()).process_state is ProcessState.IDLE
            assert not workspace.exists()
            assert list(tmp_path.iterdir()) == []
            assert exits == []
        finally:
            resume.set()
            await asyncio.gather(invocation, *([abort] if abort else []), return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("validator", ["likec4", "vacuum"])
@pytest.mark.parametrize("proxied", [False, True])
@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_cancelled_validator_reaps_child_before_tool_and_workspace_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validator: str, proxied: bool, stop: str
) -> None:
    marker = tmp_path / "pids"
    executable = tmp_path / validator
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "from pathlib import Path\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(60)\n"
        "else:\n"
        f"    Path({str(marker)!r}).write_text(str(os.getpid()) + ' ' + str(child))\n"
        "    time.sleep(60)\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    processes: list[subprocess.Popen] = []
    popen = subprocess.Popen

    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", track)

    async def scenario() -> None:
        workdirs = LocalWorkdirFactory(tmp_path / "work")
        workspace = await workdirs.prepare()
        launcher = (
            ProxySubprocessLauncher(
                proxy_url="http://proxy.invalid:8080",
                basic_auth=None,
                bearer_token=None,
                combined_ca_bundle=None,
                bypass_hosts=(),
                timeout_seconds=30,
                metrics=RuntimeAdapterMetricsState(),
            )
            if proxied
            else None
        )
        factory = (
            LikeC4ToolsetFactory(lambda *_: LikeC4ArtifactClient())
            if validator == "likec4"
            else OpenAPIToolsetFactory(lambda *_: OpenAPIArtifactClient())
        )
        tools = await factory.create_selected(
            selected=sorted(factory.exported_tools),
            allocation_id="allocation-test",
            run_id="run-test",
            namespace="builder",
            workspace=workspace,
            state=WorkerState(),
            runtime_settings=RuntimeSettings(
                llmGatewayUrl="https://gateway.example/v1",
                artifactApiUrl="https://control.example/private/v1",
                requestTimeoutSeconds=30,
            ),
            adapter_handles=AdapterHandles(tool_subprocess=launcher),
        )
        if validator == "likec4":
            await tools["write_likec4"](BASE_DOCUMENT)
            validation = tools["validate_likec4"]
        else:
            await tools["initialize_openapi"]("Test")
            validation = tools["validate_openapi"]
        task = asyncio.create_task(validation())
        pids: list[int] = []
        try:
            async with asyncio.timeout(3):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            pids = [int(value) for value in marker.read_text().split()]
            if stop == "cancel":
                task.cancel()
            else:
                await asyncio.wait_for(validation.close(), 3)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            for tool in tools.values():
                await tool.close()
            await workdirs.cleanup(workspace)
            assert not workspace.path.exists()
            assert not Path(f"/proc/{pids[0]}").exists(), "validator leader was not reaped"
            async with asyncio.timeout(3):
                while _alive(pids[1]):
                    await asyncio.sleep(0.01)
        finally:
            for pid in pids:
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)
            for process in processes:
                await asyncio.to_thread(process.wait, timeout=3)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if launcher is not None:
                launcher.close()

    asyncio.run(scenario())


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("phase", ["prepare", "cleanup"])
def test_likec4_close_joins_cancelled_file_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def scenario() -> None:
        tools = await make_likec4_tools(
            tmp_path, LikeC4ArtifactClient(), WorkerState(), namespace="builder"
        )
        await tools["write_likec4"](BASE_DOCUMENT)
        started, finish = threading.Event(), threading.Event()
        owner = Path if phase == "prepare" else likec4_tools.tempfile.TemporaryDirectory
        name = "write_text" if phase == "prepare" else "cleanup"
        original = getattr(owner, name)

        def blocked(*args, **kwargs):
            started.set()
            assert finish.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, blocked)
        monkeypatch.setattr(likec4_tools.shutil, "which", lambda _: "/unused/likec4")
        monkeypatch.setattr(
            likec4_tools,
            "run_command",
            AsyncMock(return_value=subprocess.CompletedProcess([], 0, b"[]", b"")),
        )
        task = asyncio.create_task(tools["validate_likec4"]())
        closing = None
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            closing = asyncio.create_task(tools["validate_likec4"].close())
            await asyncio.sleep(0.05)
            assert not closing.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            await asyncio.wait_for(closing, 3)
            assert not list(tmp_path.glob(".likec4-validate-*"))
        finally:
            finish.set()
            await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)

    asyncio.run(scenario())
