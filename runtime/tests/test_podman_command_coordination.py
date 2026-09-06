from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest
from fakes.podman_lifecycle import owner
from test_podman_allocation import finalization, release, service_for

from contractor_runtime.podman_command import CommandCapture
from contractor_runtime.projectfs.errors import WorkspaceStorageError
from contractor_runtime.sandbox_contracts import ExecutionRequest, SandboxErrorCode


def deadline(seconds=5):
    return datetime.now(UTC) + timedelta(seconds=seconds)


class Commands:
    def __init__(self):
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()
        self.calls = []
        self.result = CommandCapture(0, b"ok", b"", 2, 0)
        self.edit = lambda: None

    async def run(self, identity, command, cwd, *, deadline):
        self.calls.append((identity, command, cwd, deadline))
        self.entered.set()
        await self.resume.wait()
        self.edit()
        return self.result


def test_commands_serialize_with_disk_reads_and_edits_until_completion_check(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _state, spec, _events, _exits = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            context = service._context
            commands = Commands()
            fixture.backend.commands = commands
            commands.edit = lambda: (fixture.backend.entry.root / "src/main.py").write_text("new")
            executing = asyncio.create_task(
                context.execution.executor.execute(
                    ExecutionRequest("exact; shell syntax", "src"), deadline=deadline()
                )
            )
            await asyncio.wait_for(commands.entered.wait(), 2)
            read = asyncio.create_task(context.project_workspace.read_text("src/main.py"))
            await asyncio.sleep(0.02)
            assert not read.done()
            commands.resume.set()
            result = await executing
            assert result.exit_code == 0 and await read == "new"
            await context.project_workspace.write_text("src/main.py", "edited")
            assert (fixture.backend.entry.root / "src/main.py").read_text() == "edited"
            assert commands.calls[0][1:3] == ("exact; shell syntax", "src")
            assert [op for op, *_ in fixture.guardians[0].requests].count("check") >= 3
            await service.finalize(finalization(spec))
            await service.release(release(spec))

    asyncio.run(scenario())


def test_symlink_cwd_is_nonfatal_and_launches_nothing(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _, spec, _, _ = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            context = service._context
            commands = Commands()
            fixture.backend.commands = commands
            (fixture.backend.entry.root / "link").symlink_to("src", target_is_directory=True)
            result = await context.execution.executor.execute(
                ExecutionRequest("true", "link"), deadline=deadline()
            )
            assert result.error_code == SandboxErrorCode.INVALID_CWD
            assert not context.execution.rejected and commands.calls == []
            await service.finalize(finalization(spec))
            await service.release(release(spec))

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_waiter_timeout_or_cancel_never_releases_running_writer(tmp_path, cancel):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _, spec, _, _ = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            context = service._context
            guard = context.project_workspace.execution_guard
            commands = Commands()
            fixture.backend.commands = commands
            executing = asyncio.create_task(
                context.execution.executor.execute(
                    ExecutionRequest("blocked"), deadline=deadline(0.15)
                )
            )
            await asyncio.wait_for(commands.entered.wait(), 2)
            if cancel:
                executing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await executing
            else:
                result = await executing
                assert result.error_code == SandboxErrorCode.TIMEOUT
            assert guard.fenced and guard._lock.locked()
            with pytest.raises(WorkspaceStorageError):
                await context.project_workspace.read_text("src/main.py")
            assert context.worker_state.execution.failure is not None
            commands.resume.set()
            await service.finalize(finalization(spec))
            await service.release(release(spec))
            assert not guard._lock.locked()
            assert len(commands.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code",
    [SandboxErrorCode.TIMEOUT, SandboxErrorCode.OUTPUT_LIMIT, SandboxErrorCode.OUTCOME_UNKNOWN],
)
def test_failed_transport_stops_entire_container_before_return(tmp_path, code):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _, spec, _, _ = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            context = service._context
            commands = Commands()
            commands.result = CommandCapture(None, b"partial", b"", 7, 0, code)
            commands.resume.set()
            fixture.backend.commands = commands
            result = await context.execution.executor.execute(
                ExecutionRequest("bad"), deadline=deadline()
            )
            assert result.error_code == code and result.exit_code is None
            assert context.execution.stopped.is_set()
            assert all(not row["State"]["Running"] for row in fixture.cli.containers.values())
            assert context.project_workspace.execution_guard.fenced
            await service.finalize(finalization(spec))
            await service.release(release(spec))

    asyncio.run(scenario())


def test_deadline_waiting_for_guard_launches_nothing_and_cannot_revive(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _, spec, _, _ = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            context = service._context
            commands = Commands()
            fixture.backend.commands = commands
            guard = context.project_workspace.execution_guard
            entered, resume = asyncio.Event(), asyncio.Event()

            async def filesystem_operation():
                entered.set()
                await resume.wait()

            reading = asyncio.create_task(
                guard.run_async(filesystem_operation, deadline=time.monotonic() + 5)
            )
            await entered.wait()
            result = await context.execution.executor.execute(
                ExecutionRequest("never"), deadline=deadline(0.05)
            )
            assert result.error_code == SandboxErrorCode.TIMEOUT
            assert commands.calls == [] and guard._lock.locked() and guard.fenced
            resume.set()
            await reading
            await service.finalize(finalization(spec))
            await service.release(release(spec))
            assert commands.calls == []

    asyncio.run(scenario())


def test_selected_execution_changes_analysis_and_explicit_artifact_publication(tmp_path):
    import base64
    from dataclasses import replace

    from test_code_analysis_shallow import _tools as analysis_tools
    from test_run_artifacts_toolset import FakeArtifactClient

    from contractor_runtime.artifacts import ArtifactAPIError
    from contractor_runtime.capabilities import CapabilitySnapshot
    from contractor_runtime.contracts import ToolsetRef, ToolsetSelection
    from contractor_runtime.digests import _agent_template_digest
    from contractor_runtime.toolsets.code_execution import CodeExecutionToolsetFactory
    from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory

    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _, spec, _, _ = await service_for(tmp_path, fixture)
            client = FakeArtifactClient()
            execution = CodeExecutionToolsetFactory()
            artifacts = RunArtifactsToolsetFactory(lambda *_: client)
            service._factories = replace(
                service._factories, toolsets={execution.ref: execution, artifacts.ref: artifacts}
            )
            service._capabilities = CapabilitySnapshot.create(
                runtimes=service._factories.worker_runtimes,
                toolsets={
                    execution.ref: execution.exported_tools,
                    artifacts.ref: artifacts.exported_tools,
                },
                sandbox_profiles=service._factories.sandbox_profiles,
                workspace=service._factories.workspace_provider.capability,
            )
            spec.agent_template.toolsets = [
                ToolsetSelection(
                    ref=ToolsetRef(toolsetId="run-artifacts", version="1"), tools=["write_artifact"]
                ),
                ToolsetSelection(
                    ref=ToolsetRef(toolsetId="code-execution", version="1"), tools=["exec_command"]
                ),
            ]
            spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
            await service.prepare(spec)
            context = service._context
            assert context.tools["exec_command"]._executor is context.execution.executor
            commands = Commands()
            commands.resume.set()
            commands.edit = lambda: (fixture.backend.entry.root / "src/main.py").write_text(
                "def generated():\n    return 42\n"
            )
            fixture.backend.commands = commands
            analyses, _ = await analysis_tools(context.project_workspace.reader_view(), tmp_path)
            try:
                assert (await analyses["search_def"]("generated"))["items"] == []
                assert (await context.tools["exec_command"]("generate"))["exitCode"] == 0
                found = await analyses["search_def"]("generated")
                assert [row["name"] for row in found["items"]] == ["generated"]
                content = await context.project_workspace.read_text("src/main.py")
                written = await context.tools["write_artifact"](
                    "builder", "report", "text/plain", base64.b64encode(content.encode()).decode()
                )
                assert written["artifact"]["revision"]
                # The existing client, not the sandbox, owns the write fence.
                write_artifact = context.tools["write_artifact"]
                client.write_error = ArtifactAPIError(409, "allocation_write_fenced", False)
                await service.finalize(finalization(spec))
                with pytest.raises(ArtifactAPIError, match="allocation_write_fenced"):
                    await write_artifact("builder", "report", "text/plain", "eA==")
            finally:
                for tool in analyses.values():
                    await tool.close()
                await service.finalize(finalization(spec))
                await service.release(release(spec))

    asyncio.run(scenario())
