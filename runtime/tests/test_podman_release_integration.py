"""Additional real release evidence; skips are forbidden by the explicit gate."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_podman_execution_integration import allocation, deadline

from contractor_runtime.sandbox.contracts import ExecutionRequest
from contractor_runtime.sandbox.podman.engine import PodmanEngine
from contractor_runtime.sandbox.podman.io import LocalPodmanCLI, local_engine_environment
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.settings import PodmanSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE") != "1",
    reason="explicit real Podman release gate",
)


@pytest.mark.parametrize("termination", ["finalize", "abort", "lease"])
def test_real_input_required_reuse_and_concurrent_release(tmp_path, monkeypatch, termination):
    import httpx
    from a2a.client import ClientConfig, ClientFactory
    from a2a.server.tasks import TaskUpdater
    from a2a.types import AgentCard, Task, TaskState, TaskStatus
    from a2a.utils.constants import TransportProtocol
    from fakes.model import scripted_model, text_result
    from fakes.podman_workflow import ArtifactPeer, sample_spec
    from google.protobuf.json_format import ParseDict
    from test_a2a_server import data_request, send
    from test_podman_allocation import finalization, release
    from test_projectfs_storage import local_settings

    from contractor_runtime.a2a_server import ContractorAgentExecutor
    from contractor_runtime.allocation import AllocationService
    from contractor_runtime.capabilities import discover_capabilities
    from contractor_runtime.contracts import API_VERSION, AbortAllocationRequest, TerminationError
    from contractor_runtime.factories import built_in_factories
    from contractor_runtime.server import create_app
    from contractor_runtime.state import ProcessState, RuntimeState

    execute = ContractorAgentExecutor.execute

    async def waiting(self, context, event_queue):
        # Inject only the SDK waiting state; resume uses production ADK execution.
        if context.message.message_id == "ask":
            await event_queue.enqueue_event(
                Task(
                    id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
                )
            )
            await TaskUpdater(event_queue, context.task_id, context.context_id).requires_input()
            return
        await execute(self, context, event_queue)

    monkeypatch.setattr(ContractorAgentExecutor, "execute", waiting)

    async def scenario():
        image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
        assert image, "preinstalled digest-pinned test image required"
        lifecycle = PodmanLifecycle(
            PodmanSettings(enabled=True, image=image, owner="reuse-gate-" + uuid.uuid4().hex)
        )
        lifecycle.bind_health(lambda: time.monotonic() + 60, lambda: None)
        peer = ArtifactPeer()
        model = scripted_model([text_result("first"), text_result("second")])
        try:
            factories = built_in_factories(
                tmp_path / "scratch",
                artifact_client_factory=lambda *_: peer,
                model_factory=lambda _: model,
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=lifecycle,
            )
            capabilities = await discover_capabilities(factories)
            assert "podman@1" in capabilities.sandbox_profiles
            state = RuntimeState(instance_id="reuse-gate", capabilities=capabilities)
            await state.mark_registered()
            exits = []
            service = AllocationService(
                state,
                factories,
                capabilities,
                a2a_base_url="https://runtime.example",
                force_exit=exits.append,
            )
            spec, _ = sample_spec(peer)
            prepared = await service.prepare(spec)
            context = service._context
            identity = lifecycle._entry.identity
            root = Path(context.project_workspace.storage.root)
            assert (
                await context.execution.executor.execute(
                    ExecutionRequest("printf retained > retained"), deadline=deadline()
                )
            ).exit_code == 0
            application = create_app(state, allocation_service=service, require_verified_peer=False)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="https://runtime.example"
            ) as http_client:
                client = ClientFactory(
                    ClientConfig(
                        streaming=False,
                        httpx_client=http_client,
                        supported_protocol_bindings=[TransportProtocol.JSONRPC],
                    )
                ).create(ParseDict(prepared.worker_handle.agent_card, AgentCard()))
                try:
                    responses = [
                        response
                        async for response in client.send_message(
                            data_request(spec.allocation_id, message_id="ask")
                        )
                    ]
                    task = responses[-1].task
                    assert task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
                    assert lifecycle._entry.identity == identity
                    resumed = data_request(spec.allocation_id, message_id="resume")
                    resumed.message.task_id = task.id
                    resumed.message.context_id = task.context_id
                    assert (await send(client, resumed))["result"]["result"] == "first"
                    assert (
                        await send(client, data_request(spec.allocation_id, message_id="next"))
                    )["result"]["result"] == "second"
                    assert await service.prepare(spec) == prepared
                    assert lifecycle._entry.identity == identity
                    result = await context.execution.executor.execute(
                        ExecutionRequest("cat retained"), deadline=deadline()
                    )
                    assert result.exit_code == 0 and result.stdout == "retained"
                finally:
                    await client.close()
            if termination == "finalize":
                await service.finalize(finalization(spec))
            elif termination == "abort":
                await service.abort(
                    AbortAllocationRequest(
                        apiVersion=API_VERSION,
                        allocationId=spec.allocation_id,
                        abortId="release-gate",
                        deadline=datetime.now(UTC) + timedelta(seconds=10),
                        reason=TerminationError(
                            code="cancelled", message="release gate", retryable=False
                        ),
                    )
                )
            else:
                await service.expire_control_lease(10)
            await asyncio.gather(service.release(release(spec)), service.release(release(spec)))
            assert context.execution.removed and context.execution.stopped.is_set()
            assert not root.exists() and not context.workspace.path.exists()
            assert lifecycle._entry is None and not exits
            await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is ProcessState.IDLE
        finally:
            await lifecycle.close(deadline=time.monotonic() + 30)

    asyncio.run(scenario())


def test_real_command_edit_snapshot_ordering_and_cancellation(tmp_path):
    async def scenario():
        async with allocation(tmp_path) as (handle, session, root):
            command = asyncio.create_task(
                handle.executor.execute(
                    ExecutionRequest(
                        "printf started > started; sleep .5; printf command > src/result"
                    ),
                    deadline=deadline(),
                )
            )
            until = time.monotonic() + 5
            while not (root / "started").exists():
                assert time.monotonic() < until and not command.done()
                await asyncio.sleep(0.01)
            editing = asyncio.create_task(session.write_text("src/result", "edited"))
            snapshot = asyncio.create_task(session.snapshot())
            pulses = 0
            for _ in range(10):
                assert not editing.done() and not snapshot.done()
                pulses += 1
                await asyncio.sleep(0.01)
            assert pulses > 5 and (await command).exit_code == 0
            await editing
            await snapshot
            assert await session.read_text("src/result") == "edited"
            assert (
                await handle.executor.execute(
                    ExecutionRequest("mv src/result src/moved; rm src/a.txt; printf new > src/new"),
                    deadline=deadline(),
                )
            ).exit_code == 0
            current = await session.snapshot()
            paths = {item.path for item in current.files}
            assert "src/result" not in paths and "src/a.txt" not in paths
            assert {"src/moved", "src/new"} <= paths
            command = asyncio.create_task(
                handle.executor.execute(
                    ExecutionRequest("printf started > cancelling; sleep 20; touch survived"),
                    deadline=deadline(),
                )
            )
            until = time.monotonic() + 5
            while not (root / "cancelling").exists():
                assert time.monotonic() < until
                await asyncio.sleep(0.01)
            command.cancel()
            with pytest.raises(asyncio.CancelledError):
                await command
            await handle.remove(deadline=deadline())
            assert handle.removed and handle.stopped.is_set() and not (root / "survived").exists()

    asyncio.run(scenario())


def test_real_restart_recovers_lost_create_but_preserves_unrelated_container(tmp_path):
    async def scenario():
        image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
        assert image, "preinstalled digest-pinned test image required"
        settings = PodmanSettings(enabled=True, image=image, owner="lost-gate-" + uuid.uuid4().hex)
        root = tmp_path / "run_workdir"
        root.mkdir()
        (root / "keep").write_text("retain until confirmed cleanup")
        other_root = tmp_path / "unrelated" / "run_workdir"
        other_root.mkdir(parents=True)
        (other_root / "keep").write_text("unrelated")
        other = PodmanEngine(
            PodmanSettings(enabled=True, image=image, owner="other-gate-" + uuid.uuid4().hex)
        )
        successor = PodmanLifecycle(settings)
        process = None
        await other.open(deadline=time.monotonic() + 20)
        try:
            identity = await other.create("unrelated", other_root, deadline=time.monotonic() + 20)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).parent / "fakes/podman_lost_create.py"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=local_engine_environment(),
                start_new_session=True,
            )
            await asyncio.wait_for(
                process.communicate(
                    json.dumps({"settings": asdict(settings), "root": str(root)}).encode() + b"\n"
                ),
                30,
            )
            assert process.returncode == 0
            cli = LocalPodmanCLI()
            args = (
                "ps",
                "-a",
                "--filter",
                f"label=io.contractor.sandbox.owner={settings.owner}",
                "--format=json",
            )
            found = await cli.run(args, deadline=time.monotonic() + 5)
            assert len(json.loads(found.stdout)) == 1
            await successor.recover(deadline=time.monotonic() + 20)
            found = await cli.run(args, deadline=time.monotonic() + 5)
            assert json.loads(found.stdout) == []
            assert await other.inspect(identity, deadline=time.monotonic() + 5) is not None
            assert (other_root / "keep").read_text() == "unrelated"
            assert (root / "keep").exists()  # recovery never erases bind content
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            await successor.close(deadline=time.monotonic() + 20)
            for identity in await other.discover(deadline=time.monotonic() + 20):
                await other.remove(identity, deadline=time.monotonic() + 20)
            await other.close(deadline=time.monotonic() + 20)

    asyncio.run(scenario())
