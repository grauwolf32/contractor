from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.podman_lifecycle import owner
from fakes.spec import allocation_spec
from test_projectfs_storage import local_settings
from test_projectfs_zip import archive, workspace_inputs

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    SandboxProfileRef,
    TerminationError,
)
from contractor_runtime.digests import _agent_template_digest
from contractor_runtime.factories import (
    FactoryRegistry,
    PodmanWorkdirFactory,
    RunArtifactsToolsetFactory,
    StubADKWorkerRuntimeFactory,
    StubWorkerRuntime,
)
from contractor_runtime.projectfs import LocalWorkspaceProvider
from contractor_runtime.sandbox.contracts import SandboxContractError
from contractor_runtime.state import ProcessState, RuntimeState


async def service_for(tmp_path, fixture, *, fail_worker=False, runtime_factory=None):
    events = []
    source, reader = workspace_inputs([("source", "", archive({"src/main.py": b"ready\n"}))])

    class Provider(LocalWorkspaceProvider):
        async def create(self, allocation_id):
            assert fixture.backend.recovered
            events.append("hydrate")
            return await super().create(allocation_id)

        async def cleanup(self, storage):
            assert not fixture.cli.containers
            events.append("files")
            await super().cleanup(storage)

    class Worker(StubWorkerRuntime):
        async def finalize(self, deadline):
            events.append("artifacts")
            assert fixture.backend.entry.rejected or fixture.lifecycle._entry.rejected
            await super().finalize(deadline)

    class Runtime(StubADKWorkerRuntimeFactory):
        async def create(self, context):
            assert fixture.backend.entry.prepared
            root = fixture.backend.entry.root
            assert (root / "src/main.py").read_text() == "ready\n"
            events.append("worker")
            if fail_worker:
                raise RuntimeError("private failure")
            return Worker(context)

    runtime = runtime_factory or Runtime()
    sandbox = PodmanWorkdirFactory(tmp_path / "scratch")
    provider = Provider(
        local_settings(tmp_path / "project"),
        before_initialize=lambda: fixture.lifecycle.prepare_root(tmp_path / "project"),
    )
    artifacts = RunArtifactsToolsetFactory(lambda *_: reader)
    factories = FactoryRegistry(
        worker_runtimes={runtime.ref: runtime},
        toolsets={artifacts.ref: artifacts},
        sandbox_profiles={sandbox.ref: sandbox},
        workspace_provider=provider,
        artifact_client_factory=lambda *_: reader,
        execution_lifecycle=fixture.lifecycle,
    )
    capabilities = CapabilitySnapshot.create(
        runtimes=factories.worker_runtimes,
        toolsets={artifacts.ref: artifacts.exported_tools},
        sandbox_profiles=factories.sandbox_profiles,
        workspace=provider.capability,
    )
    state = RuntimeState(instance_id="podman-test")
    await state.mark_registered()
    exits = []
    service = AllocationService(
        state,
        factories,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=exits.append,
    )
    spec = allocation_spec()
    spec.workspace = source
    spec.agent_template.sandbox_profile = SandboxProfileRef(sandboxProfileId="podman", version="1")
    spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
    return service, state, spec, events, exits


def finalization(spec):
    return FinalizeAllocationRequest(
        apiVersion=API_VERSION,
        allocationId=spec.allocation_id,
        finalizationId="finalize",
        deadline=datetime.now(UTC) + timedelta(seconds=5),
    )


def release(spec):
    return ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)


def test_hydration_one_container_idempotency_finalize_and_authoritative_release(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, state, spec, events, exits = await service_for(tmp_path, fixture)
            first = await service.prepare(spec)
            assert await service.prepare(spec) == first
            assert fixture.cli.sequence == 1 and events == ["hydrate", "worker"]
            changed = spec.model_copy(deep=True)
            changed.run_metadata_labels["changed"] = "yes"
            with pytest.raises(AllocationError):
                await service.prepare(changed)
            context = service._context
            root = Path(context.project_workspace.storage.root)
            scratch = context.workspace.path
            response = await service.finalize(finalization(spec))
            assert await service.finalize(finalization(spec)) == response
            assert root.exists() and scratch.exists()
            assert all(not item["State"]["Running"] for item in fixture.cli.containers.values())
            await service.release(release(spec))
            assert not root.exists() and not scratch.exists() and not fixture.cli.containers
            assert events == ["hydrate", "worker", "artifacts", "files"]
            assert (await state.snapshot()).process_state is not ProcessState.IDLE
            await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is ProcessState.IDLE
            assert not exits

    asyncio.run(scenario())


@pytest.mark.parametrize("termination", ["abort", "lease"])
def test_abort_and_lease_remove_container_before_files(tmp_path, termination):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, state, spec, events, exits = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            if termination == "abort":
                await service.abort(
                    AbortAllocationRequest(
                        apiVersion=API_VERSION,
                        allocationId=spec.allocation_id,
                        abortId="abort",
                        deadline=datetime.now(UTC) + timedelta(seconds=5),
                        reason=TerminationError(
                            code="cancelled", message="cancelled", retryable=False
                        ),
                    )
                )
            else:
                await service.expire_control_lease(5)
            assert not fixture.cli.containers and events[-1] == "files"
            assert (await state.snapshot()).process_state is not ProcessState.IDLE
            await service.release(release(spec))
            await service.confirm_release(spec.allocation_id)
            assert not exits

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["start", "guardian", "worker"])
def test_partial_prepare_removes_container_before_hydrated_workspace(tmp_path, failure):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, state, spec, events, exits = await service_for(
                tmp_path, fixture, fail_worker=failure == "worker"
            )
            fixture.cli.noop_start = failure == "start"
            fixture.attach_fail = failure == "guardian"
            with pytest.raises(AllocationError):
                await service.prepare(spec)
            assert not fixture.cli.containers and events[-1] == "files"
            assert await service.snapshot() is None
            assert (await state.snapshot()).process_state is ProcessState.IDLE
            assert not exits

    asyncio.run(scenario())


def test_uncertain_partial_prepare_retains_context_and_fences_confirmation(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, state, spec, events, exits = await service_for(
                tmp_path, fixture, fail_worker=True
            )
            fixture.cli.noop_remove = True
            with pytest.raises(AllocationError):
                await service.prepare(spec)
            assert exits == [70] and fixture.cli.containers and "files" not in events
            assert service._context is not None and service._context.prepare_response is None
            assert Path(service._context.project_workspace.storage.root).exists()
            with pytest.raises(AllocationError):
                await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is not ProcessState.IDLE
            fixture.cli.noop_remove = False
            await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is ProcessState.IDLE
            assert events[-1] == "files"

    asyncio.run(scenario())


def test_cancelled_prepare_joins_late_create_before_removal(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _state, spec, events, exits = await service_for(tmp_path, fixture)
            fixture.cli.block_create = True
            preparing = asyncio.create_task(service.prepare(spec))
            await asyncio.wait_for(fixture.cli.entered.wait(), 3)
            preparing.cancel()
            await asyncio.sleep(0.02)
            assert not preparing.done() and "files" not in events
            assert fixture.cli.containers
            fixture.cli.resume.set()
            with pytest.raises(asyncio.CancelledError):
                await preparing
            assert "worker" not in events and events[-1] == "files"
            assert not any(call[0] == "start" for call in fixture.cli.calls)
            assert fixture.cli.sequence == 1 and not fixture.cli.containers and not exits

    asyncio.run(scenario())


def test_absent_uncertain_create_fences_until_exact_late_container_is_removed(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, state, spec, events, exits = await service_for(tmp_path, fixture)
            fixture.cli.absent_create = True
            with pytest.raises(AllocationError):
                await service.prepare(spec)
            assert exits == [70] and "files" not in events
            with pytest.raises(AllocationError):
                await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is not ProcessState.IDLE
            calls = [call for call in fixture.cli.calls if call[0] == "create"]
            assert len(calls) == 1
            # The original engine operation's exact label/name becomes visible
            # later. This is not a second create issued by lifecycle code.
            labels = dict(
                item.removeprefix("--label=").split("=", 1)
                for item in calls[0]
                if item.startswith("--label=")
            )
            name = next(
                item.removeprefix("--name=") for item in calls[0] if item.startswith("--name=")
            )
            container_id = "f" * 64
            fixture.cli.containers[container_id] = {
                "Id": container_id,
                "Name": name,
                "Config": {"Labels": labels},
                "State": {"Status": "created", "Running": False},
            }
            await service.confirm_release(spec.allocation_id)
            assert not fixture.cli.containers and events[-1] == "files"
            assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_cancelled_release_retains_one_cleanup_task(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _state, spec, events, exits = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            await service.finalize(finalization(spec))
            original = fixture.cli.run
            entered, resume = asyncio.Event(), asyncio.Event()

            async def blocked(args, *, deadline):
                if args[0] == "rm":
                    entered.set()
                    await resume.wait()
                return await original(args, deadline=deadline)

            fixture.cli.run = blocked
            releasing = asyncio.create_task(service.release(release(spec)))
            await asyncio.wait_for(entered.wait(), 3)
            releasing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await releasing
            cleanup = service._context.release_cleanup_task
            assert not cleanup.done() and "files" not in events
            retry = asyncio.create_task(service.release(release(spec)))
            await asyncio.sleep(0.02)
            assert service._context.release_cleanup_task is cleanup
            resume.set()
            await retry
            assert sum(call[0] == "rm" for call in fixture.cli.calls) == 1
            assert events[-1] == "files" and not exits
            await service.confirm_release(spec.allocation_id)

    asyncio.run(scenario())


def test_unconfirmed_stop_preserves_files_and_cannot_return_terminal_success(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, state, spec, events, exits = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            fixture.cli.noop_stop = True
            with pytest.raises(AllocationError) as failure:
                await service.finalize(finalization(spec))
            assert failure.value.code == "sandbox_cleanup_unconfirmed"
            assert exits == [70] and "files" not in events
            assert service._context.terminal_response is None
            assert (await state.snapshot()).process_state is not ProcessState.IDLE
            fixture.cli.noop_stop = False
            await service.confirm_release(spec.allocation_id)
            assert events[-1] == "files"

    asyncio.run(scenario())


def test_expired_prepare_deadline_rejects_work_without_restarting_container(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _state, spec, _events, _exits = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            handle = fixture.lifecycle._entry
            with pytest.raises(SandboxContractError):
                await handle.prepare(deadline=datetime.now(UTC) - timedelta(seconds=1))
            assert handle.rejected and fixture.cli.sequence == 1
            await service.expire_control_lease(5)

    asyncio.run(scenario())


def test_a2a_input_required_and_sequential_calls_keep_same_container(tmp_path, monkeypatch):
    # The current ADK executor completes requests directly. Inject only the
    # SDK's INPUT_REQUIRED event to test lifecycle independence of that state;
    # the resumed and following requests use the real production executor.
    import httpx
    from a2a.client import ClientConfig, ClientFactory
    from a2a.server.tasks import TaskUpdater
    from a2a.types import AgentCard, Task, TaskState, TaskStatus
    from a2a.utils.constants import TransportProtocol
    from fakes.model import scripted_model, text_result
    from google.protobuf.json_format import ParseDict
    from test_a2a_server import data_request, send

    from contractor_runtime.a2a_server import ContractorAgentExecutor
    from contractor_runtime.server import create_app
    from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory

    execute = ContractorAgentExecutor.execute

    async def waiting(self, context, event_queue):
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
        async with owner(tmp_path) as fixture:
            model = scripted_model([text_result("first"), text_result("second")])
            service, state, spec, _events, exits = await service_for(
                tmp_path,
                fixture,
                runtime_factory=AdkWorkerRuntimeFactory(model_factory=lambda _: model),
            )
            prepared = await service.prepare(spec)
            identity = fixture.lifecycle._entry.identity
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
                    assert fixture.lifecycle._entry.identity == identity
                    resumed = data_request(spec.allocation_id, message_id="resume")
                    resumed.message.task_id = task.id
                    resumed.message.context_id = task.context_id
                    assert (await send(client, resumed))["result"]["result"] == "first"
                    assert (
                        await send(client, data_request(spec.allocation_id, message_id="next"))
                    )["result"]["result"] == "second"
                    assert await service.prepare(spec) == prepared
                    assert (
                        fixture.lifecycle._entry.identity == identity and fixture.cli.sequence == 1
                    )
                finally:
                    await client.close()
            await service.finalize(finalization(spec))
            await service.release(release(spec))
            await service.confirm_release(spec.allocation_id)
            assert not exits

    asyncio.run(scenario())
