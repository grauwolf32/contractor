from __future__ import annotations

import asyncio
import gc
import weakref

import pytest
from test_allocation import finalize_and_release, make_service, make_spec
from test_archive_cache import ArchiveTransport, client, make_cache, source

from contractor_runtime.allocation import AllocationError
from contractor_runtime.archive_cache import read_archive
from contractor_runtime.state import ProcessState


@pytest.mark.parametrize("failure", ["cleanup-failed", "reply-lost"])
def test_release_pilot_replays_preserve_fenced_identity_and_replacement(
    tmp_path, runtime_capabilities, monkeypatch, failure
):
    from contractor_runtime.contracts import API_VERSION, ReleaseAllocationRequest

    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities, allocation_slots=2)
        old, peer, replacement = "pilot-old", "pilot-peer", "pilot-new"
        try:
            await service.prepare(make_spec(allocation_id=old))
            await service.prepare(make_spec(allocation_id=peer))
            owner = service._allocations[old]
            cleanup = owner._cleanup.release
            calls = 0

            async def failing_once(context, deadline):
                nonlocal calls
                calls += 1
                if failure == "cleanup-failed" and calls == 1:
                    raise OSError("injected local cleanup failure")
                await cleanup(context, deadline)

            monkeypatch.setattr(owner._cleanup, "release", failing_once)
            if failure == "cleanup-failed":
                with pytest.raises(AllocationError) as error:
                    await finalize_and_release(service, old)
                assert error.value.code == "allocation_cleanup_failed"
            else:
                # The peer loses this successful response; no heartbeat release
                # acknowledgement has arrived at the Runtime yet.
                await finalize_and_release(service, old)
            assert (await state.snapshot(old)).process_state is ProcessState.FENCED
            assert old in service._allocations
            assert (await state.snapshot(peer)).process_state is ProcessState.ALLOCATED
            with pytest.raises(AllocationError, match="capacity"):
                await service.prepare(make_spec(allocation_id=replacement))
            request = ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=old)
            await service.release(request)
            await service.release(request)
            assert calls == (2 if failure == "cleanup-failed" else 1)
            assert (await state.snapshot(old)).process_state is ProcessState.FENCED
            await service.confirm_release(old)
            await service.prepare(make_spec(allocation_id=replacement))
            new_owner = service._allocations[replacement]
            await service.release(request)
            await service.confirm_release(old)
            assert service._allocations[replacement] is new_owner
            assert (await state.snapshot(replacement)).process_state is ProcessState.ALLOCATED
            assert (await state.snapshot(peer)).process_state is ProcessState.ALLOCATED
            assert await service.active_a2a_application(replacement) is not None
        finally:
            await service.close(30)
        assert not service._allocations
        assert not (await state.snapshot()).allocations

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_first", [False, True])
def test_retired_prepare_cannot_consume_a_replacements_reservation(
    tmp_path, runtime_capabilities, monkeypatch, cancel_first
):
    from contractor_runtime.allocation.service import Allocation

    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities, allocation_slots=4)
        sandbox = service._factories.sandbox_profiles["local-workdir@1"]
        prepare_sandbox, prepare_owner = sandbox.prepare, Allocation.prepare
        first_entered, retry_entered, replacement_entered = [asyncio.Event() for _ in range(3)]
        first_resume, retry_resume, replacement_resume = [asyncio.Event() for _ in range(3)]
        owners, jobs = [], []
        builds = 0

        async def admitted(owner, spec):
            owners.append(owner)
            if len(owners) == 2:
                retry_entered.set()
                await retry_resume.wait()
            return await prepare_owner(owner, spec)

        async def build():
            nonlocal builds
            builds += 1
            if builds == 1:
                first_entered.set()
                await first_resume.wait()
                raise OSError("temporary preparation failure")
            if builds == 2:
                replacement_entered.set()
                await replacement_resume.wait()
            return await prepare_sandbox()

        monkeypatch.setattr(Allocation, "prepare", admitted)
        monkeypatch.setattr(sandbox, "prepare", build)
        try:
            jobs.append(asyncio.create_task(service.prepare(make_spec())))
            await first_entered.wait()
            jobs.append(asyncio.create_task(service.prepare(make_spec())))
            await retry_entered.wait()
            if cancel_first:
                jobs[0].cancel()
            else:
                first_resume.set()
            await asyncio.gather(jobs[0], return_exceptions=True)
            jobs.append(asyncio.create_task(service.prepare(make_spec())))
            await replacement_entered.wait()
            retry_resume.set()
            retry = (await asyncio.gather(jobs[1], return_exceptions=True))[0]
            replacement_resume.set()
            replacement = (await asyncio.gather(jobs[2], return_exceptions=True))[0]
            assert isinstance(retry, AllocationError) and retry.code == "allocation_conflict"
            assert not isinstance(replacement, BaseException)
            assert owners[0] is owners[1] and owners[2] is not owners[0]
            assert owners[0]._context is None
            assert service._allocations == {"allocation-1": owners[2]}
            assert (await state.snapshot()).allocations == ("allocation-1",)
            assert await service.active_a2a_application("allocation-1") is not None
        finally:
            first_resume.set()
            retry_resume.set()
            replacement_resume.set()
            await asyncio.gather(*jobs, return_exceptions=True)
            # The negative control can orphan an owner; retain it for test cleanup.
            for owner in owners:
                if owner._context is not None:
                    service._allocations[owner._allocation_id] = owner
            await service.close(30)
        assert not (await state.snapshot()).allocations

    asyncio.run(scenario())


def test_four_allocations_count_prepare_and_cleanup_without_blocking_neighbors(
    tmp_path, runtime_capabilities, monkeypatch
):
    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities, allocation_slots=4)
        sandbox = service._factories.sandbox_profiles["local-workdir@1"]
        prepare = sandbox.prepare
        entered, resume = asyncio.Event(), asyncio.Event()
        first = True

        async def blocked():
            nonlocal first
            if first:
                first = False
                entered.set()
                await resume.wait()
            return await prepare()

        monkeypatch.setattr(sandbox, "prepare", blocked)
        preparing = asyncio.create_task(service.prepare(make_spec(allocation_id="allocation-0")))
        await entered.wait()
        try:
            specs = [make_spec(allocation_id=f"allocation-{i}") for i in range(1, 4)]
            await asyncio.gather(*(service.prepare(spec) for spec in specs))
            assert len((await state.snapshot()).allocations) == 4
            assert len((await state.heartbeat(1, 0)).allocations) == 3
            with pytest.raises(AllocationError, match="capacity"):
                await service.prepare(make_spec(allocation_id="allocation-fifth"))
            assert await service.active_a2a_application("allocation-2") is not None
            released = service._allocations["allocation-1"]
            worker_state = weakref.ref(released._context.worker_state)
            await finalize_and_release(service, "allocation-1")
            await service.confirm_release("allocation-1")
            del released
            gc.collect()
            assert worker_state() is None
            assert (await state.snapshot("allocation-2")).process_state is ProcessState.ALLOCATED
            await service.prepare(make_spec(allocation_id="allocation-fifth"))
            await service.confirm_release("allocation-1")
            assert len(service._allocations) == 4
            assert service._allocations["allocation-fifth"]._context.worker is not None
        finally:
            resume.set()
            await preparing
            await service.close(30)
        assert not service._allocations
        assert not (await state.snapshot()).allocations
        assert not list(tmp_path.iterdir())

    asyncio.run(scenario())


def test_cache_four_readers_authorize_each_grant_and_keep_mutable_workspaces_private(tmp_path):
    from test_projectfs_zip import archive, settings

    from contractor_runtime.contracts import AllocationWorkspaceSource, AllocationWorkspaceSpec
    from contractor_runtime.projectfs import LocalWorkspaceProvider, hydrate_workspace

    async def scenario():
        root = tmp_path / "workspaces"
        cache = make_cache(root)
        transport = ArchiveTransport(archive({"source.txt": b"immutable\n"}))
        transport.gate = asyncio.Event()
        provider = LocalWorkspaceProvider(settings("local", root))
        readers = [client(cache, transport, f"allocation-{i}") for i in range(4)]

        async def hydrate(index):
            return await hydrate_workspace(
                provider=provider,
                spec=AllocationWorkspaceSpec(
                    mode="direct",
                    sources=[
                        AllocationWorkspaceSource(
                            artifact=source(f"binding-allocation-{index}"), target=""
                        )
                    ],
                ),
                artifact_reader=readers[index],
                allocation_id=f"allocation-{index}",
                timeout_seconds=10,
            )

        jobs = [asyncio.create_task(hydrate(index)) for index in range(4)]
        await transport.started.wait()
        while transport.metadata_calls < 4:
            await asyncio.sleep(0)
        transport.gate.set()
        workspaces = await asyncio.gather(*jobs)
        try:
            assert transport.payload_calls == 1 and transport.metadata_calls == 4
            assert len({workspace.storage.root for workspace in workspaces}) == 4
            from pathlib import Path

            paths = [
                Path(workspace.storage.root) / "run_workdir" / "source.txt"
                for workspace in workspaces
            ]
            paths[0].write_text("private edit")
            assert all(path.read_bytes() == b"immutable\n" for path in paths[1:])
            await workspaces[0].close()
            await provider.cleanup(workspaces[0].storage)
            assert all(path.exists() for path in paths[1:])
            assert (await read_archive(readers[3], source())).data == transport.payload
            transport.denied = 403
            from contractor_runtime.artifacts import ArtifactAPIError

            with pytest.raises(ArtifactAPIError):
                await read_archive(readers[2], source())
            assert transport.payload_calls == 1
        finally:
            for workspace in workspaces[1:]:
                await workspace.close()
                await provider.cleanup(workspace.storage)
            await cache.close()

    asyncio.run(scenario())


def test_shared_runtime_slot_contracts():
    import json
    from pathlib import Path

    from contractor_runtime.contracts import (
        AgentHeartbeat,
        AgentRegistration,
        HeartbeatResponse,
        PrivateProtocolDecodeError,
        decode_private,
    )

    root = Path(__file__).parents[2] / "api/testdata/v1alpha1"
    models = {
        "AgentRegistration": (AgentRegistration, "agent-registration.json"),
        "AgentHeartbeat": (AgentHeartbeat, "agent-heartbeat.json"),
        "HeartbeatResponse": (HeartbeatResponse, "heartbeat-response.json"),
    }
    cases = json.loads((root / "runtime-slots-cases.json").read_text())
    assert cases
    for case in cases:
        model, name = models[case["type"]]
        value = json.loads((root / "valid" / name).read_text())
        value.update(case["patch"])
        try:
            decode_private(model, json.dumps(value).encode())
        except PrivateProtocolDecodeError:
            assert not case["valid"], case["name"]
        else:
            assert case["valid"], case["name"]


def test_neighboring_adk_tools_survive_release_and_callable_cache_clear(
    tmp_path, runtime_capabilities
):
    from fakes.model import scripted_model, text_result, tool_call
    from test_adk_runtime import stage_request
    from test_allocation import TrackedArtifactClients

    from contractor_runtime.worker.adk_caches import _TOOL_CALLABLE_CACHES

    async def scenario():
        models = {}
        clients = TrackedArtifactClients()

        def model(context):
            value = scripted_model(
                [
                    tool_call(
                        "read_artifact",
                        {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                        call_id=f"read-{context.allocation_id}",
                    ),
                    text_result(context.allocation_id),
                ],
                block=context.allocation_id != "allocation-0",
            )
            models[context.allocation_id] = value
            return value

        _, service = await make_service(
            tmp_path,
            runtime_capabilities,
            allocation_slots=4,
            model_factory=model,
            artifact_client_factory=clients,
        )
        jobs = []
        try:
            for index in range(4):
                await service.prepare(
                    make_spec(
                        allocation_id=f"allocation-{index}",
                        tools=["read_artifact"],
                    )
                )
            for owner in service._allocations.values():
                jobs.append(asyncio.create_task(owner._context.worker.invoke(stage_request())))
            await asyncio.gather(*(value.started.wait() for value in models.values()))
            assert (await jobs[0]).failure is None
            assert all(cache.cache_info().currsize for cache in _TOOL_CALLABLE_CACHES)
            await finalize_and_release(service, "allocation-0")
            await service.confirm_release("allocation-0")
            assert all(not cache.cache_info().currsize for cache in _TOOL_CALLABLE_CACHES)
            for index in range(1, 4):
                models[f"allocation-{index}"].release()
            for index, result in enumerate(await asyncio.gather(*jobs[1:]), 1):
                assert result.failure is None and result.result.result == f"allocation-{index}"
                requests = models[f"allocation-{index}"].requests
                assert "read_artifact" in requests[0]["toolNames"]
                assert len(requests) >= 2  # The declared tool actually ran before completion.
        finally:
            for value in models.values():
                value.release()
            await asyncio.gather(*jobs, return_exceptions=True)
            await service.close(30)
        assert not service._allocations

    asyncio.run(scenario())
