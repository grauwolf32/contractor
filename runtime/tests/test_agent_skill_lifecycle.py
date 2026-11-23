from __future__ import annotations

import asyncio
import gc
import hashlib
import threading
import weakref
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock
from fakes.model import scripted_model, text_result, tool_call
from fakes.spec import allocation_spec
from filesystem_audit import FilesystemAudit
from test_adk_runtime import FakeArtifactClient
from test_adk_runtime import stage_request as worker_request
from test_agent_skill_toolset import FakeSkillClient, resolved_value, skill_package
from test_allocation import TrackedMetricsState, allocation_references, live_references

import contractor_runtime.agent_skills.runtime as skill_runtime
import contractor_runtime.worker.state as worker_state_module
from contractor_runtime.agent_skills import probe_native_agent_skills
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    TerminationError,
    WorkerSessionMode,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory


class ReadProxy:
    def __init__(self, owner, allocation_id):
        self.owner = owner
        self.__name__ = "read_artifact"
        self.__doc__ = f"Read exact input for {allocation_id}."

    def __getattr__(self, name):
        return getattr(self.owner, name)

    async def __call__(
        self,
        namespace: str,
        name: str,
        revision: str | None = None,
        offset: int = 0,
        length: int = 262144,
    ) -> dict:
        return await self.owner(namespace, name, revision, offset, length)


class UnhashableReadProxy(ReadProxy):
    __hash__ = None


def skill_graph_references(worker):
    owner = worker._agent_skills
    assert owner is not None and owner._adapter is not None
    # PreparedAgentSkills/DisclosureBudget are slotted and not weakrefable.
    # The owner always holds native_toolset, so its collection also proves no
    # owner survived. Adapter detachment and session deletion cover State paths.
    references = {
        "skill_toolset": weakref.ref(owner.native_toolset),
        "skill_adapter": weakref.ref(owner._adapter),
        "model": weakref.ref(worker._model),
    }
    for index, skill in enumerate(owner.skills):
        references[f"skill:{index}"] = weakref.ref(skill)
    for name, tool in owner.native_tools.items():
        references[f"native:{name}"] = weakref.ref(tool)
    for tool in owner._adapter._tools:
        references[f"disclosure:{tool.name}"] = weakref.ref(tool)
    return references


async def graph_service(tmp_path, monkeypatch, *, unhashable=False):
    monkeypatch.setattr(worker_state_module, "MetricsState", TrackedMetricsState)
    clients, built = [], []

    class Client(FakeArtifactClient):
        async def read_artifact(self, ref):
            if ref.namespace == "skills":
                return values[ref.name]
            return await super().read_artifact(ref)

    def client_factory(*_args):
        client = Client()
        clients.append(weakref.ref(client))
        return client

    def model_factory(_context):
        return scripted_model(
            [
                tool_call(
                    "read_artifact", {"namespace": "inputs", "name": "source"}, call_id="read"
                ),
                tool_call("load_skill", {"skill_name": "demo"}, call_id="activate"),
                tool_call(
                    "load_skill_resource",
                    {"skill_name": "demo", "file_path": "scripts/check.py"},
                    call_id="inert-script",
                ),
                text_result("Used this allocation's skill"),
            ]
        )

    class Factory(AdkWorkerRuntimeFactory):
        async def create(self, context):
            proxy = (UnhashableReadProxy if unhashable else ReadProxy)(
                context.tools["read_artifact"], context.allocation_id
            )
            worker = await super().create(replace(context, tools={"read_artifact": proxy}))
            references = skill_graph_references(worker)
            references["proxy"] = weakref.ref(proxy)
            for tool in worker._agent.tools:
                declaration = getattr(tool, "_get_declaration", None)
                if callable(declaration):
                    declaration()
            built.append(references)
            return worker

    base = built_in_factories(tmp_path, client_factory, model_factory)
    factories = replace(base, worker_runtimes={"adk@1": Factory(model_factory, client_factory)})
    capabilities = capabilities_for(factories)
    state = RuntimeState(instance_id="skill-graph", capabilities=capabilities)
    await state.mark_registered()
    service = AllocationService(
        state,
        factories,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=lambda _: None,
    )
    selected, value = resolved_value(skill_package(script=b"raise RuntimeError('inert')\n"))
    values = {"demo": value}
    return service, state, selected, values, clients, built


@pytest.mark.parametrize("unhashable", [False, True], ids=["hashable", "unhashable"])
@pytest.mark.parametrize("termination", ["finalize", "abort"])
def test_used_skill_graph_is_collected_and_next_allocation_is_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unhashable: bool, termination: str
) -> None:
    async def scenario() -> None:
        service, state, _selected, values, clients, built = await graph_service(
            tmp_path, monkeypatch, unhashable=unhashable
        )
        previous_session = None
        for index in range(2):
            pin, value = resolved_value(
                skill_package(
                    instructions=f"Independent guidance {index}",
                    script=b"raise RuntimeError('inert')\n",
                )
            )
            values["demo"] = value
            allocation_id = f"allocation-{index}"
            spec = allocation_spec(allocation_id=allocation_id, skills=[pin])
            spec.worker_session_mode = WorkerSessionMode.SHARED
            await service.prepare(spec)
            del spec
            context = service._context
            worker = context.worker
            owner = worker._agent_skills
            assert owner.disclosure.used == 0
            assert owner.skills[0].instructions.strip() == f"Independent guidance {index}"
            assert worker._session_lifecycle.shared_session_id is None
            references = allocation_references(service) | built[-1]
            declaration = worker._agent.tools[0]._get_declaration()
            assert declaration.description == f"Read exact input for {allocation_id}."
            completion = await worker.invoke(worker_request())
            assert completion.failure is None and completion.result is not None
            session_id = worker._session_lifecycle.shared_session_id
            assert session_id and session_id != previous_session
            previous_session = session_id
            session = await worker._session_service.get_session(
                app_name=worker._app_name, user_id=worker._user_id, session_id=session_id
            )
            assert session.state["_adk_activated_skill_contractor_worker"] == ["demo"]
            assert owner.disclosure.used > 0
            if termination == "finalize":
                await service.finalize(
                    FinalizeAllocationRequest(
                        apiVersion=API_VERSION,
                        allocationId=allocation_id,
                        finalizationId=f"final-{index}",
                        deadline=datetime.now(UTC) + timedelta(seconds=5),
                    )
                )
            else:
                await service.abort(
                    AbortAllocationRequest(
                        apiVersion=API_VERSION,
                        allocationId=allocation_id,
                        abortId=f"abort-{index}",
                        reason=TerminationError(
                            code="test_abort", message="Stop fixture", retryable=False
                        ),
                        deadline=datetime.now(UTC) + timedelta(seconds=5),
                    )
                )
            assert owner.skills == [] and owner.native_tools == {} and owner._adapter is None
            assert not owner.extraction_root.exists()
            assert (
                await worker._session_service.get_session(
                    app_name=worker._app_name, user_id=worker._user_id, session_id=session_id
                )
                is None
            )
            del context, worker, owner, completion, session, declaration
            await service.release(
                ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=allocation_id)
            )
            await service.confirm_release(allocation_id)
            assert live_references(references) == []
            gc.collect()
            assert all(reference() is None for reference in clients)
            assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


@pytest.mark.parametrize("unhashable", [False, True], ids=["hashable", "unhashable"])
def test_failed_prepare_after_skill_and_declaration_build_collects_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unhashable: bool
) -> None:
    async def scenario() -> None:
        service, state, pin, _, clients, built = await graph_service(
            tmp_path, monkeypatch, unhashable=unhashable
        )

        async def fail_commit(_allocation_id):
            raise RuntimeError("synthetic post-build failure")

        with monkeypatch.context() as patch:
            patch.setattr(state, "commit_allocation", fail_commit)
            with pytest.raises(AllocationError) as failed:
                await service.prepare(allocation_spec(skills=[pin]))
            assert failed.value.code == "allocation_preparation_failed"
            del failed
        assert await service.snapshot() is None
        assert list(tmp_path.iterdir()) == []
        assert live_references(built[0]) == []
        assert all(reference() is None for reference in clients)
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        spec = allocation_spec(allocation_id="allocation-recovered", skills=[pin])
        await service.prepare(spec)
        completion = await service._context.worker.invoke(worker_request())
        assert completion.failure is None
        await service.abort(
            AbortAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                abortId="recovered",
                reason=TerminationError(code="test_abort", message="End fixture", retryable=False),
                deadline=datetime.now(UTC) + timedelta(seconds=5),
            )
        )
        del completion
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)
        assert live_references(built[1]) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("component", ["runner", "plugin", "native"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["error", "cancelled"])
def test_interrupted_adk_skill_cleanup_keeps_the_slot_fenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str, cancelled: bool
) -> None:
    async def scenario() -> None:
        service, state, pin, _, _, _ = await graph_service(tmp_path, monkeypatch)
        exits = []
        service._force_exit = exits.append
        await service.prepare(allocation_spec(skills=[pin]))
        worker = service._context.worker
        owner = worker._agent_skills
        targets = {
            "runner": worker._runner,
            "plugin": worker._plugin,
            "native": owner.native_toolset,
        }
        started, resume = asyncio.Event(), asyncio.Event()
        calls = []
        closing_tasks = []

        async def interrupted_close():
            calls.append(component)
            closing_tasks.append(asyncio.current_task())
            started.set()
            await resume.wait()
            raise RuntimeError("private close failure")

        with monkeypatch.context() as patch:
            patch.setattr(targets[component], "close", interrupted_close)
            running = asyncio.create_task(
                service.finalize(
                    FinalizeAllocationRequest(
                        apiVersion=API_VERSION,
                        allocationId="allocation-1",
                        finalizationId="interrupted",
                        deadline=datetime.now(UTC) + timedelta(seconds=10),
                    )
                )
            )
            await asyncio.wait_for(started.wait(), 5)
            assert not running.done()
            assert service._context.workspace.path.is_dir()
            if cancelled:
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
                if component == "native":
                    # ADK joins its shielded ten-second toolset-close timer.
                    # The native owner is already held at resume.wait().
                    assert any(not task.done() for task in closing_tasks)
                    advance_clock(11)
            else:
                resume.set()
                with pytest.raises(AllocationError) as failed:
                    await running
                assert failed.value.code == "worker_stop_unconfirmed"
                assert "private close failure" not in failed.value.message
            await asyncio.gather(*closing_tasks, return_exceptions=True)
            assert calls and len(calls) <= 2
            assert all(task.done() for task in closing_tasks)
        assert exits == [70]
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert service._context.worker is not None
        with pytest.raises(AllocationError):
            await service.prepare(allocation_spec(allocation_id="forbidden-new", skills=[pin]))
        with pytest.raises(AllocationError):
            await service.release(
                ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId="allocation-1")
            )
        # Production requires process exit for an unconfirmed in-process stop.
        # Dispose this test's held owner after proving the persistent fence;
        # this is not a successful-release or weakref assertion.
        await owner.close()

    run_with_advanceable_clock(scenario())


def test_cancelled_release_joins_one_cleanup_before_skill_slot_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        service, state, pin, _, clients, built = await graph_service(tmp_path, monkeypatch)
        await service.prepare(allocation_spec(skills=[pin]))
        completion = await service._context.worker.invoke(worker_request())
        assert completion.failure is None
        del completion
        references = allocation_references(service) | built[-1]
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId="allocation-1",
                finalizationId="release-cancel",
                deadline=datetime.now(UTC) + timedelta(seconds=10),
            )
        )
        context = service._context
        original = context.sandbox.cleanup
        started, resume = asyncio.Event(), asyncio.Event()
        calls = []

        async def cleanup(workspace):
            calls.append(workspace.path)
            started.set()
            await resume.wait()
            await original(workspace)

        with monkeypatch.context() as patch:
            patch.setattr(context.sandbox, "cleanup", cleanup)
            request = ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId="allocation-1")
            running = asyncio.create_task(service.release(request))
            await asyncio.wait_for(started.wait(), 5)
            retained = context.release_cleanup_task
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert retained is context.release_cleanup_task and not retained.done()
            assert context.workspace.path.is_dir()
            assert (await state.snapshot()).process_state is ProcessState.FENCED
            with pytest.raises(AllocationError):
                await service.prepare(allocation_spec(allocation_id="too-early", skills=[pin]))
            retry = asyncio.create_task(service.release(request))
            await asyncio.sleep(0)
            assert context.release_cleanup_task is retained and len(calls) == 1
            resume.set()
            await asyncio.wait_for(retry, 5)
        assert retained.done() and context.release_prepared
        assert not context.workspace.path.exists() and len(calls) == 1
        await service.confirm_release("allocation-1")
        del context, retained, retry, running
        assert live_references(references) == []
        assert all(reference() is None for reference in clients)
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        next_id = "allocation-after-cancel"
        await service.prepare(allocation_spec(allocation_id=next_id, skills=[pin]))
        assert service._context.worker._agent_skills.disclosure.used == 0
        assert service._context.worker._session_lifecycle.shared_session_id is None
        completion = await service._context.worker.invoke(worker_request())
        assert completion.failure is None
        del completion
        await service.abort(
            AbortAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=next_id,
                abortId="after-cancel",
                reason=TerminationError(code="test_abort", message="End fixture", retryable=False),
                deadline=datetime.now(UTC) + timedelta(seconds=5),
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=next_id)
        )
        await service.confirm_release(next_id)
        assert live_references(built[-1]) == []
        assert all(reference() is None for reference in clients)

    asyncio.run(scenario())


@pytest.mark.parametrize("termination", ["finalize", "abort", "lease"])
def test_skill_state_is_erased_before_slot_can_return_idle(
    tmp_path: Path, termination: str
) -> None:
    async def scenario() -> None:
        service, state, spec = await skilled_service(tmp_path)
        await service.prepare(spec)
        snapshot = await service.snapshot()
        assert snapshot is not None
        workspace = Path(snapshot.workspace)
        extraction = workspace / ".agent-skills"
        assert extraction.is_dir()
        assert (extraction / "demo" / "scripts" / "check.py").is_file()

        if termination == "finalize":
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalization-1",
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
        elif termination == "abort":
            await service.abort(
                AbortAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    abortId="abort-1",
                    reason=TerminationError(
                        code="test_abort", message="Abort test allocation", retryable=False
                    ),
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
        else:
            await service.expire_control_lease(2)

        assert not extraction.exists()
        service_snapshot = await service.snapshot()
        assert service_snapshot is not None
        assert not service_snapshot.has_worker
        assert (await state.snapshot()).process_state in {
            ProcessState.DRAINING,
            ProcessState.FENCED,
        }

        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert not workspace.exists()

    asyncio.run(scenario())


def test_skill_preparation_failure_maps_exact_code_and_rolls_back(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = skill_package()
        selected, value = resolved_value(payload)
        selected = selected.model_copy(update={"package_digest": "sha256:" + "0" * 64})
        client = FakeSkillClient({"demo": value})
        factories = built_in_factories(
            tmp_path,
            lambda _allocation, _settings: client,  # type: ignore[arg-type]
            lambda _: scripted_model([]),
        )
        capabilities = capabilities_for(factories)
        state = RuntimeState(instance_id="runtime-skills", capabilities=capabilities)
        await state.mark_registered()
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
            force_exit=lambda _: None,
        )
        spec = allocation_spec(skills=[selected])

        with pytest.raises(AllocationError) as captured:
            await service.prepare(spec)
        assert captured.value.code == "skill_digest_mismatch"
        assert captured.value.retryable is False
        assert await service.snapshot() is None
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_skill_deletion_failure_fences_and_forces_process_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        exit_codes: list[int] = []
        service, state, spec = await skilled_service(tmp_path, exit_codes=exit_codes)
        await service.prepare(spec)
        context = service._context
        assert context is not None
        runtime = context.worker
        owner = runtime._agent_skills
        assert owner is not None
        extraction = owner.extraction_root

        def fail_remove(_: Path) -> None:
            raise OSError("synthetic private deletion failure")

        owner._remove_tree = fail_remove
        with pytest.raises(AllocationError) as captured:
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalization-1",
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
        assert captured.value.code == "worker_stop_unconfirmed"
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert exit_codes == [70]
        assert extraction.exists()
        assert "synthetic private deletion failure" not in captured.value.message

    asyncio.run(scenario())


async def skilled_service(
    tmp_path: Path, *, exit_codes: list[int] | None = None
) -> tuple[AllocationService, RuntimeState, object]:
    payload = skill_package(script=b"raise RuntimeError('must not execute')\n")
    selected, value = resolved_value(payload)
    assert selected.package_digest == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    client = FakeSkillClient({"demo": value})
    factories = built_in_factories(
        tmp_path,
        lambda _allocation, _settings: client,  # type: ignore[arg-type]
        lambda _: scripted_model([]),
    )
    capabilities = capabilities_for(factories)
    state = RuntimeState(instance_id="runtime-skills", capabilities=capabilities)
    await state.mark_registered()
    service = AllocationService(
        state,
        factories,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=(exit_codes.append if exit_codes is not None else lambda _: None),
    )
    return service, state, allocation_spec(skills=[selected])


PROBE_DIRECTORY_PREFIX = "contractor-skill-probe-"


class ProbeFilesystemTrace:
    """Record where the startup probe touches the filesystem."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, block: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.events: list[tuple[str, int, bool]] = []
        self.probe_root: Path | None = None
        load = skill_runtime.load_skill_from_dir
        cleanup = skill_runtime._ProbeSkillDirectory.cleanup

        def traced_load(directory: Path) -> object:
            skill = load(directory)
            self.probe_root = directory.parent
            self.events.append(("loaded", threading.get_ident(), self.probe_root.is_dir()))
            self.started.set()
            if block and not self.release.wait(10):
                raise RuntimeError("the test never released the probe")
            return skill

        def traced_cleanup(owner: skill_runtime._ProbeSkillDirectory) -> None:
            exists = self.probe_root is not None and self.probe_root.is_dir()
            self.events.append(("cleanup", threading.get_ident(), exists))
            cleanup(owner)

        monkeypatch.setattr(skill_runtime, "load_skill_from_dir", traced_load)
        monkeypatch.setattr(skill_runtime._ProbeSkillDirectory, "cleanup", traced_cleanup)


def test_native_skill_probe_writes_and_removes_its_skill_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = ProbeFilesystemTrace(monkeypatch)

    async def scenario() -> None:
        loop_thread = threading.get_ident()
        with FilesystemAudit(PROBE_DIRECTORY_PREFIX) as audit:
            assert await probe_native_agent_skills() is True
        assert [event[0] for event in trace.events] == ["loaded", "cleanup"]
        assert all(thread != loop_thread for _, thread, _ in trace.events)
        assert all(exists for _, _, exists in trace.events)
        root = trace.probe_root
        assert root is not None and not root.exists()
        skill = root / "probe-skill"
        assert {
            ("tempfile.mkdtemp", str(root)),
            ("os.mkdir", str(skill)),
            ("open", str(skill / "SKILL.md")),
            ("open", str(skill / "scripts" / "check.cjs")),
            ("shutil.rmtree", str(root)),
        } <= set(audit.events())
        assert audit.on_thread(loop_thread) == []

    asyncio.run(scenario())


def test_cancelled_native_skill_probe_joins_its_thread_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = ProbeFilesystemTrace(monkeypatch, block=True)

    async def scenario() -> None:
        loop_thread = threading.get_ident()
        with FilesystemAudit(PROBE_DIRECTORY_PREFIX) as audit:
            await probe_and_cancel(loop_thread)
        assert audit.events() and audit.on_thread(loop_thread) == []

    async def probe_and_cancel(loop_thread: int) -> None:
        task = asyncio.create_task(probe_native_agent_skills())
        assert await asyncio.to_thread(trace.started.wait, 10)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        # The written Skill stays until the joined thread has returned.
        assert not task.done()
        assert [event[0] for event in trace.events] == ["loaded"]
        assert trace.probe_root is not None and trace.probe_root.is_dir()
        trace.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [event[0] for event in trace.events] == ["loaded", "cleanup"]
        assert all(thread != loop_thread for _, thread, _ in trace.events)
        assert trace.events[1][2] is True
        assert not trace.probe_root.exists()

    asyncio.run(scenario())


def capabilities_for(factories: object) -> CapabilitySnapshot:
    return CapabilitySnapshot.create(
        runtimes=factories.worker_runtimes,  # type: ignore[attr-defined]
        toolsets={
            ref: factory.exported_tools
            for ref, factory in factories.toolsets.items()  # type: ignore[attr-defined]
        },
        sandbox_profiles=factories.sandbox_profiles,  # type: ignore[attr-defined]
        runtime_adapters=factories.runtime_adapters,  # type: ignore[attr-defined]
    )
