from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from fakes.discovery import FakeDiscoveryFactory
from fakes.mcp import MCPFixture, remote_tool
from test_allocation import NOW, make_spec, resign_template
from test_mcp_session import configuration, deadline

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.contracts import (
    API_VERSION,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    ToolsetRef,
    ToolsetSelection,
)
from contractor_runtime.factories import FactoryRegistry, StubADKWorkerRuntimeFactory
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.toolsets.mcp.errors import MCPCleanupError, MCPError
from contractor_runtime.toolsets.mcp.factory import MCPToolsetFactory
from contractor_runtime.toolsets.mcp.session import MCPSessionOwner
from contractor_runtime.workspace import LocalWorkdirFactory


def assert_no_sdk_tasks() -> None:
    for task in asyncio.all_tasks():
        if task is asyncio.current_task():
            continue
        for frame in task.get_stack():
            assert "/site-packages/mcp/" not in frame.f_code.co_filename


@pytest.mark.parametrize("phase", ["initialize", "tools/list", "tools/call"])
def test_cancellation_joins_sdk_tasks_in_every_network_phase(phase) -> None:
    async def scenario() -> None:
        async with MCPFixture() as server:
            owner = MCPSessionOwner(configuration(server.url), ["search"])
            if phase == "tools/call":
                await owner.prepare(deadline=deadline())
                server.block_method = phase
                operation = asyncio.create_task(
                    owner.call("search", {"query": "cancel"}, deadline=deadline())
                )
            else:
                server.block_method = phase
                operation = asyncio.create_task(owner.prepare(deadline=deadline()))
            await asyncio.wait_for(server.entered.wait(), 1)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
            await asyncio.gather(*(owner.close(deadline=deadline()) for _ in range(3)))
            assert owner.closed
            assert_no_sdk_tasks()
            assert len(server.calls) == (1 if phase == "tools/call" else 0)

    asyncio.run(scenario())


def test_accepted_call_with_lost_response_has_one_unknown_outcome_and_no_reconnect() -> None:
    async def scenario() -> None:
        async with MCPFixture() as server:
            owner = MCPSessionOwner(configuration(server.url), ["search"])
            await owner.prepare(deadline=deadline())
            server.drop_call = True
            with pytest.raises(MCPError, match="outcome_unknown"):
                await owner.call("search", {"query": "accepted-once"}, deadline=deadline())
            await owner.close(deadline=deadline())
            assert len(server.calls) == 1 and len(server.sessions) == 1
            with pytest.raises(MCPError, match="session_closed"):
                await owner.call("search", {"query": "never-replayed"}, deadline=deadline())
            assert owner.closed
            assert_no_sdk_tasks()

    asyncio.run(scenario())


def test_unconfirmed_local_transport_close_retains_owner_and_original_deadline() -> None:
    class SlowClose(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport(trust_env=False)
            self.allow_close = asyncio.Event()

        async def handle_async_request(self, request):
            return await self.inner.handle_async_request(request)

        async def aclose(self):
            await self.allow_close.wait()
            await self.inner.aclose()

    async def scenario() -> None:
        async with MCPFixture() as server:
            transport = SlowClose()
            owner = MCPSessionOwner(configuration(server.url), ["search"], transport=transport)
            await owner.prepare(deadline=deadline())
            with pytest.raises(MCPCleanupError):
                await owner.close(deadline=deadline(0.04))
            saved = owner._close_deadline
            await asyncio.sleep(0)
            with pytest.raises(MCPCleanupError):
                await owner.close(deadline=deadline(20))
            assert owner._close_deadline == saved and not owner.closed
            assert owner._client is not None and owner._settings is not None
            assert_no_sdk_tasks()
            # The test owns the injected failure. Drain its real pool without
            # changing the production owner's unconfirmed release outcome.
            transport.allow_close.set()
            await transport.inner.aclose()

    asyncio.run(scenario())


class OwnedTool:
    def __init__(self, name, owner):
        self.name, self.owner = name, owner

    async def prepare_allocation(self, *, deadline):
        await self.owner.prepare(deadline=deadline)

    async def close_before(self, *, deadline):
        await self.owner.close(deadline=deadline)

    async def close(self):
        await self.owner.close()


class OwnedFactory(FakeDiscoveryFactory):
    async def create_selected(self, *, selected, runtime_settings, private_bypass_hosts, **kwargs):
        self.owner = MCPSessionOwner(
            runtime_settings.toolsets["mcp@1"], selected, forbidden_hosts=private_bypass_hosts
        )
        assert self.owner._task is None  # Returning ownership performs no IO.
        self.created = {name: OwnedTool(name, self.owner) for name in selected}
        return self.created


class RecordingMCPFactory(MCPToolsetFactory):
    async def create_selected(self, **kwargs):
        self.created = await super().create_selected(**kwargs)
        self.owner = next(iter(self.created.values()))._owner
        assert self.owner._task is None
        return self.created


@pytest.mark.parametrize("operation", ["finalize", "lease", "cancel_prepare", "failed_prepare"])
def test_allocation_owns_session_before_prepare_and_closes_it_on_every_exit(
    tmp_path: Path, operation
) -> None:
    async def scenario() -> None:
        # Distinct loopback address from Runtime's forbidden private listener.
        async with MCPFixture(host="127.0.0.2") as server:
            factory = RecordingMCPFactory()
            registry = FactoryRegistry(
                worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
                sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
                toolsets={"mcp@1": factory},
            )
            state = RuntimeState(instance_id="mcp-lifecycle")
            await state.mark_registered()
            capabilities = await discover_capabilities(registry)
            exits = []
            service = AllocationService(
                state,
                registry,
                capabilities,
                a2a_base_url="https://runtime.example",
                now=lambda: NOW,
                force_exit=exits.append,
            )
            spec = make_spec()
            spec.agent_template.toolsets = [
                ToolsetSelection(
                    ref=ToolsetRef(toolset_id="mcp", version="1"), tools=["search", "fetch"]
                )
            ]
            spec.runtime_settings.toolsets = {"mcp@1": configuration(server.url)}
            resign_template(spec.agent_template)
            if operation == "cancel_prepare":
                server.block_method = "tools/list"
                prepare = asyncio.create_task(service.prepare(spec))
                await asyncio.wait_for(server.entered.wait(), 1)
                prepare.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await prepare
                assert (await state.snapshot()).process_state is ProcessState.IDLE
            elif operation == "failed_prepare":
                server.pages = [[remote_tool("search")]]
                with pytest.raises(AllocationError) as failure:
                    await service.prepare(spec)
                assert failure.value.code == "mcp_missing_tool"
                assert (await state.snapshot()).process_state is ProcessState.IDLE
            else:
                await service.prepare(spec)
                assert len(server.sessions) == 1
                assert (await service.snapshot()).tool_names == ("fetch", "search")
                if operation == "lease":
                    await service.expire_control_lease(1)
                    assert (await state.snapshot()).process_state is ProcessState.FENCED
                else:
                    await service.finalize(
                        FinalizeAllocationRequest(
                            apiVersion=API_VERSION,
                            allocationId=spec.allocation_id,
                            finalizationId="finish-mcp",
                            deadline=NOW + timedelta(seconds=2),
                        )
                    )
                await service.release(
                    ReleaseAllocationRequest(
                        apiVersion=API_VERSION, allocationId=spec.allocation_id
                    )
                )
                await service.confirm_release(spec.allocation_id)
            assert factory.owner.closed and not exits
            assert_no_sdk_tasks()

    asyncio.run(scenario())


def test_failed_prepare_retains_unconfirmed_tool_owner_and_never_confirms_release(tmp_path) -> None:
    class UnconfirmedTool(OwnedTool):
        async def close_before(self, *, deadline):
            await super().close_before(deadline=deadline)
            raise MCPCleanupError

    class UnconfirmedFactory(OwnedFactory):
        async def create_selected(self, **kwargs):
            await super().create_selected(**kwargs)
            self.created = {name: UnconfirmedTool(name, self.owner) for name in self.created}
            return self.created

    async def scenario() -> None:
        async with MCPFixture(host="127.0.0.2", pages=[[remote_tool("other")]]) as server:
            factory = UnconfirmedFactory()
            registry = FactoryRegistry(
                worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
                sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
                toolsets={"mcp@1": factory},
            )
            state = RuntimeState(instance_id="mcp-failed-cleanup")
            await state.mark_registered()
            exits = []
            service = AllocationService(
                state,
                registry,
                await discover_capabilities(registry),
                a2a_base_url="https://runtime.example",
                now=lambda: NOW,
                force_exit=exits.append,
            )
            spec = make_spec()
            spec.agent_template.toolsets = [
                ToolsetSelection(ref=ToolsetRef(toolset_id="mcp", version="1"), tools=["search"])
            ]
            spec.runtime_settings.toolsets = {"mcp@1": configuration(server.url)}
            resign_template(spec.agent_template)
            with pytest.raises(AllocationError):
                await service.prepare(spec)
            assert exits == [70] and (await state.snapshot()).process_state is ProcessState.FENCED
            assert service._context.tools["search"].owner is factory.owner
            with pytest.raises(AllocationError):
                await service.release(
                    ReleaseAllocationRequest(
                        apiVersion=API_VERSION, allocationId=spec.allocation_id
                    )
                )
            with pytest.raises(AllocationError):
                await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())
