from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.types import AgentCard
from a2a.utils.constants import TransportProtocol
from google.protobuf.json_format import ParseDict
from test_a2a_server import data_request, send

from contractor_runtime.allocation import AllocationService, WorkerState
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    AbortAllocationRequest,
    AllocationSpec,
    ReleaseAllocationRequest,
    StageContentRequest,
)
from contractor_runtime.contracts.tool_execution import ToolArgumentBinding
from contractor_runtime.digests import _agent_template_digest, verify_template_digests
from contractor_runtime.factories import WorkerBuildContext, built_in_factories
from contractor_runtime.server import create_app
from contractor_runtime.state import RuntimeState
from contractor_runtime.worker.tool_runtime import ToolWorkerRuntimeFactory
from contractor_runtime.workspace import AllocationWorkspace

ROOT = Path(__file__).parents[2]


def tool_spec():
    spec = AllocationSpec.model_validate_json(
        (ROOT / "api/testdata/v1alpha1/valid/allocation-spec-tool.json").read_text()
    )
    spec.lease_expires_at = datetime.now(UTC) + timedelta(minutes=10)
    return spec


def request(spec, *, target="http://fixture.invalid/?secret=canary"):
    return StageContentRequest(
        apiVersion="contractor/v1alpha1",
        subtaskId="0",
        objective="Scan",
        instructions="Scan",
        parameters={"target": target},
        artifacts={},
        resultArtifacts={"report": {"namespace": spec.namespace, "name": "report"}},
    )


class ArtifactStore:
    def __init__(self):
        self.values = {}
        self.fail_report = False
        self.ambiguous_start = False
        self.fail_terminal = False
        self.block_write = None
        self.write_started = asyncio.Event()
        self.writes = 0

    async def request(self, method, path, *, headers, body, max_response_bytes):
        parsed = urlsplit(path)
        namespace, name = map(unquote, parsed.path.split("/artifacts/")[1].split("/"))
        key = (namespace, name)
        current = self.values.get(key)
        if method == "GET":
            if current is None:
                return self.response(404, {"code": "not_found", "retryable": False})
            data, media_type, revision = current
            assert len(data) <= max_response_bytes
            return self.response(200, data, media_type, revision)
        assert method == "PUT"
        phase = "report" if name == "report" else "terminal" if current else "start"
        if phase == self.block_write:
            self.write_started.set()
            await asyncio.Event().wait()
        if name == "report" and self.fail_report:
            raise ArtifactTransportError("publication failed")
        if "if-none-match" in {key.lower(): value for key, value in headers.items()}:
            if current is not None:
                return self.response(412, {"code": "precondition_failed", "retryable": False})
        elif current is None or json.loads(headers["If-Match"]) != current[2]:
            return self.response(412, {"code": "precondition_failed", "retryable": False})
        if name.startswith("tool-invocation.") and current is not None and self.fail_terminal:
            raise ArtifactTransportError("terminal outcome is unknown")
        self.writes += 1
        revision = f"revision-{self.writes}"
        self.values[key] = (body, headers["Content-Type"], revision)
        if name.startswith("tool-invocation.") and current is None and self.ambiguous_start:
            raise ArtifactTransportError("start outcome is unknown")
        return self.response(
            201 if current is None else 200,
            {
                "apiVersion": "contractor/v1alpha1",
                "artifact": {"namespace": namespace, "name": name, "revision": revision},
                "mediaType": headers["Content-Type"],
                "size": len(body),
            },
            revision=revision,
        )

    @staticmethod
    def response(status, data, media_type="application/json", revision="revision-0"):
        payload = data if isinstance(data, bytes) else json.dumps(data).encode()
        return ArtifactHTTPResponse(
            status,
            {
                "content-type": media_type,
                "content-length": str(len(payload)),
                "etag": json.dumps(revision),
                "x-contractor-binding-created-at": "2026-09-19T00:00:00Z",
                "x-contractor-revision-created-at": "2026-09-19T00:00:00Z",
            },
            payload,
        )


class FixtureTool:
    name = "scan_nuclei"

    def __init__(self):
        self.calls = []
        self.result = {"status": "completed", "results": []}
        self.wait = False
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(self, url: str, rate_limit: int = 10) -> dict:
        self.calls.append((url, rate_limit))
        self.started.set()
        try:
            if self.wait:
                await asyncio.Event().wait()
            return self.result
        finally:
            if self.wait:
                self.cancelled.set()

    async def close(self):
        pass


async def worker(tmp_path, store, tool, *, spec=None):
    spec = spec or tool_spec()
    context = WorkerBuildContext(
        allocation_id=spec.allocation_id,
        run_id=spec.run_id,
        stage_execution_id=spec.stage_execution_id,
        logical_agent_name=spec.logical_agent_name,
        namespace=spec.namespace,
        worker_session_mode=spec.worker_session_mode,
        description=spec.agent_template.description,
        instruction="",
        card_version="1",
        model_policy=None,
        execution=spec.agent_template.execution,
        template_ref=spec.agent_template.ref,
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        tools={tool.name: tool},
        state=WorkerState(),
        a2a_base_url="https://runtime.invalid",
        runtime_settings=spec.runtime_settings,
    )
    factory = ToolWorkerRuntimeFactory(
        lambda allocation, settings: ArtifactClient(allocation, store)
    )
    return await factory.create(context)


def test_tool_worker_report_and_recreated_allocation_replay(tmp_path):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await runtime.invoke(request(spec))
        assert result.failure is None, result
        assert result.result.artifacts["report"].revision == "revision-2"
        assert await runtime.invoke(request(spec)) == result
        conflict = await runtime.invoke(request(spec, target="http://changed.invalid"))
        assert conflict.failure.code == "tool_input_conflict"
        assert len(tool.calls) == 1
        report = json.loads(store.values[(spec.namespace, "report")][0])
        assert report["observation"] == tool.result
        spec.allocation_id = "replacement-allocation"
        replacement = await worker(tmp_path, store, tool, spec=spec)
        replay = await replacement.invoke(request(spec))
        assert replay.result.artifacts == result.result.artifacts
        assert len(tool.calls) == 1
        state = await replacement.agent_state_snapshot()
        assert state.state.last_completed_invocation.metrics.model_calls == 0
        assert state.state.last_completed_invocation.metrics.tool_calls == 0
        conflict_worker = await worker(tmp_path, store, tool, spec=spec)
        assert (
            await conflict_worker.invoke(request(spec, target="http://changed.invalid"))
        ).failure.code == "tool_input_conflict"
        assert len(tool.calls) == 1
        receipts = [
            json.loads(data)
            for (namespace, name), (data, _, _) in store.values.items()
            if name.startswith("tool-invocation.")
        ]
        assert len(receipts) == 1 and "canary" not in repr(receipts)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["report", "terminal"])
def test_cancel_during_publication_finishes_state_without_rescan(tmp_path, phase):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        store.block_write = phase
        runtime = await worker(tmp_path, store, tool)
        task = asyncio.create_task(runtime.invoke(request(spec)))
        await asyncio.wait_for(store.write_started.wait(), 1)
        assert (await runtime.invoke(request(spec))).failure.code == "worker_busy"
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))
        with pytest.raises(asyncio.CancelledError):
            await task
        state = (await runtime.agent_state_snapshot()).state
        assert state.current_invocation is None
        assert state.last_completed_invocation.phase == "cancelled"
        assert (await runtime.invoke(request(spec))).failure.code == "worker_draining"
        store.block_write = None
        recreated = await worker(tmp_path, store, tool)
        assert (await recreated.invoke(request(spec))).failure.code == "tool_outcome_unknown"
        assert len(tool.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure,expected,calls",
    [
        ("ambiguous_start", "tool_outcome_unknown", 0),
        ("fail_report", "tool_report_failed", 1),
        ("fail_terminal", "tool_outcome_unknown", 1),
    ],
)
def test_artifact_failures_never_rescan(tmp_path, failure, expected, calls):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        setattr(store, failure, True)
        runtime = await worker(tmp_path, store, tool)
        first = await runtime.invoke(request(spec))
        assert first.failure.code == expected
        setattr(store, failure, False)
        recreated = await worker(tmp_path, store, tool)
        replay = await recreated.invoke(request(spec))
        assert replay.failure is not None
        assert len(tool.calls) == calls

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["deadline", "cancel", "abort"])
def test_tool_deadline_cancel_abort_and_replay(tmp_path, mode):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        tool.wait = True
        spec.agent_template.execution.timeout_seconds = 1
        runtime = await worker(tmp_path, store, tool, spec=spec)
        task = asyncio.create_task(runtime.invoke(request(spec)))
        await asyncio.wait_for(tool.started.wait(), 1)
        if mode == "cancel":
            runtime.cancel_active()
        elif mode == "abort":
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))
        if mode == "deadline":
            assert (await task).failure.code == "tool_timeout"
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert tool.cancelled.is_set()
        assert (await runtime.agent_state_snapshot()).state.current_invocation is None
        tool.wait = False
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(request(spec))).failure is not None
        assert len(tool.calls) == 1

    asyncio.run(scenario())


def test_input_types_sources_and_output_validation(tmp_path):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        bad = request(spec)
        bad.parameters = {}
        assert (await runtime.invoke(bad)).failure.code == "tool_input_invalid"
        assert not tool.calls and not store.values
        spec.agent_template.execution.arguments["rate_limit"] = ToolArgumentBinding(
            source="literal", value="10"
        )
        runtime = await worker(tmp_path, store, tool, spec=spec)
        assert (await runtime.invoke(request(spec))).failure.code == "tool_input_invalid"
        assert not tool.calls and not store.values
        spec.agent_template.execution.arguments.pop("rate_limit")
        tool.result = {"status": "completed", "invalid": float("nan")}
        runtime = await worker(tmp_path, store, tool, spec=spec)
        assert (await runtime.invoke(request(spec))).failure.code == "tool_output_invalid"
        assert len(tool.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["completed", "lease_loss"])
def test_real_allocation_and_a2a_use_scanner_without_model(tmp_path, monkeypatch, outcome):
    from test_scan_toolset import executable

    marker = tmp_path / "calls"
    pid_file = tmp_path / "pid"
    executable(
        tmp_path,
        "nuclei",
        "import os, sys, time\n"
        "if '-version' not in sys.argv:\n"
        f"    open({str(marker)!r}, 'a').write('scan\\n')\n"
        f"    open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        + ("    time.sleep(60)\n" if outcome == "lease_loss" else "")
        + "    print('{\"fixture\":true}')",
    )
    templates = tmp_path / "templates"
    templates.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("NUCLEI_TEMPLATES_DIR", str(templates))

    async def scenario():
        store, spec = ArtifactStore(), tool_spec()
        factories = built_in_factories(
            tmp_path / "work",
            artifact_client_factory=lambda allocation, settings: ArtifactClient(allocation, store),
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=("tool@1",),
            toolsets={"scan@1": frozenset({"scan_nuclei"})},
            sandbox_profiles=("local-workdir@1",),
        )
        state = RuntimeState(capabilities=capabilities)
        await state.mark_registered()
        service = AllocationService(
            factories=factories, state=state, a2a_base_url="https://runtime.example"
        )
        prepared = await service.prepare(spec)
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        card = ParseDict(prepared.worker_handle.agent_card, AgentCard())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as http:
            client = ClientFactory(
                ClientConfig(
                    streaming=False,
                    httpx_client=http,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC],
                )
            ).create(card)
            envelope = data_request(spec.allocation_id)
            ParseDict(
                request(spec).model_dump(by_alias=True, exclude_none=True),
                envelope.message.parts[0].data,
            )
            if outcome == "completed":
                result = await send(client, envelope)
                assert "failure" not in result, result
                assert result["result"]["artifacts"]["report"]["revision"] == "revision-2"
            else:
                runtime = service._context.worker
                invocation = asyncio.create_task(runtime.invoke(request(spec)))
                async with asyncio.timeout(2):
                    while not pid_file.exists() or not pid_file.read_text():
                        await asyncio.sleep(0.01)
                await service.expire_control_lease(2)
                with pytest.raises(asyncio.CancelledError):
                    await invocation
                with pytest.raises(ProcessLookupError):
                    os.kill(int(pid_file.read_text()), 0)
                assert await service.active_a2a_application(spec.allocation_id) is None
                assert (await runtime.invoke(request(spec))).failure.code == "worker_draining"
                assert (spec.namespace, "report") not in store.values
                assert all(
                    json.loads(value[0])["phase"] == "started" for value in store.values.values()
                )
            assert marker.read_text() == "scan\n"
            snapshot = (
                await runtime.agent_state_snapshot()
                if outcome == "lease_loss"
                else await service.agent_state_snapshot(spec.allocation_id)
            )
            assert snapshot.state.last_completed_invocation.metrics.model_calls == 0
            assert snapshot.state.last_completed_invocation.metrics.tool_calls == 1
            assert snapshot.state.metrics.counters.get("model_calls", 0) == 0
            assert snapshot.state.metrics.counters["tool_calls"] == 1
            await client.close()
        await service.abort(
            AbortAllocationRequest(
                apiVersion="contractor/v1alpha1",
                allocationId=spec.allocation_id,
                abortId="abort",
                reason={"code": "test_done", "message": "Done", "retryable": False},
                deadline=datetime.now(UTC) + timedelta(seconds=2),
            )
        )
        await service.release(
            ReleaseAllocationRequest(
                apiVersion="contractor/v1alpha1", allocationId=spec.allocation_id
            )
        )

    asyncio.run(scenario())


def test_tool_contract_digest_and_model_boundary():
    spec = tool_spec()
    verify_template_digests(spec.agent_template)
    assert _agent_template_digest(spec.agent_template) == spec.agent_template.ref.digest
    assert "modelPolicy" not in spec.model_dump(by_alias=True)
    assert "llmGatewayUrl" not in spec.runtime_settings.model_dump(by_alias=True)
    for path in [
        ("modelPolicy",),
        ("agentTemplate", "modelPolicy"),
        ("agentTemplate", "instructions"),
        ("agentTemplate", "summarizer"),
        ("runtimeSettings", "llmGatewayUrl"),
    ]:
        raw = spec.model_dump(by_alias=True)
        target = raw if len(path) == 1 else raw[path[0]]
        target[path[-1]] = None
        with pytest.raises(ValueError):
            AllocationSpec.model_validate(raw)
    for binding in [
        {"source": "parameter"},
        {"source": "parameter", "name": "target", "value": None},
        {"source": "literal", "name": "rate", "value": 10},
        {"source": "literal", "value": {}},
    ]:
        with pytest.raises(ValueError):
            ToolArgumentBinding.model_validate(binding)
