from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.types import AgentCard
from a2a.utils.constants import TransportProtocol
from google.protobuf.json_format import ParseDict
from target_policy_fixtures import SCAN_TEST_POLICY
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
    ArtifactRef,
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
        self.revisions = {}
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
            exact = parse_qs(parsed.query).get("revision", [None])[0]
            if exact is not None and current is not None and exact != current[2]:
                current = self.revisions.get((*key, exact))
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
        self.revisions[(*key, revision)] = self.values[key]
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


def request_with_targets(spec):
    value = request(spec)
    value.result_artifacts["targets"] = ArtifactRef(namespace=spec.namespace, name="targets")
    return value


class PublishingTool(FixtureTool):
    def __init__(self, store, spec):
        super().__init__()
        self.client = ArtifactClient(spec.allocation_id, store)
        self.namespace = spec.namespace

    async def __call__(self, url: str, rate_limit: int = 10) -> dict:
        await super().__call__(url, rate_limit)
        written = await self.client.write_artifact(
            ArtifactRef(namespace=self.namespace, name="targets"),
            data=b"https://fixture.invalid/discovered\n",
            media_type="text/plain",
            expected_revision=None,
        )
        ref = written.artifact.model_dump(by_alias=True)
        return {"status": "completed", "targetsArtifact": ref, "artifacts": {"targets": ref}}


def test_tool_additional_artifact_publication_exact_replay_and_binding_conflict(tmp_path):
    async def scenario():
        store, spec = ArtifactStore(), tool_spec()
        tool = PublishingTool(store, spec)
        stage_request = request_with_targets(spec)
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await runtime.invoke(stage_request)
        assert result.failure is None, result
        assert set(result.result.artifacts) == {"report", "targets"}
        target_ref = result.result.artifacts["targets"]
        assert target_ref.revision == "revision-2"
        report = json.loads(store.values[(spec.namespace, "report")][0])
        assert report["observation"]["targetsArtifact"] == target_ref.model_dump(by_alias=True)
        assert await runtime.invoke(stage_request) == result

        # The receipt identifies the originally published revision even after a
        # later binding write; replay must not substitute that mutable latest.
        await tool.client.write_artifact(
            ArtifactRef(namespace=spec.namespace, name="targets"),
            data=b"https://fixture.invalid/newer\n",
            media_type="text/plain",
            expected_revision=target_ref.revision,
        )
        spec.allocation_id = "replacement-allocation"
        replacement = await worker(tmp_path, store, tool, spec=spec)
        replay = await replacement.invoke(stage_request)
        assert replay.failure is None, replay
        assert replay.result.artifacts == result.result.artifacts
        assert len(tool.calls) == 1
        state = (await replacement.agent_state_snapshot()).state
        assert state.last_completed_invocation.metrics.tool_calls == 0
        assert state.last_completed_invocation.metrics.model_calls == 0

        changed = request_with_targets(spec)
        changed.result_artifacts["targets"].name = "other-targets"
        assert (await runtime.invoke(changed)).failure.code == "tool_input_conflict"
        replacement = await worker(tmp_path, store, tool, spec=spec)
        assert (await replacement.invoke(changed)).failure.code == "tool_input_conflict"
        assert len(tool.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "variant",
    ["undeclared", "primary", "foreign", "wrong_name", "mutable", "missing", "malformed"],
)
def test_tool_additional_output_ref_validation(tmp_path, variant):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        ref = {"namespace": spec.namespace, "name": "targets", "revision": "forged-revision"}
        refs = {"targets": ref}
        if variant == "undeclared":
            refs = {"other": ref}
        elif variant == "primary":
            refs = {"report": {**ref, "name": "report"}}
        elif variant == "foreign":
            ref["namespace"] = "another-worker"
        elif variant == "wrong_name":
            ref["name"] = "another-output"
        elif variant == "mutable":
            ref.pop("revision")
        elif variant == "missing":
            refs = {}
        elif variant == "malformed":
            refs = [ref]
        tool.result = {"status": "completed", "artifacts": refs}
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await runtime.invoke(request_with_targets(spec))
        assert result.failure.code == "tool_output_invalid"
        assert (spec.namespace, "report") not in store.values
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(request_with_targets(spec))).failure.code == (
            "tool_output_invalid"
        )
        assert len(tool.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("variant", ["foreign", "receipt", "memory", "alias", "duplicate", "limit"])
def test_tool_additional_binding_rejected_before_launch(tmp_path, variant):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        value = request_with_targets(spec)
        binding = value.result_artifacts["targets"]
        if variant == "foreign":
            binding.namespace = "another-worker"
        elif variant == "receipt":
            binding.name = "tool-invocation.collision"
        elif variant == "memory":
            binding.name = "memory.collision"
        elif variant == "alias":
            binding.name = "report"
        elif variant == "duplicate":
            value.result_artifacts["alias"] = binding.model_copy()
        else:
            for index in range(16):
                value.result_artifacts[f"extra{index}"] = ArtifactRef(
                    namespace=spec.namespace, name=f"extra{index}"
                )
        runtime = await worker(tmp_path, store, tool, spec=spec)
        assert (await runtime.invoke(value)).failure.code == "tool_input_invalid"
        assert not tool.calls and not store.values

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["absent", "revision", "read", "report", "report_conflict"])
def test_tool_additional_artifact_publication_failures_never_rescan(tmp_path, failure):
    class ReadFailureStore(ArtifactStore):
        async def request(self, method, path, **kwargs):
            if failure == "read" and method == "GET" and urlsplit(path).path.endswith("/targets"):
                raise ArtifactTransportError("private-publication-failure-canary")
            return await super().request(method, path, **kwargs)

    async def scenario():
        store, spec = ReadFailureStore(), tool_spec()
        tool = PublishingTool(store, spec)
        if failure in {"absent", "revision"}:
            tool = FixtureTool()
            if failure == "revision":
                store.values[(spec.namespace, "targets")] = (b"target", "text/plain", "actual")
            tool.result = {
                "status": "completed",
                "artifacts": {
                    "targets": {
                        "namespace": spec.namespace,
                        "name": "targets",
                        "revision": "forged",
                    }
                },
            }
        if failure == "report":
            store.fail_report = True
        elif failure == "report_conflict":
            store.values[(spec.namespace, "report")] = (b"existing", "text/plain", "original")
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await runtime.invoke(request_with_targets(spec))
        assert result.failure.code == "tool_report_failed"
        assert "canary" not in repr(result)
        if failure == "report_conflict":
            assert store.values[(spec.namespace, "report")][0] == b"existing"
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(request_with_targets(spec))).failure.code == (
            "tool_report_failed"
        )
        assert len(tool.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("tamper", ["mutable", "undeclared", "missing_output", "missing_artifact"])
def test_tool_additional_receipt_validation_never_rescans(tmp_path, tamper):
    async def scenario():
        store, spec = ArtifactStore(), tool_spec()
        tool = PublishingTool(store, spec)
        runtime = await worker(tmp_path, store, tool, spec=spec)
        assert (await runtime.invoke(request_with_targets(spec))).failure is None
        if tamper == "missing_artifact":
            del store.values[(spec.namespace, "targets")]
        else:
            for key, (data, media, revision) in store.values.items():
                if not key[1].startswith("tool-invocation."):
                    continue
                receipt = json.loads(data)
                if tamper == "mutable":
                    receipt["artifacts"]["targets"].pop("revision")
                elif tamper == "undeclared":
                    receipt["artifacts"]["forged"] = receipt["artifacts"].pop("targets")
                else:
                    receipt.pop("artifacts")
                store.values[key] = (json.dumps(receipt).encode(), media, revision)
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(request_with_targets(spec))).failure.code == (
            "tool_outcome_unknown"
        )
        assert len(tool.calls) == 1

    asyncio.run(scenario())


def test_failed_tool_with_no_additional_outputs_still_publishes_diagnostics(tmp_path):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        tool.result = {"status": "failed", "errorCode": "no_discovered_targets", "artifacts": {}}
        runtime = await worker(tmp_path, store, tool, spec=spec)
        assert (await runtime.invoke(request_with_targets(spec))).failure.code == (
            "tool_execution_failed"
        )
        report = json.loads(store.values[(spec.namespace, "report")][0])
        assert report["observation"] == tool.result
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(request_with_targets(spec))).failure.code == (
            "tool_execution_failed"
        )
        assert len(tool.calls) == 1

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


def test_failed_state_admission_completes_without_running_tool(tmp_path):
    from contractor_runtime.worker.state import WorkerStateError

    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool)

        async def refuse(**kwargs):
            raise WorkerStateError("a Worker invocation is already active")

        runtime._state.begin_invocation = refuse
        refused = await runtime.invoke(request(spec))
        assert refused.failure.code == "tool_outcome_unknown"
        assert tool.calls == [] and store.values == {}
        state = (await runtime.agent_state_snapshot()).state
        assert state.current_invocation is None and state.last_completed_invocation is None

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


def test_report_failure_diagnostics_preserve_scanner_success_and_replay(tmp_path):
    secret = "report-transport-secret-canary"

    class FailingReportStore(ArtifactStore):
        async def request(self, method, path, **kwargs):
            if method == "PUT" and urlsplit(path).path.endswith("/report"):
                raise ArtifactTransportError(secret)
            return await super().request(method, path, **kwargs)

    class MeasuredTool(FixtureTool):
        async def __call__(self, url: str, rate_limit: int = 10) -> dict:
            result = await super().__call__(url, rate_limit)
            self.metrics.record_tool_call(self.name, arguments={}, result={"status": "completed"})
            return result

    async def scenario():
        store, tool, spec = FailingReportStore(), MeasuredTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        tool.metrics = runtime._state.metrics
        completion = await runtime.invoke(request(spec))
        assert completion.failure.code == "tool_report_failed"
        report = runtime._state.metrics.build_report(
            report_id="worker-failed-report", duration_ms=1
        )
        assert report.metrics.tools[tool.name].calls == 1
        assert report.metrics.tools[tool.name].succeeded == 1
        assert report.metrics.tools[tool.name].failed == 0
        assert [(error.code, error.message, error.retryable) for error in report.errors] == [
            ("tool_report_failed", "Tool report publication failed", False)
        ]
        assert secret not in report.model_dump_json()
        assert "fixture.invalid" not in report.model_dump_json()
        assert (spec.namespace, "report") not in store.values

        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(request(spec))).failure.code == "tool_report_failed"
        replay = recreated._state.metrics.build_report(
            report_id="worker-replayed-report", duration_ms=1
        )
        assert [error.code for error in replay.errors] == ["tool_report_failed"]
        assert replay.metrics.tools == {}
        assert len(tool.calls) == 1

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
            # Another A2A Task's request cannot cancel this invocation.
            runtime.cancel_active(asyncio.current_task())
            await asyncio.sleep(0)
            assert not task.done() and not tool.cancelled.is_set()
            runtime.cancel_active(task)
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
            target_policy=SCAN_TEST_POLICY,
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


def sqlmap_spec():
    spec = tool_spec()
    template = spec.agent_template
    template.ref.template_id = "sqlmap-scan"
    template.description = "Check one prepared HTTP request for SQL injection."
    template.toolsets[0].tools = ["scan_sqlmap"]
    template.execution.tool = "scan_sqlmap"
    template.execution.arguments = {
        "request_ref": ToolArgumentBinding(source="artifact", name="request")
    }
    template.ref.digest = _agent_template_digest(template)
    return AllocationSpec.model_validate(spec.model_dump(by_alias=True, exclude_none=True))


def prepared_request(spec):
    return StageContentRequest(
        apiVersion="contractor/v1alpha1",
        subtaskId="0",
        objective="Check the prepared request",
        instructions="Check the prepared request",
        parameters={},
        artifacts={"request": {"namespace": "inputs", "name": "request", "revision": "request-1"}},
        resultArtifacts={"report": {"namespace": spec.namespace, "name": "report"}},
    )


async def scanner_allocation(tmp_path, store, spec):
    factories = built_in_factories(
        tmp_path / "work",
        artifact_client_factory=lambda allocation, settings: ArtifactClient(allocation, store),
        target_policy=SCAN_TEST_POLICY,
    )
    state = RuntimeState(
        capabilities=CapabilitySnapshot.create(
            runtimes=("tool@1",),
            toolsets={"scan@1": frozenset({spec.agent_template.execution.tool})},
            sandbox_profiles=("local-workdir@1",),
        )
    )
    await state.mark_registered()
    service = AllocationService(
        factories=factories, state=state, a2a_base_url="https://runtime.example"
    )
    await service.prepare(spec)
    return service


async def release_scanner_allocation(service, spec):
    await service.abort(
        AbortAllocationRequest(
            apiVersion="contractor/v1alpha1",
            allocationId=spec.allocation_id,
            abortId="done",
            reason={"code": "test_done", "message": "Done", "retryable": False},
            deadline=datetime.now(UTC) + timedelta(seconds=2),
        )
    )
    await service.release(
        ReleaseAllocationRequest(apiVersion="contractor/v1alpha1", allocationId=spec.allocation_id)
    )


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_sqlmap_prepared_request_allocation_report_and_replay(tmp_path, monkeypatch, method):
    from test_scan_toolset import executable

    marker = tmp_path / "calls"
    body = '{"id":7,"token":"body-canary-ключ"}'
    request_line = f"{method} https://target.invalid:8443/items%2Fsearch?id=7 HTTP/1.1\r\n"
    executable(
        tmp_path,
        "sqlmap",
        "import pathlib, stat, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('fixture-sqlmap')\n"
        "    sys.exit(0)\n"
        "args = sys.argv[1:]\n"
        f"assert '--method={method}' in args\n"
        "assert '--encoding=utf-8' in args\n"
        "assert '--skip-waf' in args\n"
        "assert args[args.index('-p') + 1] == 'id'\n"
        "request = pathlib.Path(args[args.index('-r') + 1])\n"
        "assert stat.S_IMODE(request.stat().st_mode) == 0o600\n"
        "assert stat.S_IMODE(request.parent.stat().st_mode) == 0o700\n"
        "raw = request.read_bytes()\n"
        f"assert raw.startswith({request_line.encode()!r})\n"
        "assert b'Host: target.invalid:8443\\r\\n' in raw\n"
        "assert b'Authorization: Bearer header-canary\\r\\n' in raw\n"
        "assert b'Cookie: session=cookie-canary\\r\\n' in raw\n"
        "assert b'Content-Type: application/json\\r\\n' in raw\n"
        f"assert raw.partition(b'\\r\\n\\r\\n')[2] == {body.encode()!r}\n"
        f"with open({str(marker)!r}, 'a') as calls:\n"
        "    calls.write('scan\\n')\n"
        "print(raw.decode())\n"
        "print(raw.decode(), file=sys.stderr)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        store, spec = ArtifactStore(), sqlmap_spec()
        store.values[("inputs", "request")] = (
            json.dumps(
                {
                    "schemaVersion": 1,
                    "method": method,
                    "url": "https://target.invalid:8443/items%2Fsearch?id=7",
                    "headers": [
                        {"name": "Authorization", "value": "Bearer header-canary"},
                        {"name": "Cookie", "value": "session=cookie-canary"},
                        {"name": "Content-Type", "value": "application/json"},
                    ],
                    "body": body,
                    "testParameters": ["id"],
                }
            ).encode(),
            "application/json",
            "request-1",
        )
        service = await scanner_allocation(tmp_path, store, spec)
        try:
            stage_request = prepared_request(spec)
            result = await service._context.worker.invoke(stage_request)
            assert result.failure is None, result
            assert marker.read_text() == "scan\n"
            report = json.loads(store.values[(spec.namespace, "report")][0])
            request_ref = stage_request.artifacts["request"].model_dump(by_alias=True)
            assert report["inputArtifacts"] == {"request": request_ref}
            assert report["observation"]["requestArtifact"] == request_ref
            assert report["observation"]["status"] == "completed"
            assert report["observation"]["diagnosticsRedacted"] is True
            snapshot = await service.agent_state_snapshot(spec.allocation_id)
            assert snapshot.state.last_completed_invocation.metrics.model_calls == 0
            assert snapshot.state.last_completed_invocation.metrics.tool_calls == 1
            assert "canary" not in repr(snapshot)
            assert "canary" not in repr(report)
            assert not list((tmp_path / "work").rglob("scan_sqlmap-*"))
        finally:
            await release_scanner_allocation(service, spec)

        spec.allocation_id = "replacement-allocation"
        replacement = await scanner_allocation(tmp_path, store, spec)
        try:
            replay = await replacement._context.worker.invoke(stage_request)
            assert replay.failure is None
            assert replay.result.artifacts == result.result.artifacts
            assert marker.read_text() == "scan\n"
            state = await replacement.agent_state_snapshot(spec.allocation_id)
            assert state.state.last_completed_invocation.metrics.model_calls == 0
            assert state.state.last_completed_invocation.metrics.tool_calls == 0
        finally:
            await release_scanner_allocation(replacement, spec)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [403, 404])
def test_sqlmap_unreadable_request_never_launches(tmp_path, monkeypatch, status):
    from test_scan_toolset import executable

    marker = tmp_path / "calls"
    executable(
        tmp_path,
        "sqlmap",
        "import pathlib, sys\n"
        "if '--version' not in sys.argv:\n"
        f"    pathlib.Path({str(marker)!r}).write_text('unexpected scan')\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    class UnreadableRequestStore(ArtifactStore):
        async def request(self, method, path, **kwargs):
            parsed = urlsplit(path)
            if method == "GET" and parsed.path.endswith("/artifacts/inputs/request"):
                assert parse_qs(parsed.query) == {"revision": ["request-1"]}
                return self.response(
                    status,
                    {
                        "code": "forbidden" if status == 403 else "not_found",
                        "retryable": False,
                        "message": "sensitive-error-canary",
                    },
                )
            return await super().request(method, path, **kwargs)

    async def scenario():
        store, spec = UnreadableRequestStore(), sqlmap_spec()
        service = await scanner_allocation(tmp_path, store, spec)
        try:
            result = await service._context.worker.invoke(prepared_request(spec))
            assert result.failure.code == "tool_input_invalid"
            assert not result.failure.retryable
            assert not marker.exists()
            assert "canary" not in repr(result)
            assert "canary" not in repr(await service.agent_state_snapshot(spec.allocation_id))
            assert not list((tmp_path / "work").rglob("scan_sqlmap-*"))
        finally:
            await release_scanner_allocation(service, spec)

    asyncio.run(scenario())


def ffuf_spec():
    spec = tool_spec()
    template = spec.agent_template
    template.ref.template_id = "ffuf-scan"
    template.description = "Fuzz one target with the supplied wordlist artifact."
    template.toolsets[0].tools = ["scan_ffuf"]
    template.execution.tool = "scan_ffuf"
    template.execution.arguments = {
        "url": ToolArgumentBinding(source="parameter", name="target"),
        "wordlist_ref": ToolArgumentBinding(source="artifact", name="wordlist"),
        "rate": ToolArgumentBinding(source="literal", value=7),
    }
    template.ref.digest = _agent_template_digest(template)
    return AllocationSpec.model_validate(spec.model_dump(by_alias=True, exclude_none=True))


def ffuf_request(spec):
    return StageContentRequest(
        apiVersion="contractor/v1alpha1",
        subtaskId="0",
        objective="Fuzz the target with the uploaded wordlist",
        instructions="Fuzz the target with the uploaded wordlist",
        parameters={"target": "https://target.invalid/FUZZ"},
        artifacts={
            "wordlist": {"namespace": "inputs", "name": "wordlist", "revision": "wordlist-1"}
        },
        resultArtifacts={"report": {"namespace": spec.namespace, "name": "report"}},
    )


@pytest.mark.parametrize("count", [2, 110])
def test_ffuf_wordlist_allocation_report_and_replay(tmp_path, monkeypatch, count):
    from test_scan_ffuf import WORDLIST_REF, ffuf_progress, ffuf_result
    from test_scan_toolset import executable

    marker = tmp_path / "ffuf-calls"
    payloads = "".join(f"item-{index}\r\n" for index in range(count))
    records = [ffuf_result(f"item-{index}", position=index + 1) for index in range(count)]
    output = "\n".join(json.dumps(record) for record in records)
    executable(
        tmp_path,
        "ffuf",
        "import pathlib, stat, sys\n"
        "if sys.argv[1:] == ['-V']:\n"
        "    print('fixture-ffuf')\n"
        "    sys.exit(0)\n"
        "args = sys.argv[1:]\n"
        "assert args[args.index('-u') + 1] == 'https://target.invalid/FUZZ'\n"
        "assert args[args.index('-rate') + 1] == '7'\n"
        "assert args[args.index('-mc') + 1] == 'all'\n"
        "wordlist = pathlib.Path(args[args.index('-w') + 1][:-5])\n"
        "assert stat.S_IMODE(wordlist.stat().st_mode) == 0o600\n"
        f"assert wordlist.read_bytes() == {payloads.replace(chr(13), '').encode()!r}\n"
        f"with open({str(marker)!r}, 'a') as calls:\n"
        "    calls.write('scan\\n')\n"
        f"print({output!r})\n"
        "print('private-diagnostic-canary', file=sys.stderr)\n"
        f"print({ffuf_progress(count, count)!r}, file=sys.stderr)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        store, spec = ArtifactStore(), ffuf_spec()
        store.values[("inputs", "wordlist")] = (
            payloads.encode(),
            "text/vnd.contractor.wordlist",
            "wordlist-1",
        )
        service = await scanner_allocation(tmp_path, store, spec)
        try:
            stage_request = ffuf_request(spec)
            result = await service._context.worker.invoke(stage_request)
            assert result.failure is None, result
            assert marker.read_text() == "scan\n"
            report = json.loads(store.values[(spec.namespace, "report")][0])
            assert report["tool"] == "scan_ffuf"
            assert report["inputArtifacts"] == {"wordlist": WORDLIST_REF}
            observation = report["observation"]
            assert observation["wordlistArtifact"] == WORDLIST_REF
            assert observation["wordlistEntries"] == count
            assert observation["scanComplete"] is True
            assert observation["payloadsAttempted"] == count
            assert observation["requestErrors"] == 0
            assert len(observation["results"]) == min(count, 100)
            assert observation["results"][0]["input"] == {"FUZZ": "item-0"}
            assert observation["resultsTruncated"] == (count > 100)
            assert observation["diagnosticsRedacted"] is True
            assert observation["stdout"] == observation["stderr"] == ""
            snapshot = await service.agent_state_snapshot(spec.allocation_id)
            assert snapshot.state.last_completed_invocation.metrics.model_calls == 0
            assert snapshot.state.last_completed_invocation.metrics.tool_calls == 1
            assert "canary" not in repr(snapshot)
            assert "canary" not in repr(report)
            assert not list((tmp_path / "work").rglob("scan_ffuf-*"))
        finally:
            await release_scanner_allocation(service, spec)

        spec.allocation_id = "replacement-allocation"
        replacement = await scanner_allocation(tmp_path, store, spec)
        try:
            replay = await replacement._context.worker.invoke(stage_request)
            assert replay.failure is None
            assert replay.result.artifacts == result.result.artifacts
            assert marker.read_text() == "scan\n"
            snapshot = await replacement.agent_state_snapshot(spec.allocation_id)
            assert snapshot.state.last_completed_invocation.metrics.model_calls == 0
            assert snapshot.state.last_completed_invocation.metrics.tool_calls == 0
        finally:
            await release_scanner_allocation(replacement, spec)

    asyncio.run(scenario())


def test_ffuf_request_failure_publishes_partial_report_without_rescan(tmp_path, monkeypatch):
    from test_scan_ffuf import WORDLIST_REF, ffuf_progress, ffuf_result
    from test_scan_toolset import executable

    marker = tmp_path / "ffuf-calls"
    executable(
        tmp_path,
        "ffuf",
        "import pathlib, sys\n"
        "if sys.argv[1:] == ['-V']:\n"
        "    sys.exit(0)\n"
        f"with open({str(marker)!r}, 'a') as calls:\n"
        "    calls.write('scan\\n')\n"
        f"print({json.dumps(ffuf_result())!r})\n"
        f"print({ffuf_progress(2, 2, 1)!r}, file=sys.stderr)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        store, spec = ArtifactStore(), ffuf_spec()
        store.values[("inputs", "wordlist")] = (b"foo\nbar\n", "text/plain", "wordlist-1")
        service = await scanner_allocation(tmp_path, store, spec)
        try:
            stage_request = ffuf_request(spec)
            result = await service._context.worker.invoke(stage_request)
            assert result.failure.code == "tool_execution_failed"
            report = json.loads(store.values[(spec.namespace, "report")][0])
            observation = report["observation"]
            assert report["inputArtifacts"] == {"wordlist": WORDLIST_REF}
            assert observation["errorCode"] == "scan_request_failed"
            assert observation["scanComplete"] is False
            assert observation["requestErrors"] == 1
            assert observation["results"][0]["input"] == {"FUZZ": "foo"}
            original_state = await service.agent_state_snapshot(spec.allocation_id)
            replay = await service._context.worker.invoke(stage_request)
            assert replay.failure.code == "tool_execution_failed"
            assert marker.read_text() == "scan\n"
            snapshot = await service.agent_state_snapshot(spec.allocation_id)
            # Same-allocation cache replay retains the original invocation metrics.
            assert snapshot == original_state
            assert snapshot.state.last_completed_invocation.metrics.tool_calls == 1
            assert not list((tmp_path / "work").rglob("scan_ffuf-*"))
        finally:
            await release_scanner_allocation(service, spec)

    asyncio.run(scenario())
