from __future__ import annotations

import asyncio
import hashlib
import io
import json
import stat
import threading
import zipfile
from datetime import UTC, datetime
from types import SimpleNamespace

import jcs
import pytest
from google.adk.tools.function_tool import FunctionTool
from test_adk_runtime import stage_request
from test_security_findings_toolset import FakeFindingClient

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
    WorkerCompletionContract,
)
from contractor_runtime.toolsets.audit_results.collector import InvocationAuditCollector
from contractor_runtime.toolsets.audit_results.contracts import (
    AuditInvocationOwner,
    AuditTrustedInputs,
    SealedAuditSnapshot,
)
from contractor_runtime.toolsets.audit_results.encoding import CanonicalAuditPackageEncoder
from contractor_runtime.toolsets.audit_results.v2 import (
    AuditResultsToolsetFactory,
    ReadAuditTaskTool,
    SubmitCheckResultTool,
)
from contractor_runtime.toolsets.security_findings.facades import GeneralFindingTool
from contractor_runtime.toolsets.security_findings.publisher import FindingPublisher


def test_v2_adk_schema_keeps_batch_and_adds_revisioned_item_submission() -> None:
    task, execution = fixture_inputs()
    owner = AuditInvocationOwner("allocation-1", "invocation-1", digest(task), ("check-authz",))
    collector = InvocationAuditCollector(
        AuditTrustedInputs(owner, task, execution), CanonicalAuditPackageEncoder()
    )
    metrics = WorkerState().metrics
    submit = SubmitCheckResultTool(collector, metrics)
    declaration = FunctionTool(submit)._get_declaration().model_dump(mode="json", by_alias=True)
    schema = declaration["parametersJsonSchema"]
    assert "tool_context" not in schema["properties"]
    assert {"item_key", "expected_revision", "results"} <= set(schema["properties"])
    assert schema["$defs"]["BatchResultArgument"]["required"] == [
        "assessment",
        "summary",
        "completed",
        "gaps",
    ]
    read = ReadAuditTaskTool(
        collector,
        ArtifactRef(namespace="inputs", name="task", revision="r1"),
        metrics,
    )
    assert FunctionTool(read)._get_declaration().name == "read_audit_task"


def test_submit_check_result_advertises_bounded_batch_member_schema() -> None:
    task, execution = fixture_inputs()
    owner = AuditInvocationOwner("allocation-1", "invocation-1", digest(task), ("check-authz",))
    collector = InvocationAuditCollector(
        AuditTrustedInputs(owner, task, execution), CanonicalAuditPackageEncoder()
    )
    tool = SubmitCheckResultTool(collector, WorkerState().metrics)
    declaration = FunctionTool(tool)._get_declaration().model_dump(mode="json", by_alias=True)
    schema = declaration["parametersJsonSchema"]
    batch = schema["$defs"]["BatchResultArgument"]

    assert batch["required"] == ["assessment", "summary", "completed", "gaps"]
    assert batch["properties"]["evidence"]["items"]["$ref"].endswith("/EvidenceArgument")
    assert schema["properties"]["results"]["anyOf"][0]["items"]["$ref"].endswith(
        "/BatchResultArgument"
    )


def test_read_audit_task_returns_validated_json_without_raw_package_bytes() -> None:
    async def scenario() -> None:
        task_package, execution_manifest = fixture_inputs()
        client = FakeAuditArtifactClient(task_package, execution_manifest)
        state = WorkerState()
        tools = await prepared_tools(client, state)

        result = await tools["read_audit_task"](FakeToolContext("worker-invocation-1"))

        assert result["task"]["item_key"] == "check-authz"
        assert result["task"]["checklist"]["statement"] == (
            "Authorization is checked before object access."
        )
        assert result["taskPackageId"] == "task-check-authz"
        assert result["requestedCoverage"] == [["source-trace"]]
        assert result["taskArtifact"] == {
            "namespace": "inputs",
            "name": "task",
            "revision": "task-r1",
        }
        assert result["executionManifestDigest"] == digest(execution_manifest)
        assert "dataBase64" not in result
        assert state.metrics.counters["tool_calls"] == 1
        assert state.metrics.tool_calls[0].arguments is None

    asyncio.run(scenario())


def test_completion_derives_identity_and_builds_canonical_package() -> None:
    async def scenario() -> None:
        task_package, execution_manifest = fixture_inputs()
        client = FakeAuditArtifactClient(task_package, execution_manifest)
        state = WorkerState()
        tools = await prepared_tools(client, state)
        await (
            tools["submit_check_result"]
            .completion_binding.current()
            .register_proposal("worker-invocation-1", "candidate-authz", [])
        )

        result = await tools["submit_check_result"](
            assessment="satisfied",
            summary="The checked control is present.",
            completed=["source-trace"],
            gaps=[],
            tool_context=FakeToolContext("worker-invocation-1"),
            evidence=[{"kind": "source-trace", "summary": "Guard at app.py:12."}],
            proposal_keys=["candidate-authz"],
        )

        assert result["status"] == "recorded" and result["complete"]
        assert "artifact" not in result
        assert client.written_payload == b""
        completion = await finish(tools)
        assert completion.kind == "complete"
        assert completion.result.artifacts["result"] == ArtifactRef(
            namespace="audit-check", name="result", revision="result-r1"
        )
        assert client.written_media_type == "application/zip"
        result_document, evidence_document, package_manifest = decode_result_package(
            client.written_payload
        )
        assert result_document == {
            "schema": "contractor.audit.check-results.v1",
            "execution_manifest_digest": digest(execution_manifest),
            "results": [
                {
                    "item_key": "check-authz",
                    "subject_key": "check-authz",
                    "assessment": "satisfied",
                    "summary": "The checked control is present.",
                    "evidence_ids": ["ev-1"],
                    "coverage": {
                        "requested": ["source-trace"],
                        "completed": ["source-trace"],
                        "gaps": [],
                    },
                    "proposals": [
                        {
                            "invocation_id": "worker-invocation-1",
                            "client_key": "candidate-authz",
                        }
                    ],
                }
            ],
        }
        assert evidence_document["evidence"] == [
            {
                "id": "ev-1",
                "kind": "source-trace",
                "summary": "Guard at app.py:12.",
                "content_member_id": "ev-content-1",
            }
        ]
        assert package_manifest["kind"] == "check-results"
        assert [member["path"] for member in package_manifest["members"]] == [
            "check-results.json",
            "evidence.json",
            "evidence/ev-1.txt",
        ]
        assert state.metrics.counters["tool_calls"] == 1
        metric = state.metrics.tool_calls[0]
        assert metric.arguments is None
        assert "Guard at app.py" not in repr(metric)

    asyncio.run(scenario())


def test_successful_finding_links_through_late_bound_audit_factory():
    async def scenario():
        state = WorkerState()
        finding_client = FakeFindingClient()
        # The finding factory may run before Audit preparation in a template.
        finding = GeneralFindingTool(FindingPublisher(finding_client, state.metrics, (), state))
        task_package, execution_manifest = fixture_inputs()
        tools = await prepared_tools(
            FakeAuditArtifactClient(task_package, execution_manifest), state
        )
        context = SimpleNamespace(
            invocation_id="worker-invocation-1", function_call_id="finding-call-1"
        )
        receipt = await finding(
            title="Observed weakness",
            description="The guard can be bypassed.",
            tool_context=context,
        )
        for invalid in (receipt["receipt_id"], receipt["proposal_id"]):
            reply = await tools["submit_check_result"](
                assessment="inconclusive",
                summary="Needs review.",
                completed=[],
                gaps=[],
                proposal_keys=[invalid],
                tool_context=context,
            )
            assert reply["error"]["field"] == "proposal_keys"
            assert (
                await tools["submit_check_result"].completion_binding.current().snapshot()
            ).items == ()
        reply = await tools["submit_check_result"](
            assessment="inconclusive",
            summary="Needs review.",
            completed=[],
            gaps=[],
            proposal_keys=[receipt["client_key"]],
            tool_context=context,
        )
        assert reply["status"] == "recorded" and reply["complete"]
        assert finding_client.requests[0]["proposal"]["client_key"] == receipt["client_key"]
        await tools["submit_check_result"].completion_binding.end()
        tools["submit_check_result"].completion_binding.begin("worker-invocation-2")
        reply = await tools["submit_check_result"](
            assessment="inconclusive",
            summary="Needs review.",
            completed=[],
            gaps=[],
            proposal_keys=[receipt["client_key"]],
            tool_context=SimpleNamespace(invocation_id="worker-invocation-2"),
        )
        assert reply["error"]["field"] == "proposal_keys"

    asyncio.run(scenario())


def test_preparation_rejects_forged_manifest_before_tools_exist() -> None:
    async def scenario() -> None:
        task_package, execution_manifest = fixture_inputs()
        forged = json.loads(execution_manifest)
        forged["items"][0]["subject_key"] = "different-subject"
        client = FakeAuditArtifactClient(task_package, jcs.canonicalize(forged))
        state = WorkerState()
        with pytest.raises(ValueError, match="audit_result_invalid"):
            await prepared_tools(client, state)
        assert client.written_payload == b""
        assert not state.metrics.counters.get("tool_calls", 0)

    asyncio.run(scenario())


def test_completion_publishes_one_complete_ordered_batch() -> None:
    async def scenario() -> None:
        task_set, execution_manifest = fixture_batch_inputs()
        client = FakeAuditArtifactClient(task_set, execution_manifest)
        state = WorkerState()
        tools = await prepared_tools(client, state)
        await (
            tools["submit_check_result"]
            .completion_binding.current()
            .register_proposal("worker-invocation-1", "candidate-input", [])
        )

        assigned = await tools["read_audit_task"](FakeToolContext("worker-invocation-1"))
        assert assigned["batchSize"] == 2
        assert [task["item_key"] for task in assigned["tasks"]] == [
            "check-authz",
            "check-input",
        ]
        assert "task" not in assigned

        await tools["submit_check_result"](
            tool_context=FakeToolContext("worker-invocation-1"),
            results=[
                {
                    "assessment": "satisfied",
                    "summary": "Authorization is checked.",
                    "completed": ["source-trace"],
                    "gaps": [],
                    "evidence": [{"kind": "source-trace", "summary": "Guard at app.py:12."}],
                },
                {
                    "assessment": "inconclusive",
                    "summary": "Validation helper is unresolved.",
                    "completed": [],
                    "gaps": ["unresolved-helper"],
                    "proposal_keys": ["candidate-input"],
                },
            ],
        )

        assert client.written_payload == b""
        assert (await finish(tools)).kind == "complete"
        result_document, evidence_document, _ = decode_result_package(client.written_payload)
        assert [item["item_key"] for item in result_document["results"]] == [
            "check-authz",
            "check-input",
        ]
        assert result_document["results"][0]["evidence_ids"] == ["ev-1-1"]
        assert result_document["results"][1]["evidence_ids"] == []
        assert evidence_document["evidence"][0]["id"] == "ev-1-1"
        assert result_document["results"][1]["proposals"] == [
            {
                "invocation_id": "worker-invocation-1",
                "client_key": "candidate-input",
            }
        ]
        metric = state.metrics.tool_calls[-1]
        assert metric.arguments is None
        assert "Guard at app.py" not in repr(metric)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["single", "batch"])
def test_prepared_allocation_decodes_inputs_once_for_64_items_and_encodes_off_loop(
    monkeypatch, mode
) -> None:
    from contractor_runtime.toolsets.audit_results import encoding, packages

    calls: dict[str, list[int]] = {"task_set": [], "manifest": [], "encode": []}
    decode_set = packages._decode_task_set
    canonical_object = packages._canonical_object
    encode = encoding.CanonicalAuditPackageEncoder.encode

    def count_task_set(*args):
        calls["task_set"].append(threading.get_ident())
        return decode_set(*args)

    def count_manifest(payload, name):
        if name == "execution manifest":
            calls["manifest"].append(threading.get_ident())
        return canonical_object(payload, name)

    def count_encode(self, snapshot, *, inputs):
        calls["encode"].append(threading.get_ident())
        return encode(self, snapshot, inputs=inputs)

    # Patch the codec where every decoding path resolves it, including any
    # module that imported the decoder by name.
    monkeypatch.setattr(packages, "_decode_task_set", count_task_set)
    monkeypatch.setattr(packages, "_canonical_object", count_manifest)
    monkeypatch.setattr(encoding.CanonicalAuditPackageEncoder, "encode", count_encode)

    async def scenario() -> None:
        loop_thread = threading.get_ident()
        task_set, execution_manifest = fixture_task_set(64)
        client = FakeAuditArtifactClient(task_set, execution_manifest)
        tools = await prepared_tools(client, WorkerState())
        submit = tools["submit_check_result"]
        context = FakeToolContext("worker-invocation-1")
        keys = [f"check-{index:03d}" for index in range(64)]
        result = {
            "assessment": "satisfied",
            "summary": "Authorization is checked.",
            "completed": ["source-trace"],
            "gaps": [],
            "evidence": [{"kind": "source-trace", "summary": "Guard at app.py:12."}],
        }
        if mode == "single":
            for key in keys:
                receipt = await submit(tool_context=context, item_key=key, **result)
                assert receipt["status"] == "recorded"
        else:
            receipt = await submit(tool_context=context, results=[result] * 64)
        assert receipt["complete"] and receipt["acceptedCount"] == 64
        assert (await finish(tools)).kind == "complete"

        # One decoding per allocation, reused by preparation, every admission
        # and publication; each admission still encodes its prospective set.
        assert len(calls["task_set"]) == 1 and len(calls["manifest"]) == 1
        assert len(calls["encode"]) == (64 if mode == "single" else 1) + 1
        assert loop_thread not in calls["task_set"] + calls["manifest"] + calls["encode"]
        result_document, _, _ = decode_result_package(client.written_payload)
        assert [item["item_key"] for item in result_document["results"]] == keys
        assert result_document["execution_manifest_digest"] == digest(execution_manifest)

        # A later invocation of the same allocation reuses the decoding too.
        binding = submit.completion_binding
        collector = binding.current()
        sealed = SealedAuditSnapshot(collector.owner, (await collector.snapshot()).items)
        await binding.end()
        binding.begin("worker-invocation-2")
        tasks = await binding.current().read_tasks("worker-invocation-2")
        assert tasks["batchSize"] == 64 and tasks["taskPackageIds"][0] == "task-check-000"
        assert len(calls["task_set"]) == 1 and len(calls["manifest"]) == 1

        # A fresh encoder over the same sealed values produces the same bytes.
        fresh = encoding.CanonicalAuditPackageEncoder().encode(sealed, inputs=collector.inputs)
        assert fresh.data == client.written_payload

    asyncio.run(scenario())


def test_submit_check_result_rejects_incomplete_batch_without_write() -> None:
    async def scenario() -> None:
        task_set, execution_manifest = fixture_batch_inputs()
        client = FakeAuditArtifactClient(task_set, execution_manifest)
        state = WorkerState()
        tools = await prepared_tools(client, state)

        result = await tools["submit_check_result"](
            tool_context=FakeToolContext("worker-invocation-1"),
            results=[
                {
                    "assessment": "inconclusive",
                    "summary": "Only one item was returned.",
                    "completed": [],
                    "gaps": ["incomplete-batch"],
                }
            ],
        )
        assert result["status"] == "error"
        assert result["error"]["field"] == "results"
        assert not (
            await tools["submit_check_result"].completion_binding.current().snapshot()
        ).items
        assert (await finish(tools)).kind == "continue"
        assert client.written_payload == b""

    asyncio.run(scenario())


def test_cancelled_audit_result_calls_are_recorded() -> None:
    async def scenario() -> None:
        collector = BlockingCollector()
        state = WorkerState()
        read = ReadAuditTaskTool(
            collector, ArtifactRef(namespace="inputs", name="task", revision="r1"), state.metrics
        )
        submit = SubmitCheckResultTool(collector, state.metrics)
        context = FakeToolContext("worker-invocation-1")
        for call in (
            read(context),
            submit(context, assessment="satisfied", summary="Checked.", completed=[], gaps=[]),
        ):
            collector.started.clear()
            task = asyncio.create_task(call)
            await collector.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert [
            (call.tool, call.error.code, call.error.retryable)
            for call in state.metrics.tool_calls
            if call.error is not None
        ] == [
            ("read_audit_task", "tool_call_cancelled", True),
            ("submit_check_result", "tool_call_cancelled", True),
        ]
        assert state.metrics.counters["tool_errors"] == 2

    asyncio.run(scenario())


class BlockingCollector:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.owner = SimpleNamespace(item_keys=("check-authz",))

    def check_invocation(self, invocation_id: str) -> None:
        del invocation_id

    async def read_tasks(self, invocation_id: str) -> dict:
        del invocation_id
        self.started.set()
        await asyncio.Event().wait()
        return {}

    async def record(self, item: object, *, expected_revision: int | None) -> object:
        del item, expected_revision
        self.started.set()
        await asyncio.Event().wait()
        return None


class FakeToolContext:
    def __init__(self, invocation_id: str) -> None:
        self.invocation_id = invocation_id


def fixture_inputs() -> tuple[bytes, bytes]:
    task = {
        "schema": "contractor.audit.item-task.v1",
        "item_key": "check-authz",
        "kind": "checklist",
        "subject_key": "check-authz",
        "workflow_role": "check",
        "source_content_digest": "sha256:" + "a" * 64,
        "source_media_type": "application/yaml",
        "source_ref": {
            "namespace": "checklists",
            "name": "demo",
            "revision": "checklist-r1",
        },
        "canonical_inventory_digest": "sha256:" + "b" * 64,
        "checklist": {
            "version": "1",
            "statement": "Authorization is checked before object access.",
            "applicability": "Always applicable to the fixture.",
            "allowed_methods": ["static-trace"],
            "required_evidence": ["source-trace"],
            "review_policy": "automatic",
        },
    }
    task_bytes = jcs.canonicalize(task)
    task_package_id = "task-check-authz"
    task_package = package(
        task_package_id,
        "item-task",
        [("task-document", "task.json", "application/json", task_bytes)],
    )
    execution = {
        "schema": "contractor.audit.execution-manifest.v1",
        "items": [
            {
                "item_key": "check-authz",
                "ordinal": 0,
                "subject_key": "check-authz",
                "task_package_id": task_package_id,
                "task_package_digest": digest(task_package),
                "task_ref": {
                    "namespace": "audit-fixture",
                    "name": "task-check-authz",
                    "revision": "task-r1",
                },
                "inputs": [],
            }
        ],
    }
    return task_package, jcs.canonicalize(execution)


def fixture_batch_inputs() -> tuple[bytes, bytes]:
    first_package, first_manifest = fixture_inputs()
    first_execution = json.loads(first_manifest)
    with zipfile.ZipFile(io.BytesIO(first_package), "r") as archive:
        second_task = json.loads(archive.read("task.json"))
    second_task["item_key"] = "check-input"
    second_task["subject_key"] = "check-input"
    second_task["checklist"]["statement"] = "Input is validated before use."
    second_task_bytes = jcs.canonicalize(second_task)
    second_package = package(
        "task-check-input",
        "item-task",
        [("task-document", "task.json", "application/json", second_task_bytes)],
    )
    task_set = package(
        "task-set-fixture",
        "item-task-set",
        [
            ("task-000", "tasks/000.zip", "application/zip", first_package),
            ("task-001", "tasks/001.zip", "application/zip", second_package),
        ],
    )
    second_item = dict(first_execution["items"][0])
    second_item.update(
        {
            "item_key": "check-input",
            "ordinal": 1,
            "subject_key": "check-input",
            "task_package_id": "task-check-input",
            "task_package_digest": digest(second_package),
            "task_ref": {
                "namespace": "audit-fixture",
                "name": "task-check-input",
                "revision": "task-r2",
            },
        }
    )
    first_execution["items"].append(second_item)
    return task_set, jcs.canonicalize(first_execution)


def fixture_task_set(count: int) -> tuple[bytes, bytes]:
    """A valid task set of ``count`` checklist items (2..64) with its manifest."""
    single, single_manifest = fixture_inputs()
    template = json.loads(single_manifest)["items"][0]
    with zipfile.ZipFile(io.BytesIO(single), "r") as archive:
        base = json.loads(archive.read("task.json"))
    members, items = [], []
    for index in range(count):
        key = f"check-{index:03d}"
        task = {**base, "item_key": key, "subject_key": key}
        payload = package(
            f"task-{key}",
            "item-task",
            [("task-document", "task.json", "application/json", jcs.canonicalize(task))],
        )
        members.append((f"task-{index:03d}", f"tasks/{index:03d}.zip", "application/zip", payload))
        items.append(
            {
                **template,
                "item_key": key,
                "ordinal": index,
                "subject_key": key,
                "task_package_id": f"task-{key}",
                "task_package_digest": digest(payload),
                "task_ref": {"namespace": "audit-fixture", "name": key, "revision": "task-r1"},
            }
        )
    task_set = package("task-set-fixture", "item-task-set", members)
    manifest = {"schema": "contractor.audit.execution-manifest.v1", "items": items}
    return task_set, jcs.canonicalize(manifest)


def package(package_id: str, kind: str, members: list[tuple[str, str, str, bytes]]) -> bytes:
    ordered = sorted(members, key=lambda item: item[1])
    manifest = {
        "schema": "contractor.audit.package.v1",
        "package_id": package_id,
        "kind": kind,
        "members": [
            {
                "id": member_id,
                "path": path,
                "media_type": media_type,
                "size": len(data),
                "digest": digest(data),
            }
            for member_id, path, media_type, data in ordered
        ],
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        write_member(archive, "manifest.json", jcs.canonicalize(manifest))
        for _, path, _, data in ordered:
            write_member(archive, path, data)
    return output.getvalue()


def write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, data)


def decode_result_package(payload: bytes) -> tuple[dict, dict, dict]:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        manifest_bytes = archive.read("manifest.json")
        result_bytes = archive.read("check-results.json")
        evidence_bytes = archive.read("evidence.json")
    package_manifest = json.loads(manifest_bytes)
    assert manifest_bytes == jcs.canonicalize(package_manifest)
    for member in package_manifest["members"]:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            data = archive.read(member["path"])
        assert member["size"] == len(data)
        assert member["digest"] == digest(data)
    return json.loads(result_bytes), json.loads(evidence_bytes), package_manifest


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken="secret",
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class FakeAuditArtifactClient:
    allocation_id = "allocation-1"

    def __init__(self, task_package: bytes, execution_manifest: bytes) -> None:
        self.task_package = task_package
        self.execution_manifest = execution_manifest
        self.written_payload = b""
        self.written_media_type = ""

    async def read_artifact(self, ref: ArtifactRef, *, max_bytes: int) -> ArtifactValue:
        now = datetime.now(UTC)
        if ref == ArtifactRef(namespace="inputs", name="task", revision="task-r1"):
            payload, media_type, revision = self.task_package, "application/zip", "task-r1"
        elif ref == ArtifactRef(
            namespace="inputs", name="execution_manifest", revision="execution-r1"
        ):
            payload, media_type, revision = (
                self.execution_manifest,
                "application/json",
                "execution-r1",
            )
        else:
            raise AssertionError(f"unexpected read {ref}")
        assert len(payload) <= max_bytes
        return ArtifactValue(
            artifact=ArtifactRef(namespace=ref.namespace, name=ref.name, revision=revision),
            media_type=media_type,
            data=payload,
            binding_created_at=now,
            revision_created_at=now,
        )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        assert target == ArtifactRef(namespace="audit-check", name="result")
        assert expected_revision is None
        self.written_payload = data
        self.written_media_type = media_type
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=ArtifactRef(
                namespace=target.namespace, name=target.name, revision="result-r1"
            ),
            mediaType=media_type,
            size=len(data),
        )


async def prepared_tools(client, state):
    contract = WorkerCompletionContract(
        kind="audit-check-results@1",
        task=ArtifactRef(namespace="inputs", name="task", revision="task-r1"),
        executionManifest=ArtifactRef(
            namespace="inputs", name="execution_manifest", revision="execution-r1"
        ),
        resultArtifact=ArtifactRef(namespace="audit-check", name="result"),
    )
    tools = await AuditResultsToolsetFactory(lambda *_: client).create_selected(
        selected=["read_audit_task", "submit_check_result"],
        allocation_id=client.allocation_id,
        namespace="audit-check",
        runtime_settings=runtime_settings(),
        state=state,
        completion_contract=contract,
    )
    tools["submit_check_result"].completion_binding.begin("worker-invocation-1")
    return tools


async def finish(tools):
    binding = tools["submit_check_result"].completion_binding
    return await binding.finish(
        request=stage_request().model_copy(
            update={"result_artifacts": {"result": binding.contract.result_artifact}}
        ),
        deadline=asyncio.get_running_loop().time() + 5,
        check_active=lambda: None,
    )


def test_http_capture_evidence_retains_trusted_time_identity_and_exact_content(
    tmp_path, monkeypatch
):
    import httpx
    from test_http_toolset import create_tools

    from contractor_runtime.toolsets.http.capture import CapturedAttempt

    original = CapturedAttempt.from_request

    def capture(cls, request, **kwargs):
        value = original(request, **kwargs)
        value.at = "2026-10-27T12:00:00.125Z"
        return value

    monkeypatch.setattr(CapturedAttempt, "from_request", classmethod(capture))

    async def scenario():
        async def handler(request):
            return httpx.Response(200, content=b"real exchange proof", request=request)

        http, state = await create_tools(tmp_path, handler)
        # This is pinned factory configuration, never model input.
        http["http_request"]._session._deployment_marker = "deployment-test"
        task, manifest = fixture_inputs()
        client = FakeAuditArtifactClient(task, manifest)
        tools = await prepared_tools(client, state)
        context = FakeToolContext("worker-invocation-1")
        try:
            await http["http_request"]("https://target.example/", tool_context=context)
            submit = tools["submit_check_result"]
            kwargs = dict(
                assessment="satisfied",
                summary="Observed control.",
                completed=["source-trace"],
                gaps=[],
                evidence=[
                    {"kind": "source-trace", "summary": "Response proof", "http_request_id": 1}
                ],
            )
            first = await submit(context, **kwargs)
            assert first["status"] == "recorded"
            state.http_captures.clear("http")
            assert await submit(context, **kwargs) == first
            forged = await submit(
                context,
                **{
                    **kwargs,
                    "evidence": [
                        {
                            "kind": "source-trace",
                            "summary": "bad",
                            "observed_at": "2026-10-27T12:00:00Z",
                        }
                    ],
                },
            )
            assert forged["status"] == "error"
            before = await submit._collector.snapshot()
            for bad in (0, True, 2, "1"):
                result = await submit(
                    context,
                    **{
                        **kwargs,
                        "evidence": [
                            {"kind": "source-trace", "summary": "bad", "http_request_id": bad}
                        ],
                    },
                )
                assert result["status"] == "error"
                assert await submit._collector.snapshot() == before
            assert (await finish(tools)).kind == "complete"
            with zipfile.ZipFile(io.BytesIO(client.written_payload)) as archive:
                evidence = json.loads(archive.read("evidence.json"))["evidence"][0]
                assert evidence["observed_at"] == "2026-10-27T12:00:00.125Z"
                assert evidence["deployment_marker"] == "deployment-test"
                assert evidence["http_request_id"] == "1"
                capture = json.loads(archive.read("evidence/ev-1.json"))
                assert capture["request_tag"] == evidence["http_request_tag"]
                assert capture["attempts"][0]["response"]["message"]["body_base64"]
        finally:
            await http["http_request"].close()

    asyncio.run(scenario())
