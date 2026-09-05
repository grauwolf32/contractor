from __future__ import annotations

import asyncio
import hashlib
import io
import json
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import jcs
import pytest

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.toolsets.audit_results import AuditResultsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


def test_submit_check_result_derives_identity_and_builds_canonical_package() -> None:
    async def scenario() -> None:
        task_package, execution_manifest = fixture_inputs()
        client = FakeAuditArtifactClient(task_package, execution_manifest)
        state = WorkerState()
        factory = AuditResultsToolsetFactory(lambda _allocation, _settings: client)
        tools = await factory.create_selected(
            selected=["submit_check_result"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="audit-check",
            runtime_settings=runtime_settings(),
            workspace=workspace(),
            state=state,
        )

        result = await tools["submit_check_result"](
            assessment="satisfied",
            summary="The checked control is present.",
            completed=["source-trace"],
            gaps=[],
            evidence=[{"kind": "source-trace", "summary": "Guard at app.py:12."}],
        )

        assert result["artifact"] == {
            "namespace": "audit-check",
            "name": "result",
            "revision": "result-r1",
        }
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
                    "proposals": [],
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
        assert metric.arguments == {
            "assessment": "satisfied",
            "content_bytes": 31,
            "completed_count": 1,
            "gap_count": 0,
            "evidence_count": 1,
        }
        assert "Guard at app.py" not in repr(metric)

    asyncio.run(scenario())


def test_submit_check_result_rejects_forged_manifest_before_write() -> None:
    async def scenario() -> None:
        task_package, execution_manifest = fixture_inputs()
        forged = json.loads(execution_manifest)
        forged["items"][0]["subject_key"] = "different-subject"
        client = FakeAuditArtifactClient(task_package, jcs.canonicalize(forged))
        state = WorkerState()
        factory = AuditResultsToolsetFactory(lambda _allocation, _settings: client)
        tools = await factory.create_selected(
            selected=["submit_check_result"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="audit-check",
            runtime_settings=runtime_settings(),
            workspace=workspace(),
            state=state,
        )

        with pytest.raises(ValueError, match="identities differ"):
            await tools["submit_check_result"](
                "inconclusive", "A bounded gap remains.", [], ["missing-source"], []
            )
        assert client.written_payload == b""
        assert state.metrics.counters["tool_errors"] == 1

    asyncio.run(scenario())


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


def workspace() -> AllocationWorkspace:
    path = Path("/tmp/contractor-audit-result-test/allocation-1")
    return AllocationWorkspace(root=path.parent, path=path)


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class FakeAuditArtifactClient:
    def __init__(self, task_package: bytes, execution_manifest: bytes) -> None:
        self.task_package = task_package
        self.execution_manifest = execution_manifest
        self.written_payload = b""
        self.written_media_type = ""

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return ()

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        now = datetime.now(UTC)
        if ref == ArtifactRef(namespace="inputs", name="task"):
            payload, media_type, revision = self.task_package, "application/zip", "task-r1"
        elif ref == ArtifactRef(namespace="inputs", name="execution_manifest"):
            payload, media_type, revision = (
                self.execution_manifest,
                "application/json",
                "execution-r1",
            )
        else:
            raise AssertionError(f"unexpected read {ref}")
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
