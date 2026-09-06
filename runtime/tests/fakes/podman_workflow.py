"""Repository sample resolved for a deterministic ADK model and Artifact peer.

No host command execution. Real-host tests replace only the model and remote
Control Plane/Artifact peers, keeping production allocation and Podman ownership.
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    AllocationWorkspaceSpecV2,
    ArtifactRef,
    ArtifactWriteResult,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    ResolvedInstructions,
    SandboxProfileRef,
    StageContentRequest,
    ToolsetRef,
    ToolsetSelection,
)
from contractor_runtime.digests import _agent_template_digest, _digest_bytes
from contractor_runtime.state import ProcessState
from fakes.model import scripted_model, text_result, tool_call
from fakes.spec import allocation_spec

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
EXPECTED_REPORT = b'{"passed": 3, "status": "ok"}\n'
COMMAND = "python3 -B check.py"


def document(path):
    return yaml.safe_load((CONFIGS / path).read_text())


def source_archive():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for name in ("calculator.py", "check.py"):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, (CONFIGS / "fixtures/podman-python-check" / name).read_bytes())
    return output.getvalue()


class ArtifactPeer:
    def __init__(self):
        self.source = source_archive()
        self.source_ref = ArtifactRef(
            namespace="inputs", name="source", revision=_digest_bytes(self.source)
        )
        self.writes = []
        self.observed = []
        self.fenced = False
        self.session = None

    @property
    def known_exact_refs(self):
        return (self.source_ref, *(ref for ref, _data in self.writes))

    @property
    def observation_cursor(self):
        return len(self.observed)

    def observed_exact_refs_since(self, cursor):
        return tuple(self.observed[cursor:])

    def clear_observations(self):
        self.observed.clear()

    async def read_artifact(self, ref):
        assert ref == self.source_ref  # never resolve a moving latest binding
        self.observed.append(ref)
        return ArtifactValue(
            artifact=ref,
            media_type="application/zip",
            data=self.source,
            binding_created_at=datetime.now(UTC),
            revision_created_at=datetime.now(UTC),
        )

    async def write_artifact(self, target, *, data, media_type, expected_revision):
        if self.fenced:
            raise ArtifactAPIError(409, "allocation_write_fenced", False)
        assert target == ArtifactRef(namespace="builder", name="check_report")
        assert media_type == "application/json" and expected_revision is None
        # Static scripted model bytes cannot hide a failed/missing disk result.
        assert data == EXPECTED_REPORT == (await self.session.read_text("report.json")).encode()
        ref = target.model_copy(update={"revision": _digest_bytes(data)})
        self.writes.append((ref, data))
        self.observed.append(ref)
        return ArtifactWriteResult(
            apiVersion=API_VERSION, artifact=ref, mediaType=media_type, size=len(data)
        )


def sample_spec(peer):
    template = document("agent-templates/podman_python_fixer.yaml")
    workflow = document("workflows/podman_python_check.yaml")["spec"]
    stage = workflow["stages"][workflow["entryStage"]]
    authored = template["spec"]
    spec = allocation_spec()
    spec.logical_agent_name = "builder"
    spec.namespace = stage["agents"]["builder"]["namespace"]
    spec.agent_template.ref.template_id = template["metadata"]["name"]
    spec.agent_template.ref.version = template["metadata"]["version"]
    spec.agent_template.description = authored["description"]
    text = (CONFIGS / authored["instructions"]["ref"]).read_text()
    spec.agent_template.instructions = ResolvedInstructions(
        ref=authored["instructions"]["ref"], text=text, digest=_digest_bytes(text.encode())
    )
    profile, version = authored["sandboxProfile"].split("@")
    spec.agent_template.sandbox_profile = SandboxProfileRef(
        sandboxProfileId=profile, version=version
    )
    spec.agent_template.toolsets = [
        ToolsetSelection(
            ref=ToolsetRef(toolsetId=item["ref"].split("@")[0], version=item["ref"].split("@")[1]),
            tools=item["tools"],
        )
        for item in authored["toolsets"]
    ]
    assert authored["modelPolicy"] == "worker@1"
    policy = document("model-policies/worker.yaml")["spec"]
    for name, value in policy.items():
        assert spec.model_policy.model_dump(by_alias=True)[name] == value
    spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
    spec.workspace = AllocationWorkspaceSpecV2(
        mode=stage["context"]["workspace"]["mode"],
        sources=[{"artifact": peer.source_ref, "target": ""}],
    )
    request = StageContentRequest(
        apiVersion=API_VERSION,
        subtaskId="0",
        objective=stage["objective"],
        instructions=(CONFIGS / stage["instructions"]["ref"]).read_text(),
        parameters={},
        artifacts={"source": peer.source_ref},
        resultArtifacts={
            "report": ArtifactRef.model_validate(stage["result"]["artifacts"]["report"]["from"])
        },
    )
    return spec, request


def sample_model(*, publish=True):
    calls = [
        ("read_file", {"path": "calculator.py"}),
        ("read_file", {"path": "check.py"}),
        ("edit", {"path": "calculator.py", "old": "return a - b", "new": "return a + b"}),
        ("exec_command", {"command": COMMAND, "cwd": "", "timeout_seconds": 30}),
        ("read_file", {"path": "report.json"}),
    ]
    if publish:
        calls.append(
            (
                "write_artifact",
                {
                    "namespace": "builder",
                    "name": "check_report",
                    "media_type": "application/json",
                    "data_base64": base64.b64encode(EXPECTED_REPORT).decode(),
                    "expected_revision": None,
                },
            )
        )
    return scripted_model(
        [tool_call(name, args, call_id=f"sample-{i}") for i, (name, args) in enumerate(calls)]
        + [text_result("Check finished")]
    )


async def exercise_sample(service, state, peer, model, *, publish=True):
    spec, request = sample_spec(peer)
    await service.prepare(spec)
    context = service._context
    session = context.project_workspace
    peer.session = session
    root, scratch = Path(session.storage.root), context.workspace.path
    assert set(context.tools) == {"read_file", "edit", "exec_command", "write_artifact"}
    assert peer.writes == []
    saved_write = context.tools["write_artifact"]
    provider = service._factories.workspace_provider
    cleanup = provider.cleanup
    cleaned = []

    async def cleanup_after_removal(storage):
        assert context.execution.removed and context.execution.stopped.is_set()
        cleaned.append(storage)
        await cleanup(storage)

    provider.cleanup = cleanup_after_removal
    try:
        completion = await context.worker.invoke(request)
        assert completion.failure is None, completion.failure
        assert completion.result is not None
        assert context.worker_state.metrics.counters.get("sandbox.completed_zero", 0) == 1
        assert await session.read_text("report.json") == EXPECTED_REPORT.decode()
        assert "return a + b" in await session.read_text("calculator.py")
        assert (await session.read_text("check.py")).encode() == (
            CONFIGS / "fixtures/podman-python-check/check.py"
        ).read_bytes()
        assert context.worker_state.metrics.counters["tool_calls.exec_command"] == 1
        if publish:
            assert "report" in completion.result.artifacts
            assert peer.writes == [(completion.result.artifacts["report"], EXPECTED_REPORT)]
        else:
            assert peer.writes == [] and not completion.result.artifacts
        assert not model.responses
        assert all(
            set(row["toolNames"]) == set(context.tools)
            for row in model.requests
            if row["toolNames"]
        )
    finally:
        # The remote Artifact peer owns the existing authoritative write fence.
        peer.fenced = True
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="sample-finalize",
                deadline=datetime.now(UTC) + timedelta(seconds=30),
            )
        )
        with pytest.raises(ArtifactAPIError, match="allocation_write_fenced"):
            await saved_write("builder", "check_report", "application/json", "e30=")
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        assert context.execution.stopped.is_set()
        assert not root.exists() and not scratch.exists()
        assert cleaned == [session.storage]
        assert (await state.snapshot()).process_state is not ProcessState.IDLE
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE
    return completion


class Commands:
    """Offline transport double; cannot run a model-selected host subprocess."""

    def __init__(self, fixture):
        self.fixture = fixture

    async def run(self, identity, command, cwd, *, deadline):
        from contractor_runtime.podman_command import CommandCapture

        root = self.fixture.backend.entry.root
        assert command == COMMAND and cwd == "" and deadline > time.monotonic()
        assert (root / "calculator.py").read_text() == "def add(a, b):\n    return a + b\n"
        assert (
            hashlib.sha256((root / "check.py").read_bytes()).digest()
            == hashlib.sha256(
                (CONFIGS / "fixtures/podman-python-check/check.py").read_bytes()
            ).digest()
        )
        (root / "report.json").write_bytes(EXPECTED_REPORT)
        return CommandCapture(0, b"", b"", 0, 0)
