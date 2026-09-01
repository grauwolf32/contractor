from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.text_artifacts import (
    MAX_TEXT_WRITE_BYTES,
    TextArtifactsToolsetFactory,
)
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "text-tool-recognizable-secret"
UTF8_DOCUMENT = "one-😀\ntwo-🚀\nthree-🧪\nfour-✅\n"


def test_text_tools_page_utf8_and_enforce_namespace_cas(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        source = client.seed(
            "inputs",
            "report",
            "text/markdown",
            UTF8_DOCUMENT.encode(),
        )
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state)

        read = await tools["read_text_artifact"](
            namespace="inputs",
            name="report",
            revision=source.revision,
            start_line=2,
            max_lines=2,
        )
        assert read == {
            "artifact": {
                "namespace": "inputs",
                "name": "report",
                "revision": source.revision,
            },
            "mediaType": "text/markdown",
            "size": len(UTF8_DOCUMENT.encode()),
            "totalLines": 4,
            "startLine": 2,
            "endLine": 3,
            "text": "two-🚀\nthree-🧪\n",
            "truncated": True,
            "partialLine": False,
        }

        created = await tools["write_text_artifact"](
            name="analysis",
            text="# Analysis\n",
            media_type="text/markdown",
        )
        assert created["artifact"]["namespace"] == "worker-space"
        first_revision = created["artifact"]["revision"]
        updated = await tools["write_text_artifact"](
            name="analysis",
            text="# Updated\n",
            media_type="text/markdown",
            expected_revision=first_revision,
        )
        assert updated["artifact"]["revision"] != first_revision
        with pytest.raises(ValueError, match="CAS"):
            await tools["write_text_artifact"](
                name="analysis",
                text="# Stale\n",
                media_type="text/markdown",
                expected_revision=first_revision,
            )
        current = await client.read_artifact(ArtifactRef(namespace="worker-space", name="analysis"))
        assert current.data == b"# Updated\n"
        assert state.metrics.counters["tool_calls.read_text_artifact"] == 1
        assert state.metrics.counters["tool_calls.write_text_artifact"] == 3

        observed = {
            (ref.namespace, ref.name, ref.revision)
            for tool in tools.values()
            for ref in tool.known_exact_refs
        }
        assert ("worker-space", "analysis", updated["artifact"]["revision"]) in observed

    asyncio.run(scenario())


def test_text_tools_reject_invalid_windows_utf8_and_oversize(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        invalid = client.seed("inputs", "binary", "application/octet-stream", b"\xff")
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state)

        with pytest.raises(ValueError, match="positive integer"):
            await tools["read_text_artifact"]("inputs", "binary", invalid.revision, 0, 1)
        with pytest.raises(ValueError, match="valid UTF-8"):
            await tools["read_text_artifact"]("inputs", "binary", invalid.revision)
        with pytest.raises(ValueError, match="1 MiB"):
            await tools["write_text_artifact"](
                "too-large",
                "x" * (MAX_TEXT_WRITE_BYTES + 1),
                "text/plain",
            )
        assert ("worker-space", "too-large") not in client.bindings

    asyncio.run(scenario())


def test_text_content_and_secrets_are_redacted_from_metrics(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state)
        document = f"private document contains {SECRET}"

        await tools["write_text_artifact"]("report", document, "text/plain")
        serialized = repr(state.metrics.snapshot()) + repr(
            state.metrics.build_report(report_id="report-1", duration_ms=1)
        )
        assert document not in serialized
        assert SECRET not in serialized
        call = state.metrics.tool_calls[0]
        assert call.arguments["content"] == {"redacted": True, "size": len(document)}
        assert call.arguments["utf8_size"] == len(document.encode())

        for tool in tools.values():
            await tool.close()
            assert tool._secrets == ()

    asyncio.run(scenario())


def test_text_tools_enforce_shared_memory_binding_visibility(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        hidden = client.seed("worker-space", "memory.hidden", "text/plain", b"hidden")
        allowed = client.seed("inputs", "memory.workflow_input", "text/plain", b"allowed")
        tools = await make_tools(tmp_path, client, WorkerState())

        calls_before = (client.read_calls, client.write_calls)
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_text_artifact"](hidden.namespace, hidden.name, hidden.revision)
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_text_artifact"](
                hidden.name,
                "must not replace",
                "text/plain",
                hidden.revision,
            )
        assert (client.read_calls, client.write_calls) == calls_before
        assert client.bindings[(hidden.namespace, hidden.name)].data == b"hidden"

        visible = await tools["read_text_artifact"](
            allowed.namespace, allowed.name, allowed.revision
        )
        assert visible["text"] == "allowed"
        observed = {
            (ref.namespace, ref.name) for tool in tools.values() for ref in tool.known_exact_refs
        }
        assert (hidden.namespace, hidden.name) not in observed
        assert (allowed.namespace, allowed.name) in observed

    asyncio.run(scenario())


def test_factory_rejects_unknown_tools_and_builtin_registry_matches(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = TextArtifactsToolsetFactory(lambda _allocation, _settings: MemoryArtifactClient())
        with pytest.raises(ValueError, match="unknown selected tools"):
            await factory.create_selected(
                selected=["delete_text_artifact"],
                allocation_id="allocation-1",
                run_id="run-1",
                namespace="analysis",
                runtime_settings=runtime_settings(),
                workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
                state=WorkerState(),
            )

    asyncio.run(scenario())
    registry = built_in_factories(tmp_path)
    assert registry.toolsets["text-artifacts@1"].exported_tools == {
        "read_text_artifact",
        "write_text_artifact",
    }


async def make_tools(
    tmp_path: Path,
    client: MemoryArtifactClient,
    state: WorkerState,
) -> dict[str, object]:
    factory = TextArtifactsToolsetFactory(lambda _allocation, _settings: client)
    selected = await factory.create_selected(
        selected=["read_text_artifact", "write_text_artifact"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="worker-space",
        runtime_settings=runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        state=state,
    )
    return dict(selected)


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken=SECRET,
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )


@dataclass(slots=True)
class StoredArtifact:
    revision: str
    media_type: str
    data: bytes = field(repr=False)


class MemoryArtifactClient:
    def __init__(self) -> None:
        self.bindings: dict[tuple[str, str], StoredArtifact] = {}
        self.history: dict[tuple[str, str, str], StoredArtifact] = {}
        self._known: dict[tuple[str, str, str], ArtifactRef] = {}
        self._next_revision = 1
        self.read_calls = 0
        self.write_calls = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    def seed(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        return self._store(namespace, name, media_type, data)

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.read_calls += 1
        if ref.revision is None:
            stored = self.bindings[(ref.namespace, ref.name)]
        else:
            stored = self.history[(ref.namespace, ref.name, ref.revision)]
        exact = ArtifactRef(
            namespace=ref.namespace,
            name=ref.name,
            revision=stored.revision,
        )
        self._remember(exact)
        return ArtifactValue(
            artifact=exact,
            media_type=stored.media_type,
            data=stored.data,
            binding_created_at=datetime.now(UTC),
            revision_created_at=datetime.now(UTC),
        )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        self.write_calls += 1
        key = (target.namespace, target.name)
        current = self.bindings.get(key)
        if expected_revision is None:
            if current is not None:
                raise ValueError("create-only CAS precondition failed")
        elif current is None or current.revision != expected_revision:
            raise ValueError("update CAS precondition failed")
        exact = self._store(target.namespace, target.name, media_type, data)
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=exact,
            mediaType=media_type,
            size=len(data),
        )

    def _store(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        revision = f"revision-{self._next_revision}"
        self._next_revision += 1
        stored = StoredArtifact(revision=revision, media_type=media_type, data=data)
        self.bindings[(namespace, name)] = stored
        self.history[(namespace, name, revision)] = stored
        exact = ArtifactRef(namespace=namespace, name=name, revision=revision)
        self._remember(exact)
        return exact

    def _remember(self, ref: ArtifactRef) -> None:
        assert ref.revision is not None
        self._known[(ref.namespace, ref.name, ref.revision)] = ref
