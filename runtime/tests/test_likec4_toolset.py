from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import contractor_runtime.toolsets.likec4 as likec4_module
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.likec4 import (
    MAX_DOCUMENT_UTF8_BYTES,
    LikeC4ToolsetFactory,
    _run_likec4,
)
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "likec4-tool-recognizable-secret"
BASE_DOCUMENT = """specification {
  element actor
  element system
}
model {
  customer = actor 'Customer'
  app = system 'Application'
  customer -> app 'Uses'
}
views {
  view index {
    include *
  }
}
"""


def test_likec4_editing_validation_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state, namespace="architecture")
        captured: dict[str, Any] = {}

        monkeypatch.setattr(
            likec4_module.shutil,
            "which",
            lambda name: "/opt/likec4/bin/likec4" if name == "likec4" else None,
        )

        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            source = Path(command[command.index("--file") + 1])
            project = Path(command[-1])
            captured.update(
                {
                    "command": command,
                    "kwargs": kwargs,
                    "content": source.read_text(),
                    "source": source,
                    "project": project,
                }
            )
            assert source.parent == project
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "valid": True,
                        "errors": [],
                        "stats": {"totalFiles": 1, "totalErrors": 0},
                    }
                ).encode(),
                b"ignored banner",
            )

        monkeypatch.setattr(likec4_module.subprocess, "run", run)
        written = await tools["write_likec4"](BASE_DOCUMENT + f"// {SECRET}\n")
        appended = await tools["append_likec4"]("// generated from source\n")
        replaced = await tools["replace_likec4"]("Application", "Backend Application")
        assert replaced["replacementCount"] == 1
        revisions = [
            written["artifact"]["revision"],
            appended["artifact"]["revision"],
            replaced["artifact"]["revision"],
        ]
        assert len(set(revisions)) == 3
        assert client.write_count == 3

        read = await tools["read_likec4"](start_line=5, max_lines=6)
        assert "Backend Application" in read["text"]
        assert read["artifact"] == replaced["artifact"]
        validation = await tools["validate_likec4"]()
        assert validation["valid"]
        assert validation["issues"] == []
        assert validation["stats"] == {"totalFiles": 1, "totalErrors": 0}
        assert "Backend Application" in captured["content"]
        assert captured["command"][:5] == [
            "/opt/likec4/bin/likec4",
            "validate",
            "--json",
            "--no-layout",
            "--file",
        ]
        assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
        assert captured["kwargs"]["timeout"] == 30
        assert captured["kwargs"]["check"] is False
        assert "shell" not in captured["kwargs"]
        assert captured["kwargs"]["env"]["CI"] == "1"
        assert SECRET not in repr(captured["kwargs"]["env"])
        assert not captured["project"].exists()
        assert not list(tmp_path.glob(".likec4-validate-*"))

        observed = {
            (ref.namespace, ref.name, ref.revision)
            for tool in tools.values()
            for ref in tool.known_exact_refs
        }
        assert (
            "architecture",
            "architecture",
            replaced["artifact"]["revision"],
        ) in observed
        metrics = repr(state.metrics.snapshot())
        assert SECRET not in metrics
        assert "Backend Application" not in metrics
        assert state.metrics.counters["tool_calls.validate_likec4"] == 1

        sessions = {id(tool._session): tool._session for tool in tools.values()}
        assert len(sessions) == 1
        for tool in tools.values():
            await tool.close()
            assert tool._secrets == ()
        session = next(iter(sessions.values()))
        assert session._content is None
        assert session._revision is None

    asyncio.run(scenario())


def test_exact_seed_is_copied_and_later_stage_resumes(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        seed_data = BASE_DOCUMENT.encode()
        seed = client.seed("inputs", "existing-likec4", "text/vnd.likec4", seed_data)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="architecture")

        with pytest.raises(ValueError, match="requires an exact revision"):
            await tools["load_likec4"]("inputs", "existing-likec4")
        copied = await tools["load_likec4"]("inputs", "existing-likec4", seed.revision)
        assert copied["copied"] and copied["changed"]
        assert copied["artifact"]["namespace"] == "architecture"
        assert client.history[("inputs", "existing-likec4", seed.revision)].data == seed_data
        writes_after_copy = client.write_count

        later = await make_tools(tmp_path, client, WorkerState(), namespace="architecture")
        resumed = await later["load_likec4"](
            "architecture", "architecture", copied["artifact"]["revision"]
        )
        assert not resumed["copied"] and not resumed["changed"]
        assert client.write_count == writes_after_copy
        updated = await later["append_likec4"]("// resumed\n")
        assert updated["artifact"]["revision"] != copied["artifact"]["revision"]
        assert client.history[("inputs", "existing-likec4", seed.revision)].data == seed_data

    asyncio.run(scenario())


def test_plain_text_seed_is_canonicalized_on_copy(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        seed = client.seed("inputs", "plain", "text/plain", BASE_DOCUMENT.encode())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="architecture")
        copied = await tools["load_likec4"]("inputs", "plain", seed.revision)
        assert copied["mediaType"] == "text/vnd.likec4"
        target = client.bindings[("architecture", "architecture")]
        assert target.media_type == "text/vnd.likec4"
        assert target.data == BASE_DOCUMENT.encode()
        assert client.history[("inputs", "plain", seed.revision)].media_type == "text/plain"

    asyncio.run(scenario())


def test_seed_requires_media_type_utf8_and_size_without_target_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        wrong_media = client.seed(
            "inputs", "binary-media", "application/octet-stream", BASE_DOCUMENT.encode()
        )
        invalid_utf8 = client.seed("inputs", "binary", "text/vnd.likec4", b"\xff")
        oversized = client.seed(
            "inputs",
            "large",
            "text/vnd.likec4",
            b"x" * (MAX_DOCUMENT_UTF8_BYTES + 1),
        )
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="architecture")

        with pytest.raises(ValueError, match=r"text/vnd\.likec4"):
            await tools["load_likec4"]("inputs", "binary-media", wrong_media.revision)
        with pytest.raises(ValueError, match="valid UTF-8"):
            await tools["load_likec4"]("inputs", "binary", invalid_utf8.revision)
        with pytest.raises(ValueError, match="1 MiB"):
            await tools["load_likec4"]("inputs", "large", oversized.revision)
        assert ("architecture", "architecture") not in client.bindings
        assert client.write_count == 0

    asyncio.run(scenario())


def test_replace_bounds_and_stale_cas_preserve_selected_document(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state, namespace="architecture")
        created = await tools["write_likec4"]("node node node\n")
        first_revision = created["artifact"]["revision"]
        writes = client.write_count

        with pytest.raises(ValueError, match="ambiguous"):
            await tools["replace_likec4"]("node", "service")
        with pytest.raises(ValueError, match="absent"):
            await tools["replace_likec4"]("missing", "service")
        with pytest.raises(ValueError, match="exceeds"):
            await tools["replace_likec4"]("node", "service", count=4)
        with pytest.raises(ValueError, match="1 through 100"):
            await tools["replace_likec4"]("node", "service", count=0)
        assert client.write_count == writes

        replaced = await tools["replace_likec4"]("node", "service", count=2)
        assert replaced["replacementCount"] == 2
        assert replaced["matchingOccurrences"] == 3
        selected = await tools["read_likec4"]()
        assert selected["text"] == "service service node\n"

        current = client.bindings[("architecture", "architecture")]
        external = await client.write_artifact(
            ArtifactRef(namespace="architecture", name="architecture"),
            data=b"external update\n",
            media_type="text/vnd.likec4",
            expected_revision=current.revision,
        )
        with pytest.raises(ValueError, match="CAS"):
            await tools["append_likec4"]("stale append\n")
        with pytest.raises(ValueError, match="CAS"):
            await tools["write_likec4"]("stale whole write\n", expected_revision=first_revision)
        assert (await tools["read_likec4"]())["text"] == "service service node\n"
        durable = client.bindings[("architecture", "architecture")]
        assert durable.revision == external.artifact.revision
        assert durable.data == b"external update\n"

        with pytest.raises(ValueError, match="1 MiB"):
            await tools["append_likec4"]("x" * (MAX_DOCUMENT_UTF8_BYTES + 1))
        assert client.bindings[("architecture", "architecture")].revision == durable.revision
        assert "service service" not in repr(state.metrics.snapshot())

    asyncio.run(scenario())


def test_read_is_line_and_utf8_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="architecture")
        await tools["write_likec4"]("one-😀\ntwo-🚀\nthree-🧪\n")
        result = await tools["read_likec4"](start_line=2, max_lines=1)
        assert result["text"] == "two-🚀\n"
        assert result["startLine"] == 2 and result["endLine"] == 2
        assert result["truncated"]
        with pytest.raises(ValueError, match="positive integer"):
            await tools["read_likec4"](start_line=0)
        with pytest.raises(ValueError, match="line count"):
            await tools["read_likec4"](start_line=10)

    asyncio.run(scenario())


def test_validation_accepts_banner_current_and_legacy_json_and_normalizes_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(likec4_module.shutil, "which", lambda _name: "/bin/likec4")
    outputs = iter(
        [
            subprocess.CompletedProcess(
                [],
                1,
                (
                    "Update available\n"
                    + json.dumps(
                        {
                            "valid": False,
                            "errors": [
                                {
                                    "message": "Unknown element",
                                    "file": "/private/tmp/project/main.c4",
                                    "line": 3,
                                    "range": {"start": {"line": 3, "character": 2}},
                                }
                            ],
                            "stats": {"totalFiles": 1, "filteredErrors": 1},
                        }
                    )
                ).encode(),
                b"private stderr banner",
            ),
            subprocess.CompletedProcess([], 0, b"[]", b""),
        ]
    )
    monkeypatch.setattr(
        likec4_module.subprocess,
        "run",
        lambda *_args, **_kwargs: next(outputs),
    )
    invalid = _run_likec4(BASE_DOCUMENT, tmp_path)
    assert not invalid["valid"]
    assert invalid["issues"][0]["file"] == "main.c4"
    assert "/private/tmp" not in repr(invalid)
    assert invalid["stats"] == {"totalFiles": 1, "filteredErrors": 1}
    assert _run_likec4(BASE_DOCUMENT, tmp_path)["valid"]
    assert not list(tmp_path.glob(".likec4-validate-*"))


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (subprocess.CompletedProcess([], 2, b"{}", b"secret stderr"), "execution failure"),
        (subprocess.CompletedProcess([], 0, b"", b"secret stderr"), "empty or oversized"),
        (subprocess.CompletedProcess([], 0, b"not-json", b""), "invalid JSON"),
        (
            subprocess.CompletedProcess([], 0, json.dumps({"valid": True}).encode(), b""),
            "unexpected JSON shape",
        ),
        (
            subprocess.CompletedProcess(
                [], 0, json.dumps({"valid": True, "errors": ["bad"]}).encode(), b""
            ),
            "invalid diagnostics",
        ),
    ],
)
def test_validation_execution_and_output_failures_are_not_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[bytes],
    message: str,
) -> None:
    monkeypatch.setattr(likec4_module.shutil, "which", lambda _name: "/bin/likec4")
    monkeypatch.setattr(
        likec4_module.subprocess,
        "run",
        lambda *_args, **_kwargs: result,
    )
    validation = _run_likec4(BASE_DOCUMENT, tmp_path)
    assert not validation["valid"]
    assert message in validation["executionError"]
    assert "secret stderr" not in repr(validation)
    assert not list(tmp_path.glob(".likec4-validate-*"))


def test_validation_missing_timeout_and_oversized_output_are_not_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(likec4_module.shutil, "which", lambda _name: None)
    missing = _run_likec4(BASE_DOCUMENT, tmp_path)
    assert not missing["available"] and not missing["valid"]

    monkeypatch.setattr(likec4_module.shutil, "which", lambda _name: "/bin/likec4")

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("likec4", 30, stderr=b"private timeout banner")

    monkeypatch.setattr(likec4_module.subprocess, "run", timeout)
    timed_out = _run_likec4(BASE_DOCUMENT, tmp_path)
    assert not timed_out["valid"] and "timed out" in timed_out["executionError"]
    assert "private" not in repr(timed_out)
    assert not list(tmp_path.glob(".likec4-validate-*"))

    monkeypatch.setattr(
        likec4_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, b"x" * (likec4_module.MAX_VALIDATOR_OUTPUT_BYTES + 1), b""
        ),
    )
    oversized = _run_likec4(BASE_DOCUMENT, tmp_path)
    assert not oversized["valid"] and "oversized" in oversized["executionError"]


def test_tool_validate_never_maps_cli_failure_to_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="architecture")
        await tools["write_likec4"](BASE_DOCUMENT)
        monkeypatch.setattr(likec4_module.shutil, "which", lambda _name: None)
        result = await tools["validate_likec4"]()
        assert not result["valid"]
        assert not result["validatorAvailable"]
        assert result["validatorExecutionError"]

    asyncio.run(scenario())


def test_factory_rejects_unknown_tools_and_builtin_registry_matches(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = LikeC4ToolsetFactory(lambda _allocation, _settings: MemoryArtifactClient())
        selected = await factory.create_selected(
            selected=["read_likec4"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="architecture",
            runtime_settings=runtime_settings(),
            workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
            state=WorkerState(),
        )
        assert set(selected) == {"read_likec4"}
        with pytest.raises(ValueError, match="unknown selected tools"):
            await factory.create_selected(
                selected=["render_likec4"],
                allocation_id="allocation-1",
                run_id="run-1",
                namespace="architecture",
                runtime_settings=runtime_settings(),
                workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
                state=WorkerState(),
            )

    asyncio.run(scenario())
    assert built_in_factories(tmp_path).toolsets["likec4@1"].exported_tools == {
        "load_likec4",
        "write_likec4",
        "read_likec4",
        "append_likec4",
        "replace_likec4",
        "validate_likec4",
    }


async def make_tools(
    tmp_path: Path,
    client: MemoryArtifactClient,
    state: WorkerState,
    *,
    namespace: str,
) -> dict[str, Any]:
    factory = LikeC4ToolsetFactory(lambda _allocation, _settings: client)
    selected = await factory.create_selected(
        selected=sorted(factory.exported_tools),
        allocation_id="allocation-1",
        run_id="run-1",
        namespace=namespace,
        runtime_settings=runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
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
        self.write_count = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    def seed(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        return self._store(namespace, name, media_type, data)

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
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
        return ArtifactValue(artifact=exact, media_type=stored.media_type, data=stored.data)

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        key = (target.namespace, target.name)
        current = self.bindings.get(key)
        if expected_revision is None:
            if current is not None:
                raise ValueError("create-only CAS precondition failed")
        elif current is None or current.revision != expected_revision:
            raise ValueError("update CAS precondition failed")
        exact = self._store(target.namespace, target.name, media_type, data)
        self.write_count += 1
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
