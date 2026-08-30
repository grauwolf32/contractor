from __future__ import annotations

import asyncio
import io
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.source_analysis import (
    MAX_ARCHIVE_ENTRIES,
    MAX_FILE_UNCOMPRESSED_BYTES,
    MAX_TOTAL_UNCOMPRESSED_BYTES,
    SourceAnalysisToolsetFactory,
    _validate_entries,
)
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "source-tool-recognizable-secret"


def test_source_archive_tools_return_bounded_file_line_evidence(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        source = client.seed(
            "inputs",
            "source",
            "application/zip",
            make_zip(
                {
                    "pyproject.toml": "[project]\nname = 'demo'\n",
                    "src/app.py": (
                        "from fastapi import FastAPI\n"
                        "app = FastAPI()\n"
                        "@app.get('/health')\n"
                        "def health(): return {'ok': True}\n"
                    ),
                    ".git/config": "private repository metadata",
                    "node_modules/pkg/index.js": "ignored dependency",
                    "assets/logo.png": b"\x89PNG\x00binary",
                }
            ),
        )
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state)

        opened = await tools["open_source_archive"]("inputs", "source", source.revision)
        assert opened["artifact"] == {
            "namespace": "inputs",
            "name": "source",
            "revision": source.revision,
        }
        assert opened["fileCount"] == 2
        assert opened["ignoredCount"] == 3
        assert client.read_calls == 1

        listed = await tools["list_source_files"]()
        assert [item["path"] for item in listed["files"]] == ["pyproject.toml", "src/app.py"]
        assert not listed["truncated"]

        fixed = await tools["search_source"]("FastAPI", "*.py")
        assert [(item["path"], item["line"]) for item in fixed["matches"]] == [
            ("src/app.py", 1),
            ("src/app.py", 2),
        ]
        regex = await tools["search_source"](r"app\.get\('/health'\)", "**/*.py", True)
        assert regex["matches"][0]["line"] == 3

        read = await tools["read_source"]("src/app.py", start_line=2, max_lines=2)
        assert read["text"] == "app = FastAPI()\n@app.get('/health')\n"
        assert read["startLine"] == 2 and read["endLine"] == 3 and read["truncated"]

        # Opening the same exact revision is idempotent and does not fetch it again.
        assert await tools["open_source_archive"]("inputs", "source", source.revision) == opened
        assert client.read_calls == 1

        serialized_metrics = repr(state.metrics.snapshot())
        assert "FastAPI" not in serialized_metrics
        assert "@app.get" not in serialized_metrics
        assert SECRET not in serialized_metrics
        assert state.metrics.counters["tool_calls.open_source_archive"] == 2

        source_path = tmp_path / "source"
        assert source_path.is_dir()
        for tool in tools.values():
            await tool.close()
        assert not source_path.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.py",
        "/absolute.py",
        "dir\\file.py",
        "dir/./file.py",
        "dir//file.py",
        "C:/drive.py",
    ],
)
def test_open_rejects_unsafe_paths_without_escape(tmp_path: Path, unsafe_path: str) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", make_zip({unsafe_path: "unsafe"}))
        tools = await make_tools(tmp_path, client, WorkerState())
        with pytest.raises(ValueError, match="unsafe member path"):
            await tools["open_source_archive"]("inputs", "source", ref.revision)
        assert not (tmp_path / "source").exists()
        assert not (tmp_path.parent / "escape.py").exists()
        assert not any(path.name.startswith(".source-staging-") for path in tmp_path.iterdir())

    asyncio.run(scenario())


def test_open_rejects_links_invalid_zip_media_and_versionless_refs(tmp_path: Path) -> None:
    async def scenario() -> None:
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(symlink, "../../outside")

        client = ReadOnlyArtifactClient()
        link_ref = client.seed("inputs", "link", "application/zip", archive.getvalue())
        bad_ref = client.seed("inputs", "bad", "application/zip", b"not-a-zip")
        text_ref = client.seed("inputs", "text", "text/plain", make_zip({"ok.py": "ok"}))
        tools = await make_tools(tmp_path, client, WorkerState())

        with pytest.raises(ValueError, match="link or special"):
            await tools["open_source_archive"]("inputs", "link", link_ref.revision)
        with pytest.raises(ValueError, match="valid bounded ZIP"):
            await tools["open_source_archive"]("inputs", "bad", bad_ref.revision)
        with pytest.raises(ValueError, match="application/zip"):
            await tools["open_source_archive"]("inputs", "text", text_ref.revision)
        with pytest.raises(ValueError, match="revision is required"):
            await tools["open_source_archive"]("inputs", "text", None)

    asyncio.run(scenario())


def test_failed_reopen_preserves_previous_exact_tree_and_valid_reopen_replaces_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        first = client.seed("inputs", "first", "application/zip", make_zip({"one.py": "one"}))
        broken = client.seed("inputs", "broken", "application/zip", b"broken")
        second = client.seed("inputs", "second", "application/zip", make_zip({"two.go": "two"}))
        tools = await make_tools(tmp_path, client, WorkerState())

        await tools["open_source_archive"]("inputs", "first", first.revision)
        with pytest.raises(ValueError):
            await tools["open_source_archive"]("inputs", "broken", broken.revision)
        assert (await tools["read_source"]("one.py"))["text"] == "one"

        await tools["open_source_archive"]("inputs", "second", second.revision)
        assert [item["path"] for item in (await tools["list_source_files"]())["files"]] == [
            "two.go"
        ]
        assert not (tmp_path / "source" / "one.py").exists()

    asyncio.run(scenario())


def test_search_and_read_validation_is_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed(
            "inputs",
            "source",
            "application/zip",
            make_zip({"main.py": "alpha\nbeta\ngamma\n"}),
        )
        tools = await make_tools(tmp_path, client, WorkerState())
        with pytest.raises(ValueError, match="called first"):
            await tools["list_source_files"]()
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        with pytest.raises(ValueError, match="valid regular expression"):
            await tools["search_source"]("(", regex=True)
        with pytest.raises(ValueError, match="start_line"):
            await tools["read_source"]("main.py", start_line=9)
        with pytest.raises(ValueError, match="normalized and relative"):
            await tools["read_source"]("main.py/")

    asyncio.run(scenario())


def test_entry_resource_limits_are_rejected_before_extraction() -> None:
    too_many = [zipfile.ZipInfo(f"f-{index}.txt") for index in range(MAX_ARCHIVE_ENTRIES + 1)]
    with pytest.raises(ValueError, match="entry limit"):
        _validate_entries(too_many)

    too_large = zipfile.ZipInfo("large.txt")
    too_large.file_size = MAX_FILE_UNCOMPRESSED_BYTES + 1
    with pytest.raises(ValueError, match="file limit"):
        _validate_entries([too_large])

    first = zipfile.ZipInfo("first.txt")
    first.file_size = MAX_FILE_UNCOMPRESSED_BYTES
    count = MAX_TOTAL_UNCOMPRESSED_BYTES // MAX_FILE_UNCOMPRESSED_BYTES
    entries = []
    for index in range(count + 1):
        item = zipfile.ZipInfo(f"part-{index}.txt")
        item.file_size = first.file_size
        entries.append(item)
    with pytest.raises(ValueError, match="uncompressed limit"):
        _validate_entries(entries)


def test_factory_descriptor_and_subset_are_exact(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = SourceAnalysisToolsetFactory(
            lambda _allocation, _settings: ReadOnlyArtifactClient()
        )
        selected = await factory.create_selected(
            selected=["read_source"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="analysis",
            runtime_settings=runtime_settings(),
            workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
            state=WorkerState(),
        )
        assert set(selected) == {"read_source"}
        with pytest.raises(ValueError, match="unknown selected tools"):
            await factory.create_selected(
                selected=["execute_source"],
                allocation_id="allocation-1",
                run_id="run-1",
                namespace="analysis",
                runtime_settings=runtime_settings(),
                workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
                state=WorkerState(),
            )

    asyncio.run(scenario())
    assert built_in_factories(tmp_path).toolsets["source-analysis@1"].exported_tools == {
        "open_source_archive",
        "list_source_files",
        "search_source",
        "read_source",
    }


async def make_tools(
    tmp_path: Path,
    client: ReadOnlyArtifactClient,
    state: WorkerState,
) -> dict[str, object]:
    factory = SourceAnalysisToolsetFactory(lambda _allocation, _settings: client)
    selected = await factory.create_selected(
        selected=["open_source_archive", "list_source_files", "search_source", "read_source"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="analysis",
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


def make_zip(files: dict[str, str | bytes]) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content.encode() if isinstance(content, str) else content)
    return result.getvalue()


@dataclass(slots=True)
class StoredArtifact:
    revision: str
    media_type: str
    data: bytes = field(repr=False)


class ReadOnlyArtifactClient:
    def __init__(self) -> None:
        self.history: dict[tuple[str, str, str], StoredArtifact] = {}
        self._known: dict[tuple[str, str, str], ArtifactRef] = {}
        self._next_revision = 1
        self.read_calls = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    def seed(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        revision = f"revision-{self._next_revision}"
        self._next_revision += 1
        self.history[(namespace, name, revision)] = StoredArtifact(revision, media_type, data)
        return ArtifactRef(namespace=namespace, name=name, revision=revision)

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        exact = ref.require_exact()
        assert exact.revision is not None
        self.read_calls += 1
        stored = self.history[(exact.namespace, exact.name, exact.revision)]
        self._known[(exact.namespace, exact.name, exact.revision)] = exact
        return ArtifactValue(artifact=exact, media_type=stored.media_type, data=stored.data)
