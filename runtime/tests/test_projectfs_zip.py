from __future__ import annotations

import asyncio
import io
import stat
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from contractor_runtime.artifacts import ArtifactClientError, ArtifactValue
from contractor_runtime.contracts import (
    AllocationWorkspaceSource,
    AllocationWorkspaceSpec,
    ArtifactRef,
)
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    WorkspacePreparationError,
    hydrate_workspace,
)
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings

REVISION = "sha256:" + "a" * 64
NOW = datetime(2026, 9, 1, tzinfo=UTC)


class FakeArtifactReader:
    def __init__(self, values: dict[tuple[str, str, str], ArtifactValue]) -> None:
        self.values = values

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        assert ref.revision is not None
        try:
            return self.values[(ref.namespace, ref.name, ref.revision)]
        except KeyError:
            raise ArtifactClientError("missing") from None


def test_safe_multi_source_archives_have_equal_local_and_memory_text_views(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = archive(
            {
                "src/app.py": b"print('ok')\n",
                "src/empty/": None,
                "assets/logo.bin": b"\x00\xffbinary",
            }
        )
        frontend = archive({"README.md": b"hello\n"})
        spec, reader = workspace_inputs(
            [("backend", "backend", backend), ("frontend", "frontend", frontend)]
        )
        local_provider = LocalWorkspaceProvider(settings("local", tmp_path / "local"))
        memory_provider = MemoryWorkspaceProvider(settings("memory"))

        local = await hydrate_workspace(
            provider=local_provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="local-allocation",
            timeout_seconds=5,
        )
        memory = await hydrate_workspace(
            provider=memory_provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="memory-allocation",
            timeout_seconds=5,
        )

        local_snapshot = await local.snapshot()
        memory_snapshot = await memory.snapshot()
        assert local_snapshot == memory_snapshot
        assert [file.path for file in local_snapshot.files] == [
            "backend/src/app.py",
            "frontend/README.md",
        ]
        assert local_snapshot.binary_paths == ("backend/assets/logo.bin",)
        assert "backend/src/empty" in local_snapshot.directories
        assert (
            local.storage.filesystem.cat(
                f"{local.storage.root}/run_workdir/backend/assets/logo.bin"
            )
            == b"\x00\xffbinary"
        )
        assert not memory.storage.filesystem.exists(
            f"{memory.storage.root}/run_workdir/backend/assets/logo.bin"
        )

        await local_provider.cleanup(local.storage)
        await memory_provider.cleanup(memory.storage)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "members",
    [
        {"../escape": b"bad"},
        {"safe\\escape": b"bad"},
        {"safe/../../escape": b"bad"},
        {"safe": b"file", "safe/child": b"conflict"},
        {"safe/": None, "safe": b"conflict"},
    ],
)
def test_invalid_archive_paths_and_type_conflicts_leave_no_partial_tree(
    tmp_path: Path, members: dict[str, bytes | None]
) -> None:
    assert_invalid_and_clean(tmp_path, archive(members), "workspace_source_invalid")


def test_zip_rejects_links_special_files_and_unicode_normalized_duplicates(
    tmp_path: Path,
) -> None:
    link = zipfile.ZipInfo("link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert_invalid_and_clean(tmp_path, archive_infos([(link, b"target")]))

    fifo = zipfile.ZipInfo("pipe")
    fifo.external_attr = (stat.S_IFIFO | 0o600) << 16
    assert_invalid_and_clean(tmp_path, archive_infos([(fifo, b"")]))

    composed = "café.txt"
    decomposed = unicodedata.normalize("NFD", composed)
    assert_invalid_and_clean(
        tmp_path,
        archive({composed: b"one", decomposed: b"two"}),
    )


def test_zip_limits_are_exact_and_compression_bombs_fail_closed(tmp_path: Path) -> None:
    async def scenario() -> None:
        exact_payload = b"12345678"
        spec, reader = workspace_inputs([("source", "", archive({"exact.txt": exact_payload}))])
        exact_provider = LocalWorkspaceProvider(
            settings(
                "local",
                tmp_path / "exact",
                limits=WorkspaceLimits(
                    max_files=1,
                    max_expanded_bytes=8,
                    max_managed_text_bytes=8,
                    max_file_bytes=8,
                ),
            )
        )
        session = await hydrate_workspace(
            provider=exact_provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="exact",
            timeout_seconds=5,
        )
        assert await session.read_text("exact.txt") == "12345678"
        await exact_provider.cleanup(session.storage)

        over_spec, over_reader = workspace_inputs(
            [("source", "", archive({"over.txt": exact_payload + b"9"}))]
        )
        with pytest.raises(WorkspacePreparationError) as over:
            await hydrate_workspace(
                provider=exact_provider,
                spec=over_spec,
                artifact_reader=over_reader,
                allocation_id="over",
                timeout_seconds=5,
            )
        assert over.value.code == "workspace_capacity_exceeded"
        assert list((tmp_path / "exact").iterdir()) == []

        bomb = archive({"bomb.txt": b"0" * (2 * 1024 * 1024)})
        bomb_spec, bomb_reader = workspace_inputs([("source", "", bomb)])
        roomy = LocalWorkspaceProvider(
            settings(
                "local",
                tmp_path / "bomb",
                limits=WorkspaceLimits(
                    max_files=10,
                    max_expanded_bytes=4 * 1024 * 1024,
                    max_managed_text_bytes=4 * 1024 * 1024,
                    max_file_bytes=4 * 1024 * 1024,
                ),
            )
        )
        with pytest.raises(WorkspacePreparationError) as rejected:
            await hydrate_workspace(
                provider=roomy,
                spec=bomb_spec,
                artifact_reader=bomb_reader,
                allocation_id="bomb",
                timeout_seconds=5,
            )
        assert rejected.value.code == "workspace_capacity_exceeded"
        assert list((tmp_path / "bomb").iterdir()) == []

    asyncio.run(scenario())


def test_wrong_media_type_or_exact_ref_is_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs([("source", "", archive({"ok.txt": b"ok"}))])
        original = next(iter(reader.values.values()))
        provider = LocalWorkspaceProvider(settings("local", tmp_path / "work"))

        reader.values[next(iter(reader.values))] = ArtifactValue(
            artifact=original.artifact,
            media_type="application/octet-stream",
            data=original.data,
            binding_created_at=NOW,
            revision_created_at=NOW,
        )
        with pytest.raises(WorkspacePreparationError) as media:
            await hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="media",
                timeout_seconds=5,
            )
        assert media.value.code == "workspace_source_invalid"

        reader.values[next(iter(reader.values))] = ArtifactValue(
            artifact=ArtifactRef(namespace="inputs", name="source", revision="other"),
            media_type="application/zip",
            data=original.data,
            binding_created_at=NOW,
            revision_created_at=NOW,
        )
        with pytest.raises(WorkspacePreparationError) as ref:
            await hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="ref",
                timeout_seconds=5,
            )
        assert ref.value.code == "workspace_source_invalid"
        assert list((tmp_path / "work").iterdir()) == []

    asyncio.run(scenario())


def test_later_source_failure_and_cancellation_erase_earlier_source(tmp_path: Path) -> None:
    class BlockingReader(FakeArtifactReader):
        started = asyncio.Event()

        async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
            self.started.set()
            await asyncio.Event().wait()
            return await super().read_artifact(ref)

    async def scenario() -> None:
        root = tmp_path / "partial"
        provider = LocalWorkspaceProvider(settings("local", root))
        spec, reader = workspace_inputs(
            [
                ("first", "first", archive({"ok.txt": b"already extracted"})),
                ("second", "second", archive({"../escape": b"bad"})),
            ]
        )
        with pytest.raises(WorkspacePreparationError):
            await hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="partial",
                timeout_seconds=5,
            )
        assert list(root.iterdir()) == []

        cancel_root = tmp_path / "cancel"
        cancel_provider = LocalWorkspaceProvider(settings("local", cancel_root))
        cancel_spec, values = workspace_inputs([("source", "", archive({"ok.txt": b"never read"}))])
        blocking = BlockingReader(values.values)
        task = asyncio.create_task(
            hydrate_workspace(
                provider=cancel_provider,
                spec=cancel_spec,
                artifact_reader=blocking,
                allocation_id="cancel",
                timeout_seconds=5,
            )
        )
        await blocking.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert list(cancel_root.iterdir()) == []

    asyncio.run(scenario())


def assert_invalid_and_clean(
    tmp_path: Path,
    payload: bytes,
    expected_code: str = "workspace_source_invalid",
) -> None:
    async def scenario() -> None:
        root = tmp_path / f"invalid-{len(list(tmp_path.iterdir()))}"
        provider = LocalWorkspaceProvider(settings("local", root))
        spec, reader = workspace_inputs([("source", "", payload)])
        with pytest.raises(WorkspacePreparationError) as rejected:
            await hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="invalid",
                timeout_seconds=5,
            )
        assert rejected.value.code == expected_code
        assert list(root.iterdir()) == []

    asyncio.run(scenario())


def workspace_inputs(
    sources: list[tuple[str, str, bytes]],
) -> tuple[AllocationWorkspaceSpec, FakeArtifactReader]:
    values: dict[tuple[str, str, str], ArtifactValue] = {}
    specs: list[AllocationWorkspaceSource] = []
    for name, target, payload in sources:
        ref = ArtifactRef(namespace="inputs", name=name, revision=REVISION)
        specs.append(AllocationWorkspaceSource(artifact=ref, target=target))
        values[(ref.namespace, ref.name, REVISION)] = ArtifactValue(
            artifact=ref,
            media_type="application/zip",
            data=payload,
            binding_created_at=NOW,
            revision_created_at=NOW,
        )
    return AllocationWorkspaceSpec(mode="direct", sources=specs), FakeArtifactReader(values)


def archive(members: dict[str, bytes | None]) -> bytes:
    infos: list[tuple[zipfile.ZipInfo, bytes]] = []
    for name, data in members.items():
        info = zipfile.ZipInfo(name)
        info.compress_type = zipfile.ZIP_DEFLATED
        if data is None:
            if not name.endswith("/"):
                info.filename += "/"
            info.external_attr = (stat.S_IFDIR | 0o755) << 16
            data = b""
        else:
            info.external_attr = (stat.S_IFREG | 0o644) << 16
        infos.append((info, data))
    return archive_infos(infos)


def archive_infos(infos: list[tuple[zipfile.ZipInfo, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for info, data in infos:
            bundle.writestr(info, data)
    return output.getvalue()


def settings(
    storage: str,
    root: Path | None = None,
    *,
    limits: WorkspaceLimits | None = None,
) -> WorkspaceSettings:
    return WorkspaceSettings(
        storage=storage,  # type: ignore[arg-type]
        work_root=root,
        limits=limits
        or WorkspaceLimits(
            max_files=100,
            max_expanded_bytes=8 * 1024 * 1024,
            max_managed_text_bytes=4 * 1024 * 1024,
            max_file_bytes=4 * 1024 * 1024,
        ),
    )
