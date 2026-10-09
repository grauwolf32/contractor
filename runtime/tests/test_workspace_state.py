from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import jcs
import pytest
from test_projectfs_zip import REVISION, archive, settings, workspace_inputs

from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import AllocationWorkspaceState, ArtifactRef
from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    LocalWorkspaceProvider,
    ManagedWorkspaceTree,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
    WorkspaceStateError,
    decode_workspace_state,
    encode_workspace_state,
    hydrate_workspace,
)
from contractor_runtime.projectfs import overlay as overlay_module
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings


def test_state_codec_is_canonical_cumulative_and_reconstructs_type_changes() -> None:
    source = source_tree()
    result = source.clone()
    result.text_files["src/a.py"] = "print('changed')\n"
    result.text_files.pop("remove.txt")
    result.directories.remove("old-dir")
    result.text_files.pop("old-dir/child.txt")
    result.text_files["old-dir"] = "directory became file\n"
    result.binary_paths.remove("image.bin")
    result.directories.add("new")
    result.text_files["new/file.txt"] = "new\n"

    encoded = encode_workspace_state(source, result)
    assert jcs.canonicalize(json.loads(encoded)) == encoded
    document = json.loads(encoded)
    assert document["apiVersion"] == "contractor.workspace/v1"
    assert document["kind"] == "WorkspaceOverlay"
    assert all("revision" not in operation for operation in document["operations"])
    reconstructed = decode_workspace_state(encoded, source, limits())
    assert reconstructed.snapshot() == result.snapshot()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(baseWorkspaceDigest="sha256:" + "0" * 64),
        lambda value: value.update(resultWorkspaceDigest="sha256:" + "0" * 64),
        lambda value: value["operations"][0].update(op="unknown"),
        lambda value: value["operations"][0].update(path="src/../escape"),
        lambda value: value.update(extra="not allowed"),
    ],
)
def test_state_rejects_wrong_digests_unknown_ops_paths_and_fields(mutation: object) -> None:
    source = source_tree()
    result = source.clone()
    result.text_files["src/a.py"] = "changed\n"
    document = json.loads(encode_workspace_state(source, result))
    mutation(document)  # type: ignore[operator]
    with pytest.raises(WorkspaceStateError, match="workspace_state_invalid"):
        decode_workspace_state(jcs.canonicalize(document), source, limits())


def test_state_rejects_noncanonical_json_duplicate_keys_nul_and_redundant_ops() -> None:
    source = source_tree()
    result = source.clone()
    result.text_files["src/a.py"] = "changed\n"
    canonical = encode_workspace_state(source, result)
    with pytest.raises(WorkspaceStateError):
        decode_workspace_state(b" " + canonical, source, limits())
    with pytest.raises(WorkspaceStateError):
        decode_workspace_state(b'{"apiVersion":"x","apiVersion":"y"}', source, limits())

    document = json.loads(canonical)
    document["operations"][0]["text"] = "bad\x00text"
    with pytest.raises(WorkspaceStateError):
        decode_workspace_state(jcs.canonicalize(document), source, limits())

    document = json.loads(canonical)
    document["operations"].append(dict(document["operations"][0]))
    with pytest.raises(WorkspaceStateError):
        decode_workspace_state(jcs.canonicalize(document), source, limits())


def test_imported_state_and_sequential_checkpoints_need_only_latest_state() -> None:
    async def scenario() -> None:
        source = source_tree()
        first = await overlay(source, "first")
        await first.write_text("src/a.py", "revision one\n")
        state_one = (await first.prepare_export()).state

        second = await overlay(source, "second")
        await second.import_state(state_one)
        assert tuple(entry.path for entry in await second.change_entries()) == ()
        await second.make_directory("generated")
        await second.write_text("generated/one.txt", "one\n")
        first_diff = await second.diff()
        assert "generated/one.txt" in first_diff.text
        assert "src/a.py" not in first_diff.text
        state_two = (await second.prepare_export()).state
        await second.commit_export(await second.prepare_export())

        await second.write_text("src/a.py", "revision three\n")
        second_diff = await second.diff()
        assert "src/a.py" in second_diff.text
        assert "generated/one.txt" not in second_diff.text
        latest = (await second.prepare_export()).state

        reconstructed = decode_workspace_state(latest, source, limits())
        assert reconstructed.snapshot() == await second.snapshot()
        assert (
            decode_workspace_state(state_two, source, limits()).text_files["generated/one.txt"]
            == "one\n"
        )
        assert all("revision" not in operation for operation in json.loads(latest)["operations"])

    asyncio.run(scenario())


def test_hydration_applies_exact_state_to_overlay_view_and_direct_copy(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [
                (
                    "source",
                    "",
                    archive({"src/a.py": b"source\n", "remove.txt": b"remove\n"}),
                )
            ]
        )
        spec.mode = "overlay"
        seed_provider = LocalWorkspaceProvider(settings("local", tmp_path / "seed"))
        seed = await hydrate_workspace(
            provider=seed_provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="seed",
            timeout_seconds=5,
        )
        assert isinstance(seed, OverlayWorkspaceSession)
        await seed.write_text("src/a.py", "imported\n")
        await seed.delete_path("remove.txt")
        state_payload = (await seed.prepare_export()).state
        await seed_provider.cleanup(seed.storage)

        state_ref = ArtifactRef(namespace="analysis", name="state", revision=REVISION)
        reader.values[(state_ref.namespace, state_ref.name, REVISION)] = ArtifactValue(
            artifact=state_ref,
            media_type="application/vnd.contractor.workspace-overlay+json",
            data=state_payload,
            binding_created_at=datetime(2026, 9, 1, tzinfo=UTC),
            revision_created_at=datetime(2026, 9, 1, tzinfo=UTC),
        )
        spec.state = AllocationWorkspaceState(artifact=state_ref)

        overlay_provider = LocalWorkspaceProvider(settings("local", tmp_path / "overlay"))
        imported = await hydrate_workspace(
            provider=overlay_provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="overlay",
            timeout_seconds=5,
        )
        assert await imported.read_text("src/a.py") == "imported\n"
        assert tuple(entry.path for entry in await imported.change_entries()) == ()  # type: ignore[attr-defined]
        assert not Path(f"{imported.storage.root}/run_workdir").exists()

        direct_spec = spec.model_copy(deep=True, update={"mode": "direct"})
        direct_provider = LocalWorkspaceProvider(settings("local", tmp_path / "direct"))
        direct = await hydrate_workspace(
            provider=direct_provider,
            spec=direct_spec,
            artifact_reader=reader,
            allocation_id="direct",
            timeout_seconds=5,
        )
        assert await direct.read_text("src/a.py") == "imported\n"
        assert Path(f"{direct.storage.root}/run_workdir/src/a.py").read_bytes() == b"imported\n"
        assert not Path(f"{direct.storage.root}/run_workdir/remove.txt").exists()
        # Imported state initializes local disk, not a retained direct overlay.
        Path(f"{direct.storage.root}/run_workdir/src/a.py").write_bytes(b"external\n")
        assert await direct.read_text("src/a.py") == "external\n"
        assert not direct._tree.paths()

        memory_provider = MemoryWorkspaceProvider(settings("memory"))
        memory = await hydrate_workspace(
            provider=memory_provider,
            spec=direct_spec,
            artifact_reader=reader,
            allocation_id="memory-direct",
            timeout_seconds=5,
        )
        assert await memory.read_text("src/a.py") == "imported\n"
        assert all(file.path != "remove.txt" for file in (await memory.snapshot()).files)
        assert not Path(f"{memory.storage.root}/run_workdir").exists()

        await overlay_provider.cleanup(imported.storage)
        await direct_provider.cleanup(direct.storage)
        await memory_provider.cleanup(memory.storage)

    asyncio.run(scenario())


def test_state_import_validates_the_result_once_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        source = source_tree()
        result = source.clone()
        result.text_files.pop("remove.txt")
        for index in range(60):
            result.directories.add(f"generated/{index % 6}")
            result.text_files[f"generated/{index % 6}/{index}.txt"] = f"{index}\n"
        result.directories.add("generated")
        payload = encode_workspace_state(source, result)
        calls: list[int] = []
        validate = overlay_module._validate_tree

        def counted(tree: ManagedWorkspaceTree, bounds: WorkspaceLimits) -> None:
            calls.append(threading.get_ident())
            validate(tree, bounds)

        monkeypatch.setattr(overlay_module, "_validate_tree", counted)
        session = await overlay(source, "import")
        await session.import_state(payload)
        assert await session.snapshot() == result.snapshot()
        assert len(calls) == 1 and calls[0] != threading.get_ident()

    asyncio.run(scenario())


def test_export_preparation_runs_off_loop_and_commit_uses_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session = await overlay(source_tree(), "export-thread")
        await session.write_text("src/a.py", "changed\n")
        calls: list[int] = []
        snapshot = ManagedWorkspaceTree.snapshot
        encode = overlay_module.encode_workspace_state
        diff = overlay_module._workspace_diff

        def counted_snapshot(tree: ManagedWorkspaceTree):
            calls.append(threading.get_ident())
            return snapshot(tree)

        def counted_encode(*args: object):
            calls.append(threading.get_ident())
            return encode(*args)

        def counted_diff(*args: object):
            calls.append(threading.get_ident())
            return diff(*args)

        monkeypatch.setattr(ManagedWorkspaceTree, "snapshot", counted_snapshot)
        monkeypatch.setattr(overlay_module, "encode_workspace_state", counted_encode)
        monkeypatch.setattr(overlay_module, "_workspace_diff", counted_diff)
        bundle = await session.prepare_export()
        assert len(calls) == 3 and all(call != threading.get_ident() for call in calls)

        def forbidden_snapshot(_: ManagedWorkspaceTree):
            raise AssertionError("commit_export must not rebuild a snapshot")

        monkeypatch.setattr(ManagedWorkspaceTree, "snapshot", forbidden_snapshot)
        assert await session.commit_export(bundle) == bundle.snapshot
        with pytest.raises(overlay_module.WorkspaceStorageError, match="workspace_export_stale"):
            await session.commit_export(bundle)

    asyncio.run(scenario())


def test_overlay_edits_validate_only_changed_text_and_keep_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        source = ManagedWorkspaceTree(
            directories={"dir"},
            text_files={f"dir/file-{index}.txt": "content\n" for index in range(40)},
        )
        session = await overlay(source, "edit-delta")
        calls: list[int] = []
        validate = overlay_module._validate_text

        def counted(text: str, bounds: WorkspaceLimits | None) -> bytes:
            calls.append(threading.get_ident())
            return validate(text, bounds)

        monkeypatch.setattr(overlay_module, "_validate_text", counted)
        await session.update_text("dir/file-0.txt", lambda value: value + "updated\n")
        await session.copy_path("dir/file-1.txt", "dir/copied.txt")
        await session.move_path("dir/file-2.txt", "dir/moved.txt")
        assert len(calls) == 3 and all(call != threading.get_ident() for call in calls)
        assert await session.read_text("dir/moved.txt") == "content\n"
        assert await session.read_text("dir/copied.txt") == "content\n"

        tight = WorkspaceLimits(
            max_files=3, max_expanded_bytes=12, max_managed_text_bytes=12, max_file_bytes=10
        )
        provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=tight))
        storage = await provider.create("edit-limits")
        bounded = OverlayWorkspaceSession(
            storage=storage,
            limits=tight,
            directories=set(),
            text_files={"a.txt": "12345", "b.txt": "67890"},
            binary_paths=set(),
        )
        with pytest.raises(overlay_module.WorkspaceStorageError, match="workspace_limit_exceeded"):
            await bounded.copy_path("a.txt", "c.txt")
        await bounded.move_path("a.txt", "c.txt")
        assert await bounded.read_text("c.txt") == "12345"

    asyncio.run(scenario())


def test_overlay_delete_and_mkdir_validate_only_changed_entries_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        source = ManagedWorkspaceTree(
            directories={"dir", "dir/nested"},
            text_files={f"dir/file-{index}.txt": "content\n" for index in range(40)}
            | {"dir/nested/deep.txt": "deep\n"},
        )
        session = await overlay(source, "path-delta")
        clones: list[int] = []
        clone = ManagedWorkspaceTree.clone

        def counted_clone(tree: ManagedWorkspaceTree) -> ManagedWorkspaceTree:
            clones.append(threading.get_ident())
            return clone(tree)

        def unexpected(*_: object) -> None:
            raise AssertionError("delete and mkdir must not validate unchanged entries")

        monkeypatch.setattr(ManagedWorkspaceTree, "clone", counted_clone)
        monkeypatch.setattr(overlay_module, "_validate_tree", unexpected)
        monkeypatch.setattr(overlay_module, "_validate_text", unexpected)
        await session.make_directory("dir/new/leaf", parents=True)
        await session.delete_path("dir/file-0.txt")
        await session.delete_path("dir/nested", recursive=True)
        await session.make_directory("dir")
        assert len(clones) == 3 and threading.get_ident() not in clones
        monkeypatch.undo()

        snapshot = await session.snapshot()
        assert snapshot.directories == ("dir", "dir/new", "dir/new/leaf")
        assert [item.path for item in snapshot.files] == sorted(
            f"dir/file-{index}.txt" for index in range(1, 40)
        )

    asyncio.run(scenario())


def test_overlay_delete_and_mkdir_keep_count_and_byte_limits() -> None:
    async def scenario() -> None:
        tight = WorkspaceLimits(
            max_files=4, max_expanded_bytes=12, max_managed_text_bytes=12, max_file_bytes=10
        )
        provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=tight))
        storage = await provider.create("path-limits")
        session = OverlayWorkspaceSession(
            storage=storage,
            limits=tight,
            directories=set(),
            text_files={"a.txt": "12345", "b.txt": "67890"},
            binary_paths=set(),
        )
        await session.make_directory("c")
        with pytest.raises(overlay_module.WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.make_directory("d/e", parents=True)
        await session.make_directory("d")
        with pytest.raises(overlay_module.WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("c/f.txt", "1")
        await session.delete_path("d")
        with pytest.raises(overlay_module.WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("c/f.txt", "123")
        # The deleted text's bytes are released for later writes.
        await session.delete_path("a.txt")
        await session.write_text("c/f.txt", "1234567")
        assert await session.read_text("c/f.txt") == "1234567"
        with pytest.raises(overlay_module.WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("b.txt", "678901")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("method", "path", "flag"),
    [
        ("delete_path", "missing.txt", False),
        ("delete_path", "tree", False),
        ("delete_path", "tree", True),
        ("delete_path", "tree/empty", False),
        ("delete_path", "blobs", True),
        ("delete_path", "image.bin", False),
        ("make_directory", "tree", False),
        ("make_directory", "file.txt", True),
        ("make_directory", "missing/child", False),
        ("make_directory", "missing/child", True),
        ("make_directory", "file.txt/child", True),
        ("make_directory", "tree/file.txt", False),
        ("make_directory", "tree/new", False),
        ("make_directory", "a/b/c/d", True),
    ],
)
def test_overlay_delete_and_mkdir_match_memory_direct_session(
    method: str, path: str, flag: bool
) -> None:
    async def scenario() -> None:
        tree = ManagedWorkspaceTree(
            directories={"tree", "tree/empty", "blobs"},
            text_files={"file.txt": "file\n", "tree/file.txt": "nested\n"},
            binary_paths={"image.bin", "blobs/blob.bin"},
        )
        bounds = WorkspaceLimits(
            max_files=9,
            max_expanded_bytes=1 << 20,
            max_managed_text_bytes=1 << 19,
            max_file_bytes=1 << 18,
        )
        provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=bounds))
        outcomes = []
        for session_type in (DirectWorkspaceSession, OverlayWorkspaceSession):
            storage = await provider.create(f"parity-{session_type.__name__}")
            arguments = dict(
                storage=storage,
                limits=bounds,
                directories=tree.directories,
                text_files=tree.text_files,
                binary_paths=tree.binary_paths,
            )
            session = (
                DirectWorkspaceSession(mode="direct", **arguments)
                if session_type is DirectWorkspaceSession
                else OverlayWorkspaceSession(**arguments)
            )
            keyword = "recursive" if method == "delete_path" else "parents"
            try:
                await getattr(session, method)(path, **{keyword: flag})
                outcome = None
            except overlay_module.WorkspaceStorageError as error:
                outcome = error.args[0]
            outcomes.append((outcome, await session.snapshot()))
        assert outcomes[0] == outcomes[1]

    asyncio.run(scenario())


def test_state_limits_bind_the_result_not_intermediate_operations() -> None:
    source = ManagedWorkspaceTree(text_files={"b.txt": "x" * 1000})
    bounds = WorkspaceLimits(
        max_files=10, max_expanded_bytes=1 << 20, max_managed_text_bytes=1010, max_file_bytes=1000
    )
    # Canonical writes are path-ordered: growing a.txt precedes shrinking b.txt.
    fits = source.clone()
    fits.text_files["a.txt"] = "y" * 1000
    fits.text_files["b.txt"] = "x"
    assert decode_workspace_state(
        encode_workspace_state(source, fits), source, bounds
    ).snapshot() == (fits.snapshot())
    too_large = source.clone()
    too_large.text_files["a.txt"] = "y" * 1000
    with pytest.raises(WorkspaceStateError, match="workspace_state_invalid"):
        decode_workspace_state(encode_workspace_state(source, too_large), source, bounds)


def test_state_import_scales_linearly_with_changed_paths() -> None:
    source = ManagedWorkspaceTree()
    for index in range(2000):
        source.directories.add(f"old/{index % 40}")
        source.text_files[f"old/{index % 40}/{index}.txt"] = f"{index}\n"
    source.directories.add("old")
    result = source.clone()
    for index in range(2000):
        if index % 2:
            result.text_files.pop(f"old/{index % 40}/{index}.txt")
        result.directories.add(f"new/{index % 40}")
        result.text_files[f"new/{index % 40}/{index}.txt"] = f"{index}\n"
    result.directories.add("new")
    bounds = WorkspaceLimits(
        max_files=10_000,
        max_expanded_bytes=1 << 24,
        max_managed_text_bytes=1 << 24,
        max_file_bytes=1 << 10,
    )
    payload = encode_workspace_state(source, result)
    started = time.monotonic()
    decoded = decode_workspace_state(payload, source, bounds)
    # Quadratic whole-tree validation took tens of seconds at this size.
    assert time.monotonic() - started < 5
    assert decoded.snapshot() == result.snapshot()


async def overlay(source: ManagedWorkspaceTree, name: str) -> OverlayWorkspaceSession:
    provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=limits()))
    storage = await provider.create(name)
    return OverlayWorkspaceSession(
        storage=storage,
        limits=limits(),
        directories=source.directories,
        text_files=source.text_files,
        binary_paths=source.binary_paths,
    )


def source_tree() -> ManagedWorkspaceTree:
    return ManagedWorkspaceTree(
        directories={"src", "old-dir"},
        text_files={
            "src/a.py": "print('source')\n",
            "remove.txt": "remove\n",
            "old-dir/child.txt": "child\n",
        },
        binary_paths={"image.bin"},
    )


def limits() -> WorkspaceLimits:
    return WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=1 << 20,
        max_managed_text_bytes=1 << 19,
        max_file_bytes=1 << 18,
    )
