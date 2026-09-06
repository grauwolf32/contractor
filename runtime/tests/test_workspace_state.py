from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import jcs
import pytest
from test_projectfs_zip import REVISION, archive, settings, workspace_inputs

from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import AllocationWorkspaceStateV2, ArtifactRef
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    ManagedWorkspaceTree,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
    WorkspaceStateError,
    decode_workspace_state,
    encode_workspace_state,
    hydrate_workspace,
)
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
        state_one = await first.export_state()

        second = await overlay(source, "second")
        await second.import_state(state_one)
        assert await second.changed_paths() == ()
        await second.make_directory("generated")
        await second.write_text("generated/one.txt", "one\n")
        first_diff = await second.diff()
        assert "generated/one.txt" in first_diff.text
        assert "src/a.py" not in first_diff.text
        state_two = await second.export_state()
        await second.commit_checkpoint()

        await second.write_text("src/a.py", "revision three\n")
        second_diff = await second.diff()
        assert "src/a.py" in second_diff.text
        assert "generated/one.txt" not in second_diff.text
        latest = await second.export_state()

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
        state_payload = await seed.export_state()
        await seed_provider.cleanup(seed.storage)

        state_ref = ArtifactRef(namespace="analysis", name="state", revision=REVISION)
        reader.values[(state_ref.namespace, state_ref.name, REVISION)] = ArtifactValue(
            artifact=state_ref,
            media_type="application/vnd.contractor.workspace-overlay+json",
            data=state_payload,
            binding_created_at=datetime(2026, 9, 1, tzinfo=UTC),
            revision_created_at=datetime(2026, 9, 1, tzinfo=UTC),
        )
        spec.state = AllocationWorkspaceStateV2(artifact=state_ref)

        overlay_provider = LocalWorkspaceProvider(settings("local", tmp_path / "overlay"))
        imported = await hydrate_workspace(
            provider=overlay_provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="overlay",
            timeout_seconds=5,
        )
        assert await imported.read_text("src/a.py") == "imported\n"
        assert await imported.changed_paths() == ()  # type: ignore[attr-defined]
        assert (
            imported.storage.filesystem.cat(f"{imported.storage.root}/run_workdir/src/a.py")
            == b"source\n"
        )
        assert imported.storage.filesystem.exists(f"{imported.storage.root}/run_workdir/remove.txt")

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
        assert (
            direct.storage.filesystem.cat(f"{direct.storage.root}/run_workdir/src/a.py")
            == b"imported\n"
        )
        assert not direct.storage.filesystem.exists(f"{direct.storage.root}/run_workdir/remove.txt")
        # Imported state initializes local disk, not a retained direct overlay.
        direct.storage.filesystem.pipe(f"{direct.storage.root}/run_workdir/src/a.py", b"external\n")
        assert await direct.read_text("src/a.py") == "external\n"
        assert not direct._tree.paths()

        await overlay_provider.cleanup(imported.storage)
        await direct_provider.cleanup(direct.storage)

    asyncio.run(scenario())


async def overlay(source: ManagedWorkspaceTree, name: str) -> OverlayWorkspaceSession:
    provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=limits()))
    storage = await provider.create(name)
    return OverlayWorkspaceSession(
        storage=storage,
        content_root=f"{storage.root}/run_workdir",
        limits=limits(),
        directories=source.directories,
        text_files=source.text_files,
        binary_paths=source.binary_paths,
        stored_binary_paths=source.stored_binary_paths,
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
        stored_binary_paths={"image.bin"},
    )


def limits() -> WorkspaceLimits:
    return WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=1 << 20,
        max_managed_text_bytes=1 << 19,
        max_file_bytes=1 << 18,
    )
