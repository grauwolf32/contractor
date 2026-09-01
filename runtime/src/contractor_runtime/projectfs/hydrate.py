"""Bounded exact-artifact ZIP hydration for project workspaces."""

from __future__ import annotations

import asyncio
import codecs
import io
import stat
import time
import zipfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol

from contractor_runtime.artifacts import ArtifactClientError, ArtifactValue
from contractor_runtime.contracts import AllocationWorkspaceSpecV2, ArtifactRef
from contractor_runtime.projectfs.overlay import (
    WORKSPACE_OVERLAY_MEDIA_TYPE,
    OverlayWorkspaceSession,
    WorkspaceStateError,
    canonical_overlay_operations,
    decode_workspace_state,
)
from contractor_runtime.projectfs.paths import (
    ProjectPathError,
    join_project_path,
    normalize_project_path,
    parent_paths,
)
from contractor_runtime.projectfs.provider import ProjectWorkspaceStorage, WorkspaceProvider
from contractor_runtime.projectfs.storage import (
    DirectWorkspaceSession,
    ManagedWorkspaceTree,
    WorkspaceStorageError,
)

WORKSPACE_SOURCE_MEDIA_TYPE = "application/zip"
_CHUNK_BYTES = 64 * 1024
_MAX_COMPRESSION_RATIO = 1000
_RATIO_FLOOR_BYTES = 1 << 20


class ArtifactReader(Protocol):
    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue: ...


class WorkspacePreparationError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        status_code: int,
        cleanup_confirmed: bool = True,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.cleanup_confirmed = cleanup_confirmed


@dataclass(slots=True)
class _TreeAccumulator:
    storage: ProjectWorkspaceStorage = field(repr=False)
    content_root: str = field(repr=False)
    max_files: int
    max_expanded_bytes: int
    max_managed_text_bytes: int
    max_file_bytes: int
    directories: set[str] = field(default_factory=set)
    text_files: dict[str, str] = field(default_factory=dict, repr=False)
    binary_paths: set[str] = field(default_factory=set)
    stored_binary_paths: set[str] = field(default_factory=set)
    path_types: dict[str, str] = field(default_factory=dict)
    files: int = 0
    entries: int = 0
    expanded_bytes: int = 0
    managed_text_bytes: int = 0


async def hydrate_workspace(
    *,
    provider: WorkspaceProvider,
    spec: AllocationWorkspaceSpecV2,
    artifact_reader: ArtifactReader,
    allocation_id: str,
    timeout_seconds: float,
) -> DirectWorkspaceSession:
    if timeout_seconds <= 0:
        raise WorkspacePreparationError(
            "workspace_capacity_exceeded", retryable=True, status_code=503
        )
    try:
        storage = await provider.create(allocation_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise WorkspacePreparationError(
            "workspace_capacity_exceeded", retryable=True, status_code=503
        ) from None

    content_root = f"{storage.root.rstrip('/')}/run_workdir"
    limits = provider.capability.limits
    accumulator = _TreeAccumulator(
        storage=storage,
        content_root=content_root,
        max_files=limits.max_files,
        max_expanded_bytes=limits.max_expanded_bytes,
        max_managed_text_bytes=limits.max_managed_text_bytes,
        max_file_bytes=limits.max_file_bytes,
    )
    deadline = time.monotonic() + timeout_seconds
    try:
        storage.filesystem.makedirs(content_root, exist_ok=False)
        for source in spec.sources:
            if time.monotonic() >= deadline:
                raise _capacity()
            value = await _read_artifact(artifact_reader, source.artifact, deadline)
            if value.artifact != source.artifact or value.media_type != WORKSPACE_SOURCE_MEDIA_TYPE:
                raise _invalid_source()
            await _blocking_cancellation_safe(
                lambda value=value, target=source.target: _extract_archive(
                    value.data, target, accumulator, deadline
                )
            )
        arguments = dict(
            storage=storage,
            content_root=content_root,
            limits=limits,
            directories=accumulator.directories,
            text_files=accumulator.text_files,
            binary_paths=accumulator.binary_paths,
            stored_binary_paths=accumulator.stored_binary_paths,
        )
        if spec.mode == "overlay":
            overlay = OverlayWorkspaceSession(**arguments)
            if spec.state is not None:
                value = await _read_artifact(artifact_reader, spec.state.artifact, deadline)
                if (
                    value.artifact != spec.state.artifact
                    or value.media_type != WORKSPACE_OVERLAY_MEDIA_TYPE
                ):
                    raise _invalid_state()
                try:
                    await overlay.import_state(value.data)
                except (WorkspaceStateError, WorkspaceStorageError):
                    raise _invalid_state() from None
            return overlay

        session = DirectWorkspaceSession(mode="direct", **arguments)
        if spec.state is not None:
            value = await _read_artifact(artifact_reader, spec.state.artifact, deadline)
            if (
                value.artifact != spec.state.artifact
                or value.media_type != WORKSPACE_OVERLAY_MEDIA_TYPE
            ):
                raise _invalid_state()
            source_tree = session._source_tree()
            try:
                result_tree = decode_workspace_state(value.data, source_tree, limits)
                try:
                    await _blocking_cancellation_safe(
                        lambda: _materialize_state(storage, content_root, source_tree, result_tree)
                    )
                except OSError:
                    raise _capacity() from None
            except (WorkspaceStateError, WorkspaceStorageError):
                raise _invalid_state() from None
            session._tree = result_tree
        return session
    except BaseException:
        try:
            await provider.cleanup(storage)
        except Exception:
            raise WorkspacePreparationError(
                "workspace_cleanup_unconfirmed",
                retryable=False,
                status_code=503,
                cleanup_confirmed=False,
            ) from None
        raise


async def _blocking_cancellation_safe(operation: Callable[[], None]) -> None:
    task = asyncio.create_task(asyncio.to_thread(operation), name="workspace-zip-hydration")
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


async def _read_artifact(
    reader: ArtifactReader, ref: ArtifactRef, deadline: float
) -> ArtifactValue:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _capacity()
    try:
        async with asyncio.timeout(remaining):
            value = await reader.read_artifact(ref)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        raise _capacity() from None
    except ArtifactClientError as error:
        raise WorkspacePreparationError(
            "workspace_source_unavailable",
            retryable=bool(getattr(error, "retryable", True)),
            status_code=503,
        ) from None
    except Exception:
        raise WorkspacePreparationError(
            "workspace_source_unavailable", retryable=True, status_code=503
        ) from None
    return value


def _extract_archive(
    payload: bytes,
    target: str,
    tree: _TreeAccumulator,
    deadline: float,
) -> None:
    try:
        normalized_target = normalize_project_path(target, allow_root=True)
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if tree.entries + len(infos) > tree.max_files:
                raise _capacity()
            planned = _preflight_archive(infos, normalized_target, tree, deadline)
            _declare_target(normalized_target, tree)
            for info, relative, kind in planned:
                _check_deadline(deadline)
                if kind == "directory":
                    _make_directory(relative, tree)
                else:
                    _extract_file(archive, info, relative, tree, deadline)
    except WorkspacePreparationError:
        raise
    except (ProjectPathError, zipfile.BadZipFile, NotImplementedError, EOFError, UnicodeError):
        raise _invalid_source() from None
    except OSError:
        raise _capacity() from None


def _preflight_archive(
    infos: list[zipfile.ZipInfo],
    target: str,
    tree: _TreeAccumulator,
    deadline: float,
) -> list[tuple[zipfile.ZipInfo, str, str]]:
    result: list[tuple[zipfile.ZipInfo, str, str]] = []
    seen_members: set[str] = set()
    declared_expanded = tree.expanded_bytes
    declared_files = tree.files
    for info in infos:
        _check_deadline(deadline)
        if info.flag_bits & 0x1 or not info.filename:
            raise _invalid_source()
        directory = info.is_dir()
        raw = info.filename[:-1] if directory and info.filename.endswith("/") else info.filename
        member = normalize_project_path(raw, allow_root=False)
        if member in seen_members:
            raise _invalid_source()
        seen_members.add(member)
        relative = join_project_path(target, member) if target else member
        kind = _zip_kind(info, directory)
        if kind == "file":
            declared_files += 1
            declared_expanded += info.file_size
            if (
                declared_files > tree.max_files
                or info.file_size > tree.max_file_bytes
                or declared_expanded > tree.max_expanded_bytes
                or info.file_size
                > max(_RATIO_FLOOR_BYTES, max(1, info.compress_size) * _MAX_COMPRESSION_RATIO)
            ):
                raise _capacity()
        _preflight_path(relative, kind, tree)
        result.append((info, relative, kind))
    tree.entries += len(infos)
    tree.files = declared_files
    tree.expanded_bytes = declared_expanded
    return result


def _zip_kind(info: zipfile.ZipInfo, directory: bool) -> str:
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if directory:
        if file_type not in {0, stat.S_IFDIR}:
            raise _invalid_source()
        return "directory"
    if file_type not in {0, stat.S_IFREG}:
        raise _invalid_source()
    return "file"


def _preflight_path(path: str, kind: str, tree: _TreeAccumulator) -> None:
    for parent in parent_paths(path):
        existing = tree.path_types.get(parent)
        if existing == "file":
            raise _invalid_source()
        tree.path_types[parent] = "directory"
    existing = tree.path_types.get(path)
    if existing is not None and existing != kind:
        raise _invalid_source()
    if existing == "file" and kind == "file":
        raise _invalid_source()
    tree.path_types[path] = kind
    # maxFiles bounds the complete managed tree, including implicit parent
    # directories. Counting only ZIP members permits a deeply nested archive to
    # hydrate a tree much larger than the capability advertised at registration.
    if len(tree.path_types) > tree.max_files:
        raise _capacity()


def _declare_target(target: str, tree: _TreeAccumulator) -> None:
    if not target:
        return
    for directory in (*parent_paths(target), target):
        _make_directory(directory, tree)


def _make_directory(path: str, tree: _TreeAccumulator) -> None:
    tree.directories.add(path)
    tree.path_types[path] = "directory"
    if len(tree.path_types) > tree.max_files:
        raise _capacity()
    tree.storage.filesystem.makedirs(_backend_path(tree.content_root, path), exist_ok=True)


def _extract_file(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    path: str,
    tree: _TreeAccumulator,
    deadline: float,
) -> None:
    for parent in parent_paths(path):
        _make_directory(parent, tree)
    local_output = None
    backend_path = _backend_path(tree.content_root, path)
    if tree.storage.storage == "local":
        local_output = tree.storage.filesystem.open(backend_path, mode="wb")
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    text_parts: list[str] = []
    text_candidate = True
    observed = 0
    try:
        with archive.open(info, mode="r") as source:
            while True:
                _check_deadline(deadline)
                chunk = source.read(_CHUNK_BYTES)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > info.file_size or observed > tree.max_file_bytes:
                    raise _invalid_source()
                if local_output is not None:
                    local_output.write(chunk)
                if text_candidate:
                    if b"\x00" in chunk:
                        text_candidate = False
                        text_parts.clear()
                    else:
                        try:
                            text_parts.append(decoder.decode(chunk, final=False))
                        except UnicodeDecodeError:
                            text_candidate = False
                            text_parts.clear()
        if observed != info.file_size:
            raise _invalid_source()
        if text_candidate:
            try:
                text_parts.append(decoder.decode(b"", final=True))
            except UnicodeDecodeError:
                text_candidate = False
                text_parts.clear()
        if text_candidate:
            text = "".join(text_parts)
            tree.managed_text_bytes += observed
            if tree.managed_text_bytes > tree.max_managed_text_bytes:
                raise _capacity()
            tree.text_files[path] = text
            if local_output is None:
                with tree.storage.filesystem.open(backend_path, mode="wb") as destination:
                    destination.write(text.encode("utf-8"))
        else:
            tree.binary_paths.add(path)
            if local_output is not None:
                tree.stored_binary_paths.add(path)
    finally:
        if local_output is not None:
            local_output.close()


def _backend_path(root: str, relative: str) -> str:
    return f"{root.rstrip('/')}/{relative}"


def _materialize_state(
    storage: ProjectWorkspaceStorage,
    content_root: str,
    source: ManagedWorkspaceTree,
    result: ManagedWorkspaceTree,
) -> None:
    filesystem = storage.filesystem
    for operation in canonical_overlay_operations(source, result):
        target = _backend_path(content_root, operation.path)
        if operation.op == "delete_path":
            if filesystem.exists(target):
                filesystem.rm(target, recursive=True)
        elif operation.op == "create_directory":
            filesystem.makedirs(target, exist_ok=False)
        else:
            assert operation.text is not None
            with filesystem.open(target, mode="wb") as destination:
                destination.write(operation.text.encode("utf-8"))


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise _capacity()


def _invalid_source() -> WorkspacePreparationError:
    return WorkspacePreparationError("workspace_source_invalid", retryable=False, status_code=422)


def _capacity() -> WorkspacePreparationError:
    return WorkspacePreparationError("workspace_capacity_exceeded", retryable=True, status_code=503)


def _invalid_state() -> WorkspacePreparationError:
    return WorkspacePreparationError("workspace_state_invalid", retryable=False, status_code=422)
