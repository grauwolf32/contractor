"""Selected artifact-backed Worker tools for one logical Memory Namespace."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactTransportError,
    ArtifactValue,
    ArtifactWriteValue,
)
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.memory import (
    ARTIFACT_NAME_PREFIX,
    MAXIMUM_EXACT_ORDINAL,
    MEDIA_TYPE,
    REASON_PAYLOAD_TOO_LARGE,
    SCHEMA_VERSION,
    MemoryCodecError,
    StoredMemoryNote,
    artifact_name,
    decode_note,
    encode_note,
    full_projection,
    normalize_note,
    preview_projection,
)
from contractor_runtime.toolsets.artifact_visibility import PURPOSE_RESERVED_NAMESPACES
from contractor_runtime.toolsets.run_artifacts import ArtifactClientFactory, ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAXIMUM_NOTES = 128

_MEMORY_ERROR_RETRYABILITY = MappingProxyType(
    {
        "memory_invalid": False,
        "memory_not_found": False,
        "memory_too_large": False,
        "memory_namespace_full": False,
        "memory_changed": True,
        "memory_forbidden": False,
        "memory_unavailable": True,
    }
)


class MemoryToolError(RuntimeError):
    """Bounded model-facing failure with no note body, description or tags."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        del retryable
        normalized = code if code in _MEMORY_ERROR_RETRYABILITY else "memory_unavailable"
        self.code = normalized
        self.retryable = _MEMORY_ERROR_RETRYABILITY[normalized]
        super().__init__(f"Memory operation failed ({normalized})")


class MemoryToolsetFactory:
    ref = "memory-tools@1"
    exported_tools = frozenset(
        {
            "list_memories",
            "read_memory",
            "write_memory",
            "append_memory",
            "search_memory",
            "list_memory_tags",
        }
    )
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

    async def probe(self) -> frozenset[str]:
        # The implementation needs only the already configured private Artifact
        # client. Capability discovery must not perform allocation network I/O.
        return self.exported_tools

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, adapter_handles
        if namespace in PURPOSE_RESERVED_NAMESPACES:
            raise ValueError("MemoryTools requires a non-purpose Agent Namespace")
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("memory-tools@1 requires State.metrics")
        session = _MemorySession(self._client_factory(allocation_id, runtime_settings), namespace)
        builders: dict[str, Callable[[], Any]] = {
            "list_memories": lambda: ListMemoriesTool(session, metrics),
            "read_memory": lambda: ReadMemoryTool(session, metrics),
            "write_memory": lambda: WriteMemoryTool(session, metrics),
            "append_memory": lambda: AppendMemoryTool(session, metrics),
            "search_memory": lambda: SearchMemoryTool(session, metrics),
            "list_memory_tags": lambda: ListMemoryTagsTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


@dataclass(frozen=True, slots=True)
class _LoadedNote:
    note: StoredMemoryNote
    binding_created_at: datetime
    revision_created_at: datetime
    revision: str
    payload: bytes


class _MemorySession:
    def __init__(self, client: ArtifactClient, namespace: str) -> None:
        self._client = client
        self._namespace = namespace
        self._lock = asyncio.Lock()

    async def list_memories(self) -> list[dict[str, Any]]:
        async with self._lock:
            notes = await self._load_all()
            return [_preview(note) for note in _ordered(notes)]

    async def read_memory(self, name: str) -> dict[str, Any]:
        binding_name = _validated_artifact_name(name)
        async with self._lock:
            loaded = await self._read_required(binding_name)
            return _full(loaded)

    async def write_memory(
        self,
        name: str,
        content: str,
        description: str,
        tags: Sequence[str],
    ) -> dict[str, Any]:
        binding_name = _validated_artifact_name(name)
        # Reject all caller-controlled fields before an Artifact mutation. The
        # final payload is checked again after the trusted ordinal is known.
        _encode_input(name, content, description, tags, 0)
        async with self._lock:
            existing = await self._read_optional(binding_name)
            if existing is None:
                current = await self._load_all()
                existing = next((item for item in current if item.note.name == name), None)
                if existing is None:
                    if len(current) >= MAXIMUM_NOTES:
                        raise MemoryToolError("memory_namespace_full")
                    ordinal = max((item.note.ordinal for item in current), default=-1) + 1
                    if ordinal > MAXIMUM_EXACT_ORDINAL:
                        raise MemoryToolError("memory_namespace_full")
                    expected_revision = None
                else:
                    ordinal = existing.note.ordinal
                    expected_revision = existing.revision
            else:
                ordinal = existing.note.ordinal
                expected_revision = existing.revision
            note, payload = _encode_input(name, content, description, tags, ordinal)
            metadata = await self._write_exact(
                binding_name, payload, expected_revision=expected_revision
            )
            return _full_from_metadata(note, metadata)

    async def append_memory(self, name: str, content: str) -> dict[str, Any]:
        binding_name = _validated_artifact_name(name)
        _validate_append_content(content)
        async with self._lock:
            existing = await self._read_required(binding_name)
            note, payload = _encode_input(
                existing.note.name,
                existing.note.content + "\n" + content,
                existing.note.description,
                existing.note.tags,
                existing.note.ordinal,
            )
            metadata = await self._write_exact(
                binding_name, payload, expected_revision=existing.revision
            )
            return _full_from_metadata(note, metadata)

    async def search_memory(self, tags: Sequence[str]) -> list[dict[str, Any]]:
        normalized = _validate_search_tags(tags)
        async with self._lock:
            notes = await self._load_all()
            selected = [item for item in notes if normalized.intersection(item.note.tags)]
            return [_preview(note) for note in _ordered(selected)]

    async def list_memory_tags(self) -> list[str]:
        async with self._lock:
            notes = await self._load_all()
            return sorted({tag for item in notes for tag in item.note.tags})

    async def _load_all(self) -> list[_LoadedNote]:
        try:
            refs = await self._client.list_artifacts(
                self._namespace, name_prefix=ARTIFACT_NAME_PREFIX, limit=MAXIMUM_NOTES + 1
            )
        except (ArtifactTransportError, ArtifactAPIError) as error:
            raise _mapped_client_error(error) from None
        memory_refs = sorted(
            (ref for ref in refs if ref.name.startswith(ARTIFACT_NAME_PREFIX)),
            key=lambda ref: (ref.namespace, ref.name),
        )
        if len(memory_refs) > MAXIMUM_NOTES:
            raise MemoryToolError("memory_unavailable", retryable=True)
        result: list[_LoadedNote] = []
        seen: set[str] = set()
        for ref in memory_refs:
            if ref.namespace != self._namespace or ref.revision is not None or ref.name in seen:
                raise MemoryToolError("memory_unavailable", retryable=True)
            seen.add(ref.name)
            try:
                value = await self._client.read_artifact(ref)
            except (ArtifactTransportError, ArtifactAPIError) as error:
                raise _mapped_client_error(error) from None
            result.append(_decode_value(self._namespace, ref.name, value))
        if not _valid_ordinal_set(result):
            raise MemoryToolError("memory_unavailable")
        return result

    async def _read_optional(self, binding_name: str) -> _LoadedNote | None:
        try:
            value = await self._client.read_artifact(
                ArtifactRef(namespace=self._namespace, name=binding_name)
            )
        except ArtifactAPIError as error:
            if error.code == "artifact_not_found":
                return None
            raise _mapped_client_error(error) from None
        except ArtifactTransportError as error:
            raise _mapped_client_error(error) from None
        return _decode_value(self._namespace, binding_name, value)

    async def _read_required(self, binding_name: str) -> _LoadedNote:
        value = await self._read_optional(binding_name)
        if value is None:
            raise MemoryToolError("memory_not_found")
        return value

    async def _write_exact(
        self,
        binding_name: str,
        payload: bytes,
        *,
        expected_revision: str | None,
    ) -> ArtifactWriteValue | ArtifactValue:
        target = ArtifactRef(namespace=self._namespace, name=binding_name)
        try:
            return await self._client.write_artifact(
                target,
                data=payload,
                media_type=MEDIA_TYPE,
                expected_revision=expected_revision,
            )
        except ArtifactTransportError:
            pass
        except ArtifactAPIError as error:
            if error.code == "artifact_conflict":
                raise MemoryToolError("memory_changed", retryable=True) from None
            mapped = _mapped_client_error(error)
            if mapped.code == "memory_forbidden" or not 500 <= error.status_code < 600:
                raise mapped from None

        # Transport loss and server failures may follow a committed PUT (for
        # example, a lost database acknowledgement mapped to HTTP 500). Replay
        # only the same immutable bytes with the same precondition, once.
        try:
            return await self._client.write_artifact(
                target,
                data=payload,
                media_type=MEDIA_TYPE,
                expected_revision=expected_revision,
            )
        except (ArtifactTransportError, ArtifactAPIError):
            pass

        try:
            current = await self._client.read_artifact(target)
        except (ArtifactTransportError, ArtifactAPIError) as read_error:
            # The reconciliation read is authoritative for whether the current
            # value could be inspected. A retry conflict alone cannot prove
            # memory_changed when that read was forbidden or unavailable.
            raise _mapped_client_error(read_error) from None
        if current.media_type == MEDIA_TYPE and current.data == payload:
            return current
        raise MemoryToolError("memory_changed", retryable=True)


class _BaseMemoryTool:
    name: str
    description: str

    def __init__(self, session: _MemorySession, metrics: ToolMetrics) -> None:
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        return None

    def contractor_raw_argument_error(self, args: object) -> MemoryToolError | None:
        """Validate original ADK arguments before FunctionTool filters them."""

        if _valid_raw_arguments(self.name, args):
            return None
        error = MemoryToolError("memory_invalid")
        self._failure(_raw_argument_metric(self.name, args), error, time.perf_counter_ns())
        return error

    def _success(
        self, arguments: Mapping[str, Any], result: Mapping[str, Any], started_ns: int
    ) -> None:
        diagnostics = dict(arguments)
        for name in ("count", "content_bytes", "description_bytes", "tag_count"):
            if name in result:
                diagnostics[f"result_{name}"] = result[name]
        self._metrics.record_tool_call(
            self.name,
            arguments=diagnostics,
            result=result,
            duration_ms=_elapsed_ms(started_ns),
        )

    def _failure(self, arguments: Mapping[str, Any], error: Exception, started_ns: int) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments=arguments,
            error=error,
            duration_ms=_elapsed_ms(started_ns),
        )


class ListMemoriesTool(_BaseMemoryTool):
    name = "list_memories"
    description = """List shared memory note previews, most recently updated first.

    Use this to discover exact names before reading or writing notes.

    Returns:
        Note metadata and descriptions without content. Use read_memory for a body.
    """

    async def __call__(self) -> list[dict[str, Any]]:
        started = time.perf_counter_ns()
        try:
            result = await self._session.list_memories()
            self._success({}, {"count": len(result)}, started)
            return result
        except Exception as error:
            bounded = _normalize_tool_error(error)
            self._failure({}, bounded, started)
            raise bounded from None


class ReadMemoryTool(_BaseMemoryTool):
    name = "read_memory"
    description = """Read a shared memory note by its exact logical name.

    Use list_memories or search_memory to discover names. Reuse content already
    read until the note changes.

    Args:
        name: Exact logical note name in the Worker's shared memory namespace.

    Returns:
        The full note, including content, description, tags, ordinal, created_at
        and updated_at.
    """

    async def __call__(self, name: str) -> dict[str, Any]:
        started = time.perf_counter_ns()
        arguments = {"name": _safe_metric_name(name)}
        try:
            result = await self._session.read_memory(name)
            self._success(arguments, _note_metric(result), started)
            return result
        except Exception as error:
            bounded = _normalize_tool_error(error)
            self._failure(arguments, bounded, started)
            raise bounded from None


class WriteMemoryTool(_BaseMemoryTool):
    name = "write_memory"
    description = """Create or replace a shared memory note.

    Replacement overwrites content, description and tags. Use append_memory to
    extend a note while preserving metadata. Check existing notes to avoid duplicates.
    The original creation time and ordinal are preserved.

    Args:
        name: Stable lowercase snake_case note name in the shared memory namespace.
        content: Non-empty note text; the complete encoded note is limited to 32 KiB.
        description: Preview summary, at most 512 UTF-8 bytes; defaults to empty.
        tags: Up to 3 lowercase tags; defaults to an empty list. Use list_memory_tags
            to reuse existing tags. Tags may contain letters, digits, _ and -.

    Returns:
        The full stored note with tags, description, ordinal, created_at and updated_at.
    """

    async def __call__(
        self,
        name: str,
        content: str,
        description: str = "",
        tags: list[str] = [],  # noqa: B006 - schema requires an optional array default
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        arguments = _mutation_metric(name, content, description, tags)
        try:
            result = await self._session.write_memory(name, content, description, tags)
            self._success(arguments, _note_metric(result), started)
            return result
        except Exception as error:
            bounded = _normalize_tool_error(error)
            self._failure(arguments, bounded, started)
            raise bounded from None


class AppendMemoryTool(_BaseMemoryTool):
    name = "append_memory"
    description = """Append a newline and text to an existing shared memory note.

    Preserves the note description, tags, creation time and ordinal.

    Args:
        name: Exact logical name of an existing note.
        content: Non-empty text to append after a newline.

    Returns:
        The full updated note with tags, description, ordinal, created_at and updated_at.
    """

    async def __call__(self, name: str, content: str) -> dict[str, Any]:
        started = time.perf_counter_ns()
        arguments = {
            "name": _safe_metric_name(name),
            "content_bytes": _utf8_size(content),
        }
        try:
            result = await self._session.append_memory(name, content)
            self._success(arguments, _note_metric(result), started)
            return result
        except Exception as error:
            bounded = _normalize_tool_error(error)
            self._failure(arguments, bounded, started)
            raise bounded from None


class SearchMemoryTool(_BaseMemoryTool):
    name = "search_memory"
    description = """Find shared memory note previews matching any supplied tag.

    Use read_memory only for matches whose full content is needed.

    Args:
        tags: Tags to match with OR semantics and exact spelling and case.

    Returns:
        Matching previews, most recently updated first, without note content.
    """

    async def __call__(self, tags: list[str]) -> list[dict[str, Any]]:
        started = time.perf_counter_ns()
        arguments = {"tag_count": len(tags) if isinstance(tags, list) else 0}
        try:
            result = await self._session.search_memory(tags)
            self._success(arguments, {"count": len(result)}, started)
            return result
        except Exception as error:
            bounded = _normalize_tool_error(error)
            self._failure(arguments, bounded, started)
            raise bounded from None


class ListMemoryTagsTool(_BaseMemoryTool):
    name = "list_memory_tags"
    description = """List unique tags currently used by shared memory notes.

    Returns:
        Tag strings in lexical order for reuse in search_memory or write_memory.
    """

    async def __call__(self) -> list[str]:
        started = time.perf_counter_ns()
        try:
            result = await self._session.list_memory_tags()
            self._success({}, {"count": len(result)}, started)
            return result
        except Exception as error:
            bounded = _normalize_tool_error(error)
            self._failure({}, bounded, started)
            raise bounded from None


def _decode_value(namespace: str, binding_name: str, value: ArtifactValue) -> _LoadedNote:
    if (
        value.artifact.namespace != namespace
        or value.artifact.name != binding_name
        or value.artifact.revision is None
        or value.media_type != MEDIA_TYPE
        or not _valid_server_time(value.binding_created_at)
        or not _valid_server_time(value.revision_created_at)
        or value.revision_created_at < value.binding_created_at
    ):
        raise MemoryToolError("memory_unavailable", retryable=True)
    try:
        note = decode_note(binding_name, value.data)
    except MemoryCodecError:
        raise MemoryToolError("memory_unavailable", retryable=True) from None
    revision = value.artifact.revision
    assert revision is not None
    return _LoadedNote(
        note=note,
        binding_created_at=value.binding_created_at,
        revision_created_at=value.revision_created_at,
        revision=revision,
        payload=value.data,
    )


def _encode_input(
    name: str,
    content: str,
    description: str,
    tags: Sequence[str],
    ordinal: int,
) -> tuple[StoredMemoryNote, bytes]:
    if not isinstance(tags, (list, tuple)):
        raise MemoryToolError("memory_invalid")
    try:
        note = normalize_note(
            StoredMemoryNote(
                schema_version=SCHEMA_VERSION,
                name=name,
                content=content,
                description=description,
                tags=tuple(tags),
                ordinal=ordinal,
            )
        )
        return note, encode_note(note)
    except MemoryCodecError as error:
        reason = (
            "memory_too_large" if error.reason == REASON_PAYLOAD_TOO_LARGE else "memory_invalid"
        )
        raise MemoryToolError(reason) from None
    except TypeError:
        raise MemoryToolError("memory_invalid") from None


def _validated_artifact_name(name: str) -> str:
    try:
        return artifact_name(name)
    except MemoryCodecError:
        raise MemoryToolError("memory_invalid") from None


def _validate_append_content(content: str) -> None:
    if type(content) is not str or not content:
        raise MemoryToolError("memory_invalid")
    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        raise MemoryToolError("memory_invalid") from None


def _validate_search_tags(tags: Sequence[str]) -> set[str]:
    if not isinstance(tags, (list, tuple)):
        raise MemoryToolError("memory_invalid") from None
    values = tuple(tags)
    if not values:
        raise MemoryToolError("memory_invalid")
    try:
        note = normalize_note(
            StoredMemoryNote(
                schema_version=SCHEMA_VERSION,
                name="search_validation",
                content="x",
                description="",
                tags=values,
                ordinal=0,
            )
        )
    except MemoryCodecError:
        raise MemoryToolError("memory_invalid") from None
    if len(note.tags) != len(values):
        raise MemoryToolError("memory_invalid")
    return set(note.tags)


def _ordered(notes: Sequence[_LoadedNote]) -> list[_LoadedNote]:
    # Stable sorts express the exact mixed-direction total order without
    # converting server timestamps to lossy numeric values.
    result = sorted(notes, key=lambda item: item.note.name)
    result.sort(key=lambda item: item.note.ordinal, reverse=True)
    result.sort(key=lambda item: item.revision_created_at, reverse=True)
    return result


def _valid_ordinal_set(notes: Sequence[_LoadedNote]) -> bool:
    return {item.note.ordinal for item in notes} == set(range(len(notes)))


_RAW_ARGUMENT_FIELDS = MappingProxyType(
    {
        "list_memories": ({}, {}),
        "read_memory": ({"name": "string"}, {}),
        "write_memory": (
            {"name": "string", "content": "string"},
            {"description": "string", "tags": "string_list"},
        ),
        "append_memory": ({"name": "string", "content": "string"}, {}),
        "search_memory": ({"tags": "string_list"}, {}),
        "list_memory_tags": ({}, {}),
    }
)


def _valid_raw_arguments(tool_name: str, args: object) -> bool:
    if type(args) is not dict or tool_name not in _RAW_ARGUMENT_FIELDS:
        return False
    required, optional = _RAW_ARGUMENT_FIELDS[tool_name]
    if not set(required).issubset(args) or set(args) - set(required) - set(optional):
        return False
    fields = dict(required)
    fields.update(optional)
    for name, value in args.items():
        expected = fields[name]
        if expected == "string" and type(value) is not str:
            return False
        if expected == "string_list" and (
            type(value) is not list or any(type(item) is not str for item in value)
        ):
            return False
    return True


def _raw_argument_metric(tool_name: str, args: object) -> dict[str, Any]:
    raw = args if isinstance(args, Mapping) else {}
    if tool_name == "write_memory":
        return _mutation_metric(
            raw.get("name"),
            raw.get("content"),
            raw.get("description"),
            raw.get("tags"),
        )
    if tool_name == "append_memory":
        return {
            "name": _safe_metric_name(raw.get("name")),
            "content_bytes": _utf8_size(raw.get("content")),
        }
    if tool_name == "read_memory":
        return {"name": _safe_metric_name(raw.get("name"))}
    if tool_name == "search_memory":
        tags = raw.get("tags")
        return {"tag_count": len(tags) if isinstance(tags, list | tuple) else 0}
    return {}


def _full(note: _LoadedNote) -> dict[str, Any]:
    projection = full_projection(note.note, note.binding_created_at, note.revision_created_at)
    return {
        "name": projection.name,
        "content": projection.content,
        "description": projection.description,
        "tags": list(projection.tags),
        "ordinal": projection.ordinal,
        "created_at": _timestamp(projection.created_at),
        "updated_at": _timestamp(projection.updated_at),
    }


def _preview(note: _LoadedNote) -> dict[str, Any]:
    projection = preview_projection(note.note, note.binding_created_at, note.revision_created_at)
    return {
        "name": projection.name,
        "description": projection.description,
        "tags": list(projection.tags),
        "ordinal": projection.ordinal,
        "created_at": _timestamp(projection.created_at),
        "updated_at": _timestamp(projection.updated_at),
    }


def _full_from_metadata(
    note: StoredMemoryNote, metadata: ArtifactWriteValue | ArtifactValue
) -> dict[str, Any]:
    if (
        not _valid_server_time(metadata.binding_created_at)
        or not _valid_server_time(metadata.revision_created_at)
        or metadata.revision_created_at < metadata.binding_created_at
    ):
        raise MemoryToolError("memory_unavailable", retryable=True)
    revision = metadata.artifact.require_exact().revision
    assert revision is not None
    return _full(
        _LoadedNote(
            note=note,
            binding_created_at=metadata.binding_created_at,
            revision_created_at=metadata.revision_created_at,
            revision=revision,
            payload=b"",
        )
    )


def _valid_server_time(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _mutation_metric(
    name: str, content: object, description: object, tags: object
) -> dict[str, Any]:
    return {
        "name": _safe_metric_name(name),
        "content_bytes": _utf8_size(content),
        "description_bytes": _utf8_size(description),
        "tag_count": len(tags) if isinstance(tags, (list, tuple)) else 0,
    }


def _note_metric(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": result.get("name"),
        "content_bytes": _utf8_size(result.get("content", "")),
        "description_bytes": _utf8_size(result.get("description", "")),
        "tag_count": len(result.get("tags", ()))
        if isinstance(result.get("tags"), (list, tuple))
        else 0,
    }


def _utf8_size(value: object) -> int:
    if not isinstance(value, str):
        return 0
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return 0


def _safe_metric_name(value: object) -> str:
    if not isinstance(value, str):
        return "[invalid]"
    try:
        artifact_name(value)
    except MemoryCodecError:
        return "[invalid]"
    return value


def _mapped_client_error(error: Exception) -> MemoryToolError:
    if isinstance(error, ArtifactAPIError) and error.code in {
        "allocation_not_found",
        "allocation_write_fenced",
        "artifact_access_denied",
    }:
        return MemoryToolError("memory_forbidden")
    return MemoryToolError("memory_unavailable", retryable=True)


def _normalize_tool_error(error: Exception) -> MemoryToolError:
    if isinstance(error, MemoryToolError):
        return MemoryToolError(error.code)
    return MemoryToolError("memory_unavailable")


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    from contractor_runtime.toolsets.run_artifacts import _unconfigured_client as factory

    return factory(allocation_id, runtime_settings)


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
