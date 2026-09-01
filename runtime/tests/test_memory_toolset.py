from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from google.adk.tools import FunctionTool

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactTransportError,
    ArtifactValue,
    ArtifactWriteValue,
)
from contractor_runtime.contracts import API_VERSION, ArtifactRef, RuntimeSettings
from contractor_runtime.memory import (
    MAXIMUM_NAME_BYTES,
    MAXIMUM_PAYLOAD_BYTES,
    MEDIA_TYPE,
    SCHEMA_VERSION,
    StoredMemoryNote,
    encode_note,
)
from contractor_runtime.toolsets.memory import (
    MAXIMUM_NOTES,
    MemoryToolError,
    MemoryToolsetFactory,
    _full,
    _LoadedNote,
    _ordered,
    _preview,
)
from contractor_runtime.workspace import AllocationWorkspace

CONTENT_CANARY = "memory-content-recognizable-canary"
DESCRIPTION_CANARY = "memory-description-recognizable-canary"
TAG_CANARY = "recognizable-tag-canary"
EPOCH = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def test_factory_probe_selection_and_model_schemas_are_exact() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = MemoryToolsetFactory(lambda _allocation, _settings: client)
        assert await factory.probe() == factory.exported_tools

        selected = ["list_memories", "read_memory", "search_memory"]
        tools = await create_tools(factory, state, selected)
        assert set(tools) == set(selected)

        all_tools = await create_tools(factory, WorkerState(), sorted(factory.exported_tools))
        declarations = {
            name: FunctionTool(tool)
            ._get_declaration()
            .model_dump(mode="json", by_alias=True, exclude_none=True)
            for name, tool in all_tools.items()
        }
        parameters = {
            name: declaration.get("parametersJsonSchema", {"type": "object", "properties": {}})
            for name, declaration in declarations.items()
        }
        assert set(parameters["list_memories"].get("properties", {})) == set()
        assert set(parameters["list_memory_tags"].get("properties", {})) == set()
        assert set(parameters["read_memory"]["properties"]) == {"name"}
        assert parameters["read_memory"]["required"] == ["name"]
        assert set(parameters["append_memory"]["properties"]) == {"name", "content"}
        assert set(parameters["append_memory"]["required"]) == {"name", "content"}
        assert set(parameters["search_memory"]["properties"]) == {"tags"}
        assert parameters["search_memory"]["required"] == ["tags"]
        assert set(parameters["write_memory"]["properties"]) == {
            "name",
            "content",
            "description",
            "tags",
        }
        assert set(parameters["write_memory"]["required"]) == {"name", "content"}
        encoded = repr(parameters)
        for hidden in (
            "namespace",
            "run_id",
            "artifact",
            "revision",
            "expected_revision",
            "allocation",
        ):
            assert hidden not in encoded

    asyncio.run(scenario())


def test_write_replace_append_list_search_and_tag_semantics() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            state,
            [
                "write_memory",
                "append_memory",
                "read_memory",
                "list_memories",
                "search_memory",
                "list_memory_tags",
            ],
        )

        first = await tools["write_memory"](
            "repo_overview", "first", "overview", ["repository", "architecture"]
        )
        second = await tools["write_memory"](
            "findings", "finding one", "security", ["security", "repository"]
        )
        replaced = await tools["write_memory"](
            "repo_overview", "replacement", tags=["architecture"]
        )
        appended = await tools["append_memory"]("findings", "finding two")

        assert first["ordinal"] == 0
        assert second["ordinal"] == 1
        assert replaced["ordinal"] == first["ordinal"]
        assert replaced["created_at"] == first["created_at"]
        assert replaced["updated_at"] != first["updated_at"]
        assert replaced["description"] == ""
        assert replaced["tags"] == ["architecture"]
        assert appended["content"] == "finding one\nfinding two"
        assert appended["description"] == "security"
        assert appended["tags"] == ["repository", "security"]
        assert appended["created_at"] == second["created_at"]

        read = await tools["read_memory"]("findings")
        assert read == appended
        listed = await tools["list_memories"]()
        assert [note["name"] for note in listed] == ["findings", "repo_overview"]
        assert all("content" not in note for note in listed)
        searched = await tools["search_memory"](["architecture", "security"])
        assert {note["name"] for note in searched} == {"findings", "repo_overview"}
        assert all("content" not in note for note in searched)
        assert await tools["list_memory_tags"]() == [
            "architecture",
            "repository",
            "security",
        ]
        assert all(
            set(note)
            == {
                "name",
                "content",
                "description",
                "tags",
                "ordinal",
                "created_at",
                "updated_at",
            }
            for note in (first, second, replaced, appended, read)
        )
        calls_by_tool = {call.tool: call for call in state.metrics.tool_calls}
        assert calls_by_tool["list_memories"].arguments["result_count"] == 2
        assert calls_by_tool["search_memory"].arguments["result_count"] == 2
        assert calls_by_tool["list_memory_tags"].arguments["result_count"] == 3
        assert calls_by_tool["read_memory"].arguments["result_content_bytes"] == len(
            appended["content"].encode()
        )
        assert "artifact" not in repr((first, second, replaced, appended, read, listed))
        assert "revision" not in repr((first, second, replaced, appended, read, listed))

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["before", "after"])
def test_response_loss_replays_exact_create_bytes_and_precondition(fault: str) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient(write_faults=[fault])
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            WorkerState(),
            ["write_memory"],
        )
        result = await tools["write_memory"](
            "response_loss", CONTENT_CANARY, DESCRIPTION_CANARY, [TAG_CANARY]
        )
        assert result["content"] == CONTENT_CANARY
        assert len(client.write_attempts) == 2
        first, second = client.write_attempts
        assert first == second
        assert first.expected_revision is None
        assert client.semantic_writes == 1
        stored = client.binding("builder", "memory.response_loss")
        assert stored is not None and stored.payload == first.payload

    asyncio.run(scenario())


def test_response_loss_append_is_not_applied_twice() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            WorkerState(),
            ["write_memory", "append_memory", "read_memory"],
        )
        await tools["write_memory"]("append_once", "line one")
        client.write_faults.append("after")
        before_attempts = len(client.write_attempts)
        result = await tools["append_memory"]("append_once", "line two")
        assert result["content"] == "line one\nline two"
        assert (await tools["read_memory"]("append_once"))["content"] == result["content"]
        attempts = client.write_attempts[before_attempts:]
        assert len(attempts) == 2 and attempts[0] == attempts[1]
        assert client.semantic_writes == 2

    asyncio.run(scenario())


def test_stale_cas_and_newer_value_after_lost_response_return_memory_changed() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            WorkerState(),
            ["write_memory", "append_memory"],
        )
        created = await tools["write_memory"]("shared", "base")
        assert created["content"] == "base"

        client.advance_payload = encoded_note("shared", "external", ordinal=0)
        client.write_faults.append("advance_before")
        with pytest.raises(MemoryToolError) as stale:
            await tools["append_memory"]("shared", "ours")
        assert stale.value.code == "memory_changed"
        assert stale.value.retryable
        assert client.decoded_content("builder", "memory.shared") == "external"

        client.advance_payload = encoded_note("shared", "newer external", ordinal=0)
        client.write_faults.append("after_advance")
        with pytest.raises(MemoryToolError) as newer:
            await tools["append_memory"]("shared", "lost append")
        assert newer.value.code == "memory_changed"
        assert client.decoded_content("builder", "memory.shared") == "newer external"

    asyncio.run(scenario())


def test_namespace_quota_rejects_new_note_but_allows_existing_update() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        for ordinal in range(MAXIMUM_NOTES):
            client.seed_note("builder", f"note_{ordinal}", f"body {ordinal}", ordinal=ordinal)
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            WorkerState(),
            ["write_memory", "list_memories"],
        )
        with pytest.raises(MemoryToolError) as full:
            await tools["write_memory"]("overflow", "no eviction")
        assert full.value.code == "memory_namespace_full"
        assert client.binding("builder", "memory.note_0") is not None
        updated = await tools["write_memory"]("note_0", "updated at capacity")
        assert updated["ordinal"] == 0
        assert len(await tools["list_memories"]()) == MAXIMUM_NOTES

    asyncio.run(scenario())


@pytest.mark.parametrize("ordinals", [(0, 0), (0, 2)], ids=["duplicate", "gap"])
def test_impossible_ordinals_fail_only_global_views_and_create(
    ordinals: tuple[int, int],
) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        client.seed_note("builder", "first", "first body", ordinal=ordinals[0])
        client.seed_note("builder", "second", "second body", ordinal=ordinals[1])
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            WorkerState(),
            [
                "read_memory",
                "write_memory",
                "list_memories",
                "search_memory",
                "list_memory_tags",
            ],
        )

        assert (await tools["read_memory"]("first"))["content"] == "first body"
        updated = await tools["write_memory"]("first", "updated")
        assert updated["content"] == "updated"
        assert updated["ordinal"] == ordinals[0]

        operations = (
            lambda: tools["list_memories"](),
            lambda: tools["search_memory"](["tag"]),
            lambda: tools["list_memory_tags"](),
            lambda: tools["write_memory"]("third", "body"),
        )
        for operation in operations:
            with pytest.raises(MemoryToolError) as raised:
                await operation()
            assert raised.value.code == "memory_unavailable"
            assert raised.value.retryable

    asyncio.run(scenario())


def test_append_uses_only_the_real_final_payload_bound() -> None:
    async def scenario() -> None:
        existing = "x"
        fragment = _maximum_fitting_append_fragment("a", existing)
        assert len(fragment) > MAXIMUM_PAYLOAD_BYTES // 2

        short_client = FakeArtifactClient()
        short_tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: short_client),
            WorkerState(),
            ["write_memory", "append_memory"],
        )
        await short_tools["write_memory"]("a", existing)
        appended = await short_tools["append_memory"]("a", fragment)
        assert appended["content"] == existing + "\n" + fragment

        long_name = "a" + "b" * (MAXIMUM_NAME_BYTES - 1)
        long_client = FakeArtifactClient()
        long_tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: long_client),
            WorkerState(),
            ["write_memory", "append_memory"],
        )
        await long_tools["write_memory"](long_name, existing)
        with pytest.raises(MemoryToolError) as too_large:
            await long_tools["append_memory"](long_name, fragment)
        assert too_large.value.code == "memory_too_large"
        assert not too_large.value.retryable

        for invalid in ("", "\ud800"):
            with pytest.raises(MemoryToolError) as malformed:
                await short_tools["append_memory"]("a", invalid)
            assert malformed.value.code == "memory_invalid"
            assert not malformed.value.retryable

    asyncio.run(scenario())


def test_memory_order_is_updated_desc_ordinal_desc_name_asc() -> None:
    timestamp = EPOCH

    def loaded(name: str, ordinal: int, updated_at: datetime) -> _LoadedNote:
        note = StoredMemoryNote(SCHEMA_VERSION, name, "body", "", (), ordinal)
        return _LoadedNote(note, EPOCH, updated_at, "revision", encode_note(note))

    notes = [
        loaded("zeta", 1, timestamp),
        loaded("beta", 2, timestamp),
        loaded("alpha", 2, timestamp),
        loaded("newest", 0, timestamp + timedelta(seconds=1)),
    ]
    assert [item.note.name for item in _ordered(notes)] == [
        "newest",
        "alpha",
        "beta",
        "zeta",
    ]


def test_memory_model_timestamps_are_normalized_to_utc_z() -> None:
    local = timezone(timedelta(hours=3))
    created = datetime(2026, 9, 1, 10, 11, 12, tzinfo=local)
    updated = created + timedelta(seconds=1)
    note = StoredMemoryNote(SCHEMA_VERSION, "note", "body", "", (), 0)
    loaded = _LoadedNote(note, created, updated, "revision", encode_note(note))

    assert _full(loaded)["created_at"] == "2026-09-01T07:11:12Z"
    assert _preview(loaded)["updated_at"] == "2026-09-01T07:11:13Z"


def test_memory_errors_use_the_closed_code_and_retryability_table() -> None:
    for code, supplied_retryable, expected_code, expected_retryable in (
        ("memory_invalid", True, "memory_invalid", False),
        ("memory_changed", False, "memory_changed", True),
        ("memory_forbidden", True, "memory_forbidden", False),
        ("internal_arbitrary", False, "memory_unavailable", True),
    ):
        error = MemoryToolError(code, retryable=supplied_retryable)
        assert error.code == expected_code
        assert error.retryable is expected_retryable
        assert code not in str(error) or code == expected_code


def test_not_found_invalid_search_and_corrupt_payload_errors_are_bounded() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        client.seed_raw(
            "builder",
            "memory.corrupt",
            b'{"not":"a note"}',
            MEDIA_TYPE,
        )
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            WorkerState(),
            ["read_memory", "list_memories", "search_memory", "append_memory"],
        )
        with pytest.raises(MemoryToolError) as missing:
            await tools["read_memory"]("missing")
        assert missing.value.code == "memory_not_found"
        with pytest.raises(MemoryToolError) as append_missing:
            await tools["append_memory"]("missing", "x")
        assert append_missing.value.code == "memory_not_found"
        for invalid in ([], ["same", "same"], ["a", "b", "c", "d"], ["Uppercase"]):
            with pytest.raises(MemoryToolError) as search:
                await tools["search_memory"](invalid)
            assert search.value.code == "memory_invalid"
        with pytest.raises(MemoryToolError) as corrupt:
            await tools["list_memories"]()
        assert corrupt.value.code == "memory_unavailable"
        assert corrupt.value.retryable
        rendered = repr((missing.value, append_missing.value, corrupt.value))
        assert "not a note" not in rendered

    asyncio.run(scenario())


def test_artifact_failures_map_to_the_closed_memory_error_contract() -> None:
    async def scenario() -> None:
        list_failure = FakeArtifactClient(
            list_error=ArtifactTransportError("private endpoint details")
        )
        list_tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: list_failure),
            WorkerState(),
            ["list_memories"],
        )
        with pytest.raises(MemoryToolError) as unavailable:
            await list_tools["list_memories"]()
        assert unavailable.value.code == "memory_unavailable"
        assert unavailable.value.retryable
        assert "private endpoint details" not in str(unavailable.value)

        read_failure = FakeArtifactClient(
            read_error=ArtifactAPIError(403, "artifact_access_denied", False)
        )
        read_tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: read_failure),
            WorkerState(),
            ["read_memory"],
        )
        with pytest.raises(MemoryToolError) as forbidden_read:
            await read_tools["read_memory"]("secret")
        assert forbidden_read.value.code == "memory_forbidden"
        assert not forbidden_read.value.retryable

        write_failure = FakeArtifactClient(write_faults=["forbidden"])
        write_tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: write_failure),
            WorkerState(),
            ["write_memory"],
        )
        with pytest.raises(MemoryToolError) as forbidden_write:
            await write_tools["write_memory"]("blocked", "body")
        assert forbidden_write.value.code == "memory_forbidden"
        assert not forbidden_write.value.retryable
        assert write_failure.semantic_writes == 0

    asyncio.run(scenario())


def test_calls_are_serialized_and_metrics_retain_only_safe_dimensions() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await create_tools(
            MemoryToolsetFactory(lambda _allocation, _settings: client),
            state,
            ["write_memory"],
        )
        first, second = await asyncio.gather(
            tools["write_memory"]("first_note", CONTENT_CANARY, DESCRIPTION_CANARY, [TAG_CANARY]),
            tools["write_memory"]("second_note", "second body", tags=["second-tag"]),
        )
        assert {first["ordinal"], second["ordinal"]} == {0, 1}
        assert client.maximum_active_operations == 1
        assert len(state.metrics.tool_calls) == 2
        for call in state.metrics.tool_calls:
            assert call.arguments is not None
            assert set(call.arguments) == {
                "name",
                "content_bytes",
                "description_bytes",
                "tag_count",
                "result_content_bytes",
                "result_description_bytes",
                "result_tag_count",
            }
        rendered = repr(state.metrics)
        assert CONTENT_CANARY not in rendered
        assert DESCRIPTION_CANARY not in rendered
        assert TAG_CANARY not in rendered
        by_name = {call.arguments["name"]: call for call in state.metrics.tool_calls}
        assert by_name["first_note"].arguments == {
            "name": "first_note",
            "content_bytes": len(CONTENT_CANARY.encode()),
            "description_bytes": len(DESCRIPTION_CANARY.encode()),
            "tag_count": 1,
            "result_content_bytes": len(CONTENT_CANARY.encode()),
            "result_description_bytes": len(DESCRIPTION_CANARY.encode()),
            "result_tag_count": 1,
        }
        assert by_name["second_note"].arguments["content_bytes"] == len(b"second body")

    asyncio.run(scenario())


async def create_tools(
    factory: MemoryToolsetFactory,
    state: WorkerState,
    selected: list[str],
) -> dict[str, Any]:
    settings = RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken="secret-not-needed-by-memory",
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )
    workspace_path = Path("/tmp/contractor-memory-tool-test/allocation-test")
    return dict(
        await factory.create_selected(
            selected=selected,
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="builder",
            runtime_settings=settings,
            workspace=AllocationWorkspace(root=workspace_path.parent, path=workspace_path),
            state=state,
        )
    )


@dataclass(frozen=True, slots=True)
class WriteAttempt:
    namespace: str
    name: str
    payload: bytes
    media_type: str
    expected_revision: str | None


@dataclass(slots=True)
class StoredArtifact:
    revision: str
    media_type: str
    payload: bytes
    binding_created_at: datetime
    revision_created_at: datetime


class FakeArtifactClient:
    def __init__(
        self,
        *,
        write_faults: list[str] | None = None,
        list_error: Exception | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self._bindings: dict[tuple[str, str], StoredArtifact] = {}
        self._next_revision = 0
        self._clock = 0
        self.write_faults = list(write_faults or [])
        self.list_error = list_error
        self.read_error = read_error
        self.write_attempts: list[WriteAttempt] = []
        self.semantic_writes = 0
        self.advance_payload: bytes | None = None
        self.active_operations = 0
        self.maximum_active_operations = 0
        self._known: dict[tuple[str, str, str], ArtifactRef] = {}

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    @asynccontextmanager
    async def _operation(self):
        self.active_operations += 1
        self.maximum_active_operations = max(self.maximum_active_operations, self.active_operations)
        await asyncio.sleep(0)
        try:
            yield
        finally:
            self.active_operations -= 1

    async def list_artifacts(self, namespace: str | None = None) -> list[ArtifactRef]:
        async with self._operation():
            if self.list_error is not None:
                raise self.list_error
            return [
                ArtifactRef(namespace=item_namespace, name=name)
                for item_namespace, name in sorted(self._bindings)
                if namespace is None or item_namespace == namespace
            ]

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        async with self._operation():
            if self.read_error is not None:
                raise self.read_error
            stored = self._bindings.get((ref.namespace, ref.name))
            if stored is None or (ref.revision is not None and ref.revision != stored.revision):
                raise ArtifactAPIError(404, "artifact_not_found", False)
            exact = ArtifactRef(namespace=ref.namespace, name=ref.name, revision=stored.revision)
            self._known[(ref.namespace, ref.name, stored.revision)] = exact
            return ArtifactValue(
                artifact=exact,
                media_type=stored.media_type,
                data=stored.payload,
                binding_created_at=stored.binding_created_at,
                revision_created_at=stored.revision_created_at,
            )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteValue:
        async with self._operation():
            attempt = WriteAttempt(
                target.namespace, target.name, data, media_type, expected_revision
            )
            self.write_attempts.append(attempt)
            fault = self.write_faults.pop(0) if self.write_faults else ""
            if fault == "forbidden":
                raise ArtifactAPIError(409, "allocation_write_fenced", False)
            if fault == "before":
                raise ArtifactTransportError("synthetic response loss before commit")
            if fault == "advance_before":
                self._advance(target.namespace, target.name)
            stored = self._apply(attempt)
            if fault == "after_advance":
                self._advance(target.namespace, target.name)
            if fault in {"after", "after_advance"}:
                raise ArtifactTransportError("synthetic response loss after commit")
            exact = ArtifactRef(
                namespace=target.namespace, name=target.name, revision=stored.revision
            )
            self._known[(target.namespace, target.name, stored.revision)] = exact
            return ArtifactWriteValue(
                apiVersion=API_VERSION,
                artifact=exact,
                mediaType=stored.media_type,
                size=len(stored.payload),
                binding_created_at=stored.binding_created_at,
                revision_created_at=stored.revision_created_at,
            )

    def _apply(self, attempt: WriteAttempt) -> StoredArtifact:
        key = (attempt.namespace, attempt.name)
        current = self._bindings.get(key)
        if (attempt.expected_revision is None and current is not None) or (
            attempt.expected_revision is not None
            and (current is None or current.revision != attempt.expected_revision)
        ):
            raise ArtifactAPIError(409, "artifact_conflict", True)
        binding_created_at = current.binding_created_at if current is not None else self._tick()
        revision_created_at = self._tick()
        self._next_revision += 1
        stored = StoredArtifact(
            revision=f"revision-{self._next_revision}",
            media_type=attempt.media_type,
            payload=attempt.payload,
            binding_created_at=binding_created_at,
            revision_created_at=revision_created_at,
        )
        self._bindings[key] = stored
        self.semantic_writes += 1
        return stored

    def _advance(self, namespace: str, name: str) -> None:
        current = self._bindings.get((namespace, name))
        if current is None or self.advance_payload is None:
            raise AssertionError("advance fault requires an existing binding and payload")
        self._next_revision += 1
        self._bindings[(namespace, name)] = StoredArtifact(
            revision=f"revision-{self._next_revision}",
            media_type=MEDIA_TYPE,
            payload=self.advance_payload,
            binding_created_at=current.binding_created_at,
            revision_created_at=self._tick(),
        )
        self.semantic_writes += 1

    def seed_note(self, namespace: str, name: str, content: str, *, ordinal: int) -> None:
        self.seed_raw(
            namespace,
            f"memory.{name}",
            encoded_note(name, content, ordinal=ordinal),
            MEDIA_TYPE,
        )

    def seed_raw(self, namespace: str, name: str, payload: bytes, media_type: str) -> None:
        key = (namespace, name)
        self._next_revision += 1
        timestamp = self._tick()
        self._bindings[key] = StoredArtifact(
            revision=f"revision-{self._next_revision}",
            media_type=media_type,
            payload=payload,
            binding_created_at=timestamp,
            revision_created_at=timestamp,
        )

    def binding(self, namespace: str, name: str) -> StoredArtifact | None:
        return self._bindings.get((namespace, name))

    def decoded_content(self, namespace: str, name: str) -> str:
        stored = self._bindings[(namespace, name)]
        import json

        return str(json.loads(stored.payload)["content"])

    def _tick(self) -> datetime:
        self._clock += 1
        return EPOCH + timedelta(microseconds=self._clock)


def encoded_note(name: str, content: str, *, ordinal: int) -> bytes:
    return encode_note(
        StoredMemoryNote(
            schema_version=SCHEMA_VERSION,
            name=name,
            content=content,
            description="",
            tags=(),
            ordinal=ordinal,
        )
    )


def _maximum_fitting_append_fragment(name: str, existing: str) -> str:
    low, high = 1, MAXIMUM_PAYLOAD_BYTES
    while low < high:
        middle = low + (high - low + 1) // 2
        try:
            encoded_note(name, existing + "\n" + "x" * middle, ordinal=0)
        except Exception:
            high = middle - 1
        else:
            low = middle
    return "x" * low
