from __future__ import annotations

import asyncio
import random
import tracemalloc
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes.artifact_reads import CountingPayload, count_derivations, hold_preparation
from fakes.worker_state import WorkerState
from google.adk.tools import FunctionTool
from google.genai import types
from shared_config_defaults import MAX_ARTIFACT_BYTES

import contractor_runtime.toolsets.common.artifact_read_cache as read_cache
import contractor_runtime.toolsets.run_artifacts.tools as text_module
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactResponseLimitError,
    ArtifactValue,
)
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactMetadataResult,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.lines import split_lines
from contractor_runtime.factories import built_in_factories
from contractor_runtime.llm.openai import _tools
from contractor_runtime.toolsets.run_artifacts.tools import (
    MAX_VISIBLE_TEXT_BYTES,
    RunArtifactsToolsetFactory,
)
from contractor_runtime.toolsets.text_artifacts.tools import (
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

        read = await tools["read_artifact"](
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
            "nextStartLine": 4,
            "nextLineOffset": 0,
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
        )
        assert updated["artifact"]["revision"] != first_revision
        # An external writer changes the binding without updating the tool observation.
        client.seed("worker-space", "analysis", "text/markdown", b"# Concurrent\n")
        with pytest.raises(ValueError, match="CAS"):
            await tools["write_text_artifact"](
                name="analysis",
                text="# Stale\n",
                media_type="text/markdown",
                )
        current = await client.read_artifact(ArtifactRef(namespace="worker-space", name="analysis"))
        assert current.data == b"# Concurrent\n"
        assert state.metrics.counters["tool_calls.read_artifact"] == 1
        assert state.metrics.counters["tool_calls.write_text_artifact"] == 3

        observed = {
            (ref.namespace, ref.name, ref.revision)
            for tool in tools.values()
            for ref in tool.known_exact_refs
        }
        assert ("worker-space", "analysis", updated["artifact"]["revision"]) in observed

    asyncio.run(scenario())


def test_read_artifact_pages_through_an_oversized_line(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        # One multi-byte line longer than the visible limit, then a short line.
        long_line = "ä" * (MAX_VISIBLE_TEXT_BYTES // 2 + 7) + "\n"
        document = "head\n" + long_line + "tail\n"
        source = client.seed("inputs", "log", "text/plain", document.encode())
        tools = await make_tools(tmp_path, client, WorkerState())
        read = tools["read_artifact"]

        first = await read("inputs", "log", source.revision, start_line=1, max_lines=3)
        assert first["text"] == "head\n"
        assert (first["nextStartLine"], first["nextLineOffset"]) == (2, 0)

        parts: list[str] = []
        start, offset = 2, 0
        while True:
            page = await read(
                "inputs", "log", source.revision, start_line=start, max_lines=1, line_offset=offset
            )
            parts.append(page["text"])
            assert len(page["text"].encode()) <= MAX_VISIBLE_TEXT_BYTES
            if not page["partialLine"]:
                break
            start, offset = page["nextStartLine"], page["nextLineOffset"]
        assert "".join(parts) == long_line
        last = await read("inputs", "log", source.revision, start_line=3)
        assert last["text"] == "tail\n"
        assert last["truncated"] is False
        assert (last["nextStartLine"], last["nextLineOffset"]) == (None, None)
        assert client.read_calls == 1

        with pytest.raises(ValueError, match="line_offset"):
            await read("inputs", "log", source.revision, start_line=1, line_offset=5)
        with pytest.raises(ValueError, match="line_offset"):
            await read("inputs", "log", source.revision, start_line=1, line_offset=-1)
        with pytest.raises(ValueError, match="UTF-8 character"):
            await read("inputs", "log", source.revision, start_line=2, line_offset=1)

    asyncio.run(scenario())


def test_exact_artifact_cache_is_shared_bounded_and_cleared_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        run_client = MemoryArtifactClient()
        text_client = MemoryArtifactClient()
        source = run_client.seed("inputs", "report", "text/plain", b"one\ntwo\n")
        assert text_client.seed("inputs", "report", "text/plain", b"one\ntwo\n") == source
        state = WorkerState()
        text_tools = await make_tools(tmp_path, text_client, state)
        run_factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: run_client)
        run_tools = await run_factory.create_selected(
            selected=["read_artifact"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="worker-space",
            runtime_settings=runtime_settings(),
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
            resources=state.resources,
        )
        read_bytes = run_tools["read_artifact"]
        read_text = text_tools["read_artifact"]
        cache = state.resources.artifact_read_cache
        try:
            first = await read_bytes("inputs", "report", source.revision, max_lines=1)
            assert first["size"] == 8 and first["nextStartLine"] == 2
            cursor = text_client.observation_cursor
            assert (await read_text("inputs", "report", source.revision, start_line=2))[
                "text"
            ] == "two\n"
            assert run_client.read_calls == 1 and text_client.read_calls == 0
            assert text_client.observed_exact_refs_since(cursor) == (source,)
            assert cache._entry is not None and len(cache._entry.value.data) == 8

            updated = text_client.seed("inputs", "report", "text/plain", b"new\n")
            current = await read_text("inputs", "report")
            assert current["artifact"]["revision"] == updated.revision
            assert current["text"] == "new\n"
            assert text_client.read_calls == 1
            assert cache._entry is not None and cache._entry.value.artifact == updated

            await read_bytes.close()
            assert cache._entry is None
            assert (await read_text("inputs", "report", updated.revision))["text"] == "new\n"
            assert text_client.read_calls == 2
            monkeypatch.setattr(read_cache, "artifact_read_cache_max_bytes", lambda: 3)
            await read_text("inputs", "report", source.revision)
            assert cache._entry is None
            await read_text("inputs", "report", source.revision)
            assert text_client.read_calls == 4
        finally:
            await read_text.close()
        assert cache._entry is None

    asyncio.run(scenario())


def numbered_line(number: int) -> str:
    return f"{number:07d} äöü € 😀 {'x' * (number % 97)}\n"


def numbered_document(min_bytes: int) -> tuple[bytes, int]:
    parts: list[bytes] = []
    size = 0
    while size < min_bytes:
        parts.append(numbered_line(len(parts) + 1).encode())
        size += len(parts[-1])
    return b"".join(parts), len(parts)


def test_ten_pages_of_a_32_mib_artifact_prepare_once_and_decode_only_their_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        data, total = numbered_document(32 * 1024 * 1024)
        payload = CountingPayload(data)
        counted = count_derivations(monkeypatch, text_module)
        client = MemoryArtifactClient()
        source = client.seed("inputs", "log", "text/plain", payload)
        read = (await make_tools(tmp_path, client, WorkerState()))["read_artifact"]

        pages = []
        start, offset = 1, 0
        for _ in range(8):
            page = await read("inputs", "log", source.revision, start, 400, offset)
            pages.append(page)
            start, offset = page["nextStartLine"], page["nextLineOffset"]
        pages.append(await read("inputs", "log", source.revision, start_line=total // 2))
        pages.append(await read("inputs", "log", source.revision, start_line=total))

        for page in pages:
            expected = range(page["startLine"], page["endLine"] + 1)
            assert page["text"] == "".join(numbered_line(number) for number in expected)
            assert page["totalLines"] == total and page["partialLine"] is False
        assert pages[7]["endLine"] == 3200
        assert pages[-1]["truncated"] is False and pages[-1]["nextStartLine"] is None
        # One download and one full-payload derivation serve all ten pages.
        assert client.read_calls == 1
        assert (counted.validations, counted.indexes) == (1, 1)
        # A reader that decoded and split the whole payload per page did ten of each.
        assert (payload.decodes, counted.splits) == (0, 0)
        # Each page decodes only its own window.
        assert counted.window_bytes == [len(page["text"].encode()) for page in pages]
        assert max(counted.window_bytes) <= MAX_VISIBLE_TEXT_BYTES

    asyncio.run(scenario())


@pytest.mark.parametrize("fixture", ["short-lines", "astral-in-ascii", "astral-only"])
def test_cache_budget_holds_for_worst_case_inputs_of_the_largest_artifact(
    tmp_path: Path, fixture: str
) -> None:
    emoji = "😀".encode()
    data = {
        # One line per byte: a per-line index would dwarf the payload.
        "short-lines": b"\n" * MAX_ARTIFACT_BYTES,
        # One astral character makes Python store the decoded text in 4 bytes
        # per character: four times the payload.
        "astral-in-ascii": emoji + b"a" * (MAX_ARTIFACT_BYTES - len(emoji)),
        "astral-only": emoji * (MAX_ARTIFACT_BYTES // len(emoji)),
    }[fixture]
    assert len(data) == MAX_ARTIFACT_BYTES

    async def scenario() -> None:
        client = MemoryArtifactClient()
        source = client.seed("inputs", "large", "text/plain", data)
        state = WorkerState()
        read = (await make_tools(tmp_path, client, state))["read_artifact"]

        tracemalloc.start()
        try:
            first = await read("inputs", "large", source.revision, max_lines=400)
            if fixture == "short-lines":
                assert (first["totalLines"], first["text"]) == (MAX_ARTIFACT_BYTES, "\n" * 400)
                last = await read("inputs", "large", source.revision, MAX_ARTIFACT_BYTES - 1)
                assert (last["text"], last["endLine"], last["truncated"]) == (
                    "\n\n",
                    MAX_ARTIFACT_BYTES,
                    False,
                )
            else:
                assert first["totalLines"] == 1 and first["partialLine"] is True
                assert len(first["text"].encode()) <= MAX_VISIBLE_TEXT_BYTES
                assert first["nextLineOffset"] == len(first["text"].encode())
                tail = MAX_ARTIFACT_BYTES - 2 * len(emoji)
                last = await read("inputs", "large", source.revision, 1, 1, tail)
                assert last["text"] == data[tail:].decode()
                assert (last["partialLine"], last["truncated"]) == (False, False)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Paging never materialized the decoded payload or one object per line.
        assert peak < 16 * 1024 * 1024
        entry = state.resources.artifact_read_cache._entry
        assert entry is not None and entry.lines is not None and entry.value.data is data
        index_budget = read_cache.artifact_read_cache_max_bytes() - MAX_ARTIFACT_BYTES
        starts = entry.lines._starts_before_block
        index_bytes = len(starts) * starts.itemsize
        assert index_bytes == index_budget
        assert len(data) + index_bytes <= read_cache.artifact_read_cache_max_bytes()
        assert client.read_calls == 1

    asyncio.run(scenario())


def test_cache_budget_is_the_documented_limit() -> None:
    # LIM-ART-7: one 64 MiB payload plus 16,385 eight-byte line counts.
    assert read_cache.artifact_payload_max_bytes() == MAX_ARTIFACT_BYTES
    assert (
        read_cache.artifact_read_cache_max_bytes()
        == MAX_ARTIFACT_BYTES + read_cache._line_index_bytes(MAX_ARTIFACT_BYTES)
    )


def test_concurrent_first_reads_share_one_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        held = hold_preparation(monkeypatch)
        client = MemoryArtifactClient()
        source = client.seed("inputs", "log", "text/plain", b"one\ntwo\n")
        other = client.seed("inputs", "other", "text/plain", b"other\n")
        state = WorkerState()
        read = (await make_tools(tmp_path, client, state))["read_artifact"]
        cursor = client.observation_cursor

        first = asyncio.create_task(read("inputs", "log", source.revision))
        second = asyncio.create_task(read("inputs", "log", start_line=2))
        # A read of another revision waits too: it cannot evict the entry
        # while its preparation runs.
        replacing = asyncio.create_task(read("inputs", "other", other.revision))
        assert await asyncio.to_thread(held.started.wait, 60)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not first.done() and not second.done() and not replacing.done()
        held.release.set()

        assert (await first)["text"] == "one\ntwo\n"
        assert (await second)["text"] == "two\n"
        assert (await replacing)["text"] == "other\n"
        assert held.calls == 2 and client.read_calls == 2
        # The download and the cached read of the log are content observations.
        assert client.observed_exact_refs_since(cursor) == (source, source, other)
        entry = state.resources.artifact_read_cache._entry
        assert entry is not None and entry.value.artifact == other

    asyncio.run(scenario())


def test_cancelled_preparation_is_joined_discarded_and_blocks_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        held = hold_preparation(monkeypatch)
        client = MemoryArtifactClient()
        source = client.seed("inputs", "log", "text/plain", b"one\ntwo\n")
        state = WorkerState()
        read = (await make_tools(tmp_path, client, state))["read_artifact"]
        cache = state.resources.artifact_read_cache

        reading = asyncio.create_task(read("inputs", "log", source.revision))
        assert await asyncio.to_thread(held.started.wait, 60)
        reading.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        # The cancelled read waits for its thread before it ends.
        assert not reading.done()
        held.release.set()
        with pytest.raises(asyncio.CancelledError):
            await reading
        # The payload stays cached, its derived state is discarded.
        entry = cache._entry
        assert entry is not None and (entry.utf8, entry.lines) == (None, None)
        assert (await read("inputs", "log", source.revision))["text"] == "one\ntwo\n"
        assert (client.read_calls, held.calls) == (1, 2)

        await cache.clear()
        held.started.clear()
        held.release.clear()
        reading = asyncio.create_task(read("inputs", "log", source.revision))
        assert await asyncio.to_thread(held.started.wait, 60)
        reading.cancel()
        closing = asyncio.create_task(read.close())
        for _ in range(5):
            await asyncio.sleep(0)
        # Close waits for the cancelled preparation before it clears the cache.
        assert not reading.done() and not closing.done()
        held.release.set()
        with pytest.raises(asyncio.CancelledError):
            await reading
        await closing
        assert cache._entry is None
        assert (client.read_calls, held.calls) == (2, 3)

    asyncio.run(scenario())


def test_replaced_revision_takes_no_derived_state_from_its_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        counted = count_derivations(monkeypatch, text_module)
        client = MemoryArtifactClient()
        old = client.seed("inputs", "log", "text/plain", b"a\nb\nc\n")
        read = (await make_tools(tmp_path, client, WorkerState()))["read_artifact"]

        assert (await read("inputs", "log", old.revision, start_line=3))["text"] == "c\n"
        new = client.seed("inputs", "log", "text/plain", b"only\n")
        current = await read("inputs", "log")
        assert current["artifact"]["revision"] == new.revision
        assert (current["text"], current["totalLines"]) == ("only\n", 1)
        assert (client.read_calls, counted.indexes) == (2, 2)

        cursor = client.observation_cursor
        cached = await read("inputs", "log", new.revision)
        assert cached["text"] == "only\n"
        assert client.observed_exact_refs_since(cursor) == (new,)
        assert (client.read_calls, counted.indexes) == (2, 2)

        again = await read("inputs", "log", old.revision, start_line=3)
        assert (again["text"], again["totalLines"]) == ("c\n", 3)
        with pytest.raises(ToolInputError, match="start_line exceeds"):
            await read("inputs", "log", new.revision, start_line=2)
        # Every download is validated and indexed once, for its own revision.
        assert (client.read_calls, counted.indexes, counted.validations) == (4, 4, 4)

    asyncio.run(scenario())


def test_run_and_text_readers_share_one_utf8_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        counted = count_derivations(monkeypatch, text_module)
        client = MemoryArtifactClient()
        source = client.seed("inputs", "log", "text/plain", "één\ntwee\n".encode())
        state = WorkerState()
        text_read = (await make_tools(tmp_path, client, state))["read_artifact"]
        run_tools = await RunArtifactsToolsetFactory(
            lambda _allocation, _settings: client
        ).create_selected(
            selected=["read_artifact"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="worker-space",
            runtime_settings=runtime_settings(),
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
            resources=state.resources,
        )
        read_bytes = run_tools["read_artifact"]
        assert (await read_bytes("inputs", "log", source.revision))["size"] == 11
        assert (counted.validations, counted.indexes) == (1, 1)
        assert (await text_read("inputs", "log", source.revision))["totalLines"] == 2
        assert (await read_bytes("inputs", "log", source.revision))["size"] == 11
        assert (counted.validations, counted.indexes, client.read_calls) == (1, 1, 1)

    asyncio.run(scenario())


def test_utf8_validation_spans_chunks_and_rejects_like_strict_decoding(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        chunk = read_cache._UTF8_CHUNK_BYTES
        emoji = "😀".encode()
        # A character split across a validation chunk boundary is valid.
        straddling = b"a" * (chunk - 2) + emoji + b"\n"
        valid = client.seed("inputs", "valid", "text/plain", straddling)
        tools = await make_tools(tmp_path, client, WorkerState())
        read = tools["read_artifact"]
        page = await read("inputs", "valid", valid.revision)
        assert (page["text"], page["partialLine"]) == ("a" * MAX_VISIBLE_TEXT_BYTES, True)
        assert (await read("inputs", "valid", valid.revision, 1, 1, chunk - 2))["text"] == "😀\n"

        truncated = b"a" * (2 * chunk) + emoji[:3]
        surrogate = b"a" * chunk + "\ud800".encode("utf-8", "surrogatepass")
        for name, data in (("truncated", truncated), ("surrogate", surrogate)):
            with pytest.raises(UnicodeDecodeError):
                data.decode("utf-8", errors="strict")
            invalid = client.seed("inputs", name, "application/octet-stream", data)
            for _ in range(2):
                with pytest.raises(ToolInputError) as caught:
                    await read("inputs", name, invalid.revision)
                assert str(caught.value) == "artifact is not valid UTF-8"

    asyncio.run(scenario())


def reference_text_page(
    document: str, start_line: int, max_lines: int, line_offset: int, visible: int
) -> dict[str, object] | str:
    """The read_artifact window as computed before the line index existed."""

    lines = split_lines(document, keepends=True)
    total_lines = len(lines)
    if total_lines > 0 and start_line > total_lines:
        return "start_line exceeds artifact line count"
    window: tuple[str, int, bool, int, int] | None = None
    if not lines:
        if line_offset:
            return "line_offset exceeds the line length"
        window = ("", 0, False, 1, 0)
    else:
        first = lines[start_line - 1].encode()
        if line_offset and line_offset >= len(first):
            return "line_offset exceeds the line length"
        if line_offset and (first[line_offset] & 0xC0) == 0x80:
            return "line_offset must start a UTF-8 character"
        parts = [
            first[line_offset:],
            *(line.encode() for line in lines[start_line : start_line - 1 + max_lines]),
        ]
        result: list[bytes] = []
        size = 0
        for encoded in parts:
            remaining = visible - size
            if len(encoded) <= remaining:
                result.append(encoded)
                size += len(encoded)
                continue
            if not result:
                cut = remaining
                while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
                    cut -= 1
                window = (encoded[:cut].decode(), start_line, True, start_line, line_offset + cut)
            break
        if window is None:
            end_line = start_line + len(result) - 1
            window = (b"".join(result).decode(), end_line, False, end_line + 1, 0)
    text, end_line, partial_line, next_start_line, next_line_offset = window
    truncated = partial_line or end_line < total_lines
    return {
        "totalLines": total_lines,
        "startLine": start_line,
        "endLine": end_line,
        "text": text,
        "truncated": truncated,
        "partialLine": partial_line,
        "nextStartLine": next_start_line if truncated else None,
        "nextLineOffset": next_line_offset if truncated else None,
    }


def test_indexed_pages_match_whole_text_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tiny blocks and windows put line breaks, CRLF pairs and multi-byte
    # characters on every block and window boundary.
    visible = 9
    monkeypatch.setattr(read_cache, "LINE_INDEX_BLOCK_BYTES", 3)
    monkeypatch.setattr(text_module, "MAX_VISIBLE_TEXT_BYTES", visible)
    pieces = ["a", "bc", "ä", "€", "😀", "\r", "\n", "\r\n", "\u2028", "\x85", "\f"]
    rng = random.Random(407035)
    documents = ["", "\n", "\r", "\r\n", "x", "\r\r\n\n", "a" * 20, "😀" * 5 + "\r\n"]
    documents += ["".join(rng.choices(pieces, k=rng.randint(1, 40))) for _ in range(120)]

    async def scenario() -> None:
        client = MemoryArtifactClient()
        read = (await make_tools(tmp_path, client, WorkerState()))["read_artifact"]
        compared = 0
        for number, document in enumerate(documents):
            name = f"doc-{number}"
            source = client.seed("inputs", name, "text/plain", document.encode())
            lines = split_lines(document, keepends=True)
            longest = max((len(line.encode()) for line in lines), default=0)
            for start_line in range(1, len(lines) + 2):
                for max_lines in (1, 3, 400):
                    offsets = range(longest + 2) if max_lines == 1 else (0, 1)
                    for line_offset in offsets:
                        case = (document, start_line, max_lines, line_offset)
                        expected = reference_text_page(*case, visible)
                        call = read("inputs", name, source.revision, *case[1:])
                        if isinstance(expected, str):
                            with pytest.raises(ToolInputError) as caught:
                                await call
                            assert str(caught.value) == expected, case
                        else:
                            page = await call
                            assert {key: page[key] for key in expected} == expected, case
                        compared += 1
        assert compared > 5000

    asyncio.run(scenario())


def test_reader_schema_exposes_supported_arguments(tmp_path: Path) -> None:
    async def scenario() -> None:
        tools = await make_tools(tmp_path, MemoryArtifactClient(), WorkerState())
        declaration = FunctionTool(tools["read_artifact"])._get_declaration()
        schema = _tools([types.Tool(function_declarations=[declaration])])[0]["function"][
            "parameters"
        ]
        assert set(schema["properties"]) == {
            "name",
            "revision",
            "max_lines",
            "namespace",
            "line_offset",
            "start_line",
        }

    asyncio.run(scenario())


def test_read_artifact_numbers_lines_like_read_file(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        # Only \r\n, \r and \n end lines; other Unicode separators are content.
        document = "a\fb\x1cc\x85d\u2028e\u2029f\vg\r\nsecond\rthird\n"
        source = client.seed("inputs", "log", "text/plain", document.encode())
        tools = await make_tools(tmp_path, client, WorkerState())
        read = tools["read_artifact"]
        whole = await read("inputs", "log", source.revision)
        assert whole["totalLines"] == 3 and whole["text"] == document
        second = await read("inputs", "log", source.revision, start_line=2, max_lines=1)
        assert second["text"] == "second\r"
        assert second["nextStartLine"] == 3

    asyncio.run(scenario())


def test_text_tools_reject_invalid_windows_utf8_and_oversize(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        invalid = client.seed("inputs", "binary", "application/octet-stream", b"\xff")
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state)

        with pytest.raises(ValueError, match="positive integer"):
            await tools["read_artifact"]("inputs", "binary", invalid.revision, 0, 1)
        with pytest.raises(ValueError, match="valid UTF-8"):
            await tools["read_artifact"]("inputs", "binary", invalid.revision)
        with pytest.raises(ValueError, match="1 MiB"):
            await tools["write_text_artifact"](
                "too-large",
                "x" * (MAX_TEXT_WRITE_BYTES + 1),
                "text/plain",
            )
        assert ("worker-space", "too-large") not in client.bindings

    asyncio.run(scenario())


def test_read_artifact_rejects_binary_media_types_before_download(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        archive = client.seed("inputs", "source", "application/zip", b"PK\x03\x04")
        tools = await make_tools(tmp_path, client, WorkerState())

        for revision in (None, archive.revision):
            with pytest.raises(ValueError, match=r"binary \(application/zip\)"):
                await tools["read_artifact"]("inputs", "source", revision)
        assert client.metadata_calls == 2 and client.read_calls == 0

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
            await tools["read_artifact"](hidden.namespace, hidden.name, hidden.revision)
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_text_artifact"](
                hidden.name,
                "must not replace",
                "text/plain",
            )
        assert (client.read_calls, client.write_calls) == calls_before
        assert client.bindings[(hidden.namespace, hidden.name)].data == b"hidden"

        visible = await tools["read_artifact"](allowed.namespace, allowed.name, allowed.revision)
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
                resources=WorkerState().resources,
            )

    asyncio.run(scenario())
    registry = built_in_factories(tmp_path)
    assert registry.toolsets["text-artifacts@1"].exported_tools == {"write_text_artifact"}


async def make_tools(
    tmp_path: Path,
    client: MemoryArtifactClient,
    state: WorkerState,
) -> dict[str, object]:
    factory = TextArtifactsToolsetFactory(lambda _allocation, _settings: client)
    selected = await factory.create_selected(
        selected=["write_text_artifact"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="worker-space",
        runtime_settings=runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        resources=state.resources,
    )
    reader = await RunArtifactsToolsetFactory(
        lambda _allocation, _settings: client
    ).create_selected(
        selected=["read_artifact"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="worker-space",
        runtime_settings=runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        resources=state.resources,
    )
    return dict(selected) | dict(reader)


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
        self._observed: list[ArtifactRef] = []
        self._next_revision = 1
        self.metadata_calls = 0
        self.read_calls = 0
        self.write_calls = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    @property
    def observation_cursor(self) -> int:
        return len(self._observed)

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        return tuple(self._observed[cursor:])

    def observe_cached_read(self, ref: ArtifactRef) -> None:
        self._remember(ref.require_exact())

    def seed(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        return self._store(namespace, name, media_type, data)

    async def read_artifact_metadata(self, ref: ArtifactRef) -> ArtifactMetadataResult:
        self.metadata_calls += 1
        if ref.revision is None:
            stored = self.bindings[(ref.namespace, ref.name)]
        else:
            stored = self.history[(ref.namespace, ref.name, ref.revision)]
        return ArtifactMetadataResult(
            contentCacheKey="test_metadata_version",
            apiVersion=API_VERSION,
            artifact=ArtifactRef(namespace=ref.namespace, name=ref.name, revision=stored.revision),
            mediaType=stored.media_type,
            size=len(stored.data),
        )

    async def read_artifact(
        self, ref: ArtifactRef, *, max_bytes: int = MAX_ARTIFACT_BYTES
    ) -> ArtifactValue:
        self.read_calls += 1
        try:
            if ref.revision is None:
                stored = self.bindings[(ref.namespace, ref.name)]
            else:
                stored = self.history[(ref.namespace, ref.name, ref.revision)]
        except KeyError:
            raise ArtifactAPIError(404, "artifact_not_found", False) from None
        if len(stored.data) > max_bytes:
            raise ArtifactResponseLimitError("response too large")
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
        self._observed.append(ref)


def test_text_writer_canonicalizes_optional_analysis_and_rejects_bad_shape(tmp_path: Path) -> None:
    import json

    from contractor_runtime.toolsets.common.finding_analysis import MEDIA_TYPE, canonical_analysis

    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState())
        value = {
            "schema": "contractor.finding.analysis.v1",
            "client_key": "candidate",
            "sources": [],
            "locations": [],
            "flows": [],
            "http_exchanges": [],
        }
        authored = json.dumps(value, indent=2)
        result = await tools["write_text_artifact"](
            name="typed-analysis", text=authored, media_type=MEDIA_TYPE
        )
        saved = await client.read_artifact(ArtifactRef.model_validate(result["artifact"]))
        assert saved.data == canonical_analysis(authored)
        assert result["size"] == len(saved.data)
        with pytest.raises(ToolInputError, match="finding analysis is invalid"):
            await tools["write_text_artifact"](
                name="bad-analysis", text='{"sources":null}', media_type=MEDIA_TYPE
            )

    asyncio.run(scenario())


def test_worker_gateway_transcript_continues_an_exact_multibyte_line(tmp_path: Path) -> None:
    from datetime import timedelta

    from fakes.model import ScriptedLlm, text_result, tool_call
    from test_adk_runtime import create_runtime, stage_request

    from contractor_runtime.llm.openai import _completion_request

    transcripts = []

    class TranscriptModel(ScriptedLlm):
        async def generate_content_async(self, llm_request, stream=False):
            transcripts.append(_completion_request(self.model, llm_request))
            async for response in super().generate_content_async(llm_request, stream):
                yield response

    async def scenario():
        client = MemoryArtifactClient()
        content = ("é😀" * 25_000 + "\nend\n").encode()
        exact = client.seed("inputs", "log", "text/plain", content)
        state = WorkerState()
        reader = (await make_tools(tmp_path, client, state))["read_artifact"]
        # Both character widths divide this first 128-KiB page only after
        # trimming to the last complete UTF-8 character.
        cut = MAX_VISIBLE_TEXT_BYTES
        while content[cut] & 0xC0 == 0x80:
            cut -= 1
        model = TranscriptModel(
            model="fixture",
            responses=[
                tool_call("read_artifact", exact.model_dump(exclude_none=True), call_id="page-one"),
                tool_call(
                    "read_artifact",
                    {**exact.model_dump(exclude_none=True), "start_line": 1, "line_offset": cut},
                    call_id="page-two",
                ),
                text_result("Inspected the assigned material"),
            ],
        )
        runtime = await create_runtime(tmp_path, state, {"read_artifact": reader}, model)
        try:
            result = await runtime.invoke(
                stage_request().model_copy(update={"result_artifacts": {}})
            )
            assert result.failure is None
            tool_messages = [
                message for message in transcripts[2]["messages"] if message["role"] == "tool"
            ]
            import json

            pages = [json.loads(message["content"]) for message in tool_messages]
            assert len(pages) == 2
            assert pages[0]["nextStartLine"] == 1 and pages[0]["nextLineOffset"] == cut
            assert all(page["artifact"] == exact.model_dump() for page in pages)
            assert "".join(page["text"] for page in pages).encode() == content
            assert all("dataBase64" not in page for page in pages)
            names = [tool["function"]["name"] for tool in transcripts[0]["tools"]]
            assert names == ["read_artifact"]
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
