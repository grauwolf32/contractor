from __future__ import annotations

import asyncio
import io
import json
import stat
import threading
import unicodedata
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes.worker_state import WorkerState

import contractor_runtime.projectfs.paths as paths_module
import contractor_runtime.toolsets.source_analysis.tools as source_tools_module
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.contracts.settings import default_source_archive_limits
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.lines import split_lines
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.source_analysis.tools import (
    MAX_ARCHIVE_ENTRIES,
    MAX_FILE_UNCOMPRESSED_BYTES,
    MAX_LIST_RESULTS,
    MAX_PATH_UTF8_BYTES,
    MAX_TOTAL_UNCOMPRESSED_BYTES,
    SourceAnalysisToolsetFactory,
    _validate_entries,
)
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "source-tool-recognizable-secret"


def producer_fixture() -> dict:
    fixture = Path(__file__).resolve().parents[2] / "api/testdata/source-member-paths.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_source_read_prepares_only_the_requested_window_off_loop(tmp_path, monkeypatch):
    loop_thread = threading.get_ident()
    preparations: list[tuple[int, int]] = []
    prepare = source_tools_module.bounded_line_window

    def observed(lines, **kwargs):
        preparations.append((threading.get_ident(), len(lines)))
        return prepare(lines, **kwargs)

    monkeypatch.setattr(source_tools_module, "bounded_line_window", observed)

    async def scenario():
        client = ReadOnlyArtifactClient()
        payload = b"x\n" * (MAX_FILE_UNCOMPRESSED_BYTES // 2)
        ref = client.seed("inputs", "source", "application/zip", make_zip({"large.py": payload}))
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        try:
            result = await tools["read_source"]("large.py", start_line=2, max_lines=1)
            assert result == {
                "path": "large.py",
                "size": len(payload),
                "totalLines": len(payload) // 2,
                "startLine": 2,
                "endLine": 2,
                "text": "x\n",
                "truncated": True,
                "partialLine": False,
            }
            assert len(preparations) == 1
            assert preparations[0][0] != loop_thread
            assert preparations[0][1] == 1
        finally:
            await tools["read_source"].close()

    asyncio.run(scenario())


@pytest.mark.parametrize("content", ["", "\n", "a\r\nb\rc\nd", "a\u2028b\n終\r", "終" * 100_000])
def test_source_read_streamed_windows_keep_response_semantics(tmp_path, content):
    async def scenario():
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", make_zip({"source.py": content}))
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        lines = split_lines(content, keepends=True)
        try:
            for start in [1, 2, len(lines) + 1]:
                for count in [1, 3]:
                    if lines and start > len(lines):
                        with pytest.raises(ToolInputError, match="line count"):
                            await tools["read_source"](
                                "source.py", start_line=start, max_lines=count
                            )
                        continue
                    text, end, partial = source_tools_module.bounded_line_window(
                        lines,
                        start_line=start,
                        max_lines=count,
                        max_bytes=source_tools_module.MAX_VISIBLE_READ_BYTES,
                    )
                    result = await tools["read_source"](
                        "source.py", start_line=start, max_lines=count
                    )
                    assert result["text"] == text
                    assert result["endLine"] == end
                    assert result["totalLines"] == len(lines)
                    assert result["partialLine"] == partial
                    assert result["truncated"] == (partial or end < len(lines))
        finally:
            await tools["read_source"].close()

    asyncio.run(scenario())


def test_cancelled_source_read_keeps_preparation_owned_until_close(tmp_path, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    prepare = source_tools_module.bounded_line_window

    def held(lines, **kwargs):
        assert threading.get_ident() != loop_thread
        entered.set()
        try:
            if not release.wait(5):
                raise RuntimeError("read preparation was not released")
            return prepare(lines, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(source_tools_module, "bounded_line_window", held)

    async def scenario():
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", make_zip({"source.py": "x\n"}))
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        reading = asyncio.create_task(tools["read_source"]("source.py"))
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
            closing = asyncio.create_task(tools["read_source"].close())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not closing.done()
            assert (tmp_path / "source/source.py").exists()
            assert not finished.is_set()
        finally:
            release.set()
            await asyncio.gather(reading, return_exceptions=True)
            if closing is not None:
                await closing
            else:
                await tools["read_source"].close()
        assert finished.is_set()
        assert not (tmp_path / "source").exists()

    asyncio.run(scenario())


def test_shared_source_member_paths_match_producer_fixture() -> None:
    cases = producer_fixture()
    for path in cases["valid"]:
        assert source_tools_module._validated_member_path(path) == path
    for path in cases["invalid"]:
        with pytest.raises(ValueError):
            source_tools_module._validated_member_path(path)


def test_shared_source_readability_lists_match_producer_fixture() -> None:
    cases = producer_fixture()
    assert sorted(source_tools_module.IGNORED_DIRECTORY_NAMES) == sorted(
        cases["ignoredDirectoryNames"]
    )
    assert sorted(source_tools_module.BINARY_EXTENSIONS) == sorted(cases["binaryExtensions"])


def test_shared_source_bundle_openability_matches_producer_fixture(tmp_path: Path) -> None:
    errors = {
        "unreadable": "no readable UTF-8 files",
        "nested": "nests an entry below a regular file",
    }
    for index, bundle in enumerate(producer_fixture()["bundles"]):
        non_utf8 = set(bundle.get("nonUtf8", []))
        # The Go producer stores member names NFC-normalized.
        members: dict[str, str | bytes] = {
            unicodedata.normalize("NFC", path): b"\xff\xfe\n" if path in non_utf8 else "text\n"
            for path in bundle["files"]
        }
        staging = tmp_path / f"bundle-{index}"
        if bundle["openable"]:
            files, _, _ = source_tools_module._extract_archive(make_zip(members), staging)
            assert files, bundle["name"]
        else:
            with pytest.raises(ValueError, match=errors[bundle["reason"]]):
                source_tools_module._extract_archive(make_zip(members), staging)


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

        fixed = await tools["search_source"]("FastAPI", "src/*.py")
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


def test_source_files_named_like_ignored_directories_remain_visible(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        source = client.seed(
            "inputs",
            "named-files",
            "application/zip",
            make_zip(
                {
                    "scripts/build": "needle build\n",
                    "bin/target": "needle target\n",
                    "docs/coverage": "needle coverage\n",
                    "src/dist": "needle dist\n",
                    "vendor": "needle vendor\n",
                    "src/app.py": "normal source\n",
                    "node_modules/": "",
                    "node_modules/pkg/index.js": "ignored dependency",
                    "build/": "",
                    "build/out.js": "ignored build output",
                    ".git/config": "ignored metadata",
                }
            ),
        )
        tools = await make_tools(tmp_path, client, WorkerState())
        opened = await tools["open_source_archive"]("inputs", "named-files", source.revision)
        assert opened["fileCount"] == 6
        assert opened["ignoredCount"] == 5
        named = {"scripts/build", "bin/target", "docs/coverage", "src/dist", "vendor"}
        listed = await tools["list_source_files"]()
        assert {item["path"] for item in listed["files"]} == named | {"src/app.py"}
        searched = await tools["search_source"]("needle")
        assert {item["path"] for item in searched["matches"]} == named
        for path in named:
            assert (await tools["read_source"](path))["text"] == f"needle {Path(path).name}\n"

        with_vendor_dir = client.seed(
            "inputs",
            "vendor-directory",
            "application/zip",
            make_zip({"vendor/lib/x.go": "ignored", "src/app.py": "visible"}),
        )
        reopened = await tools["open_source_archive"](
            "inputs", "vendor-directory", with_vendor_dir.revision
        )
        assert reopened["fileCount"] == 1 and reopened["ignoredCount"] == 1
        assert [item["path"] for item in (await tools["list_source_files"]())["files"]] == [
            "src/app.py"
        ]

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


def test_search_and_read_number_lines_like_read_file(tmp_path: Path) -> None:
    async def scenario() -> None:
        # str.splitlines() would also break at the form feed, U+2028, NEL and
        # vertical tab, numbering "seven" as line 8 instead of line 4.
        content = "one\x0ctwo\u2028three\nfour\x85five\x0bsix\rsixth\r\nseven\n"
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", make_zip({"main.py": content}))
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)

        found = await tools["search_source"]("seven")
        assert [(item["path"], item["line"]) for item in found["matches"]] == [("main.py", 4)]
        separated = await tools["search_source"]("three")
        assert separated["matches"][0]["line"] == 1
        assert separated["matches"][0]["text"] == "one\x0ctwo\u2028three"

        read = await tools["read_source"]("main.py", start_line=2, max_lines=2)
        assert read["totalLines"] == 4
        assert read["text"] == "four\x85five\x0bsix\rsixth\r\n"
        assert (read["startLine"], read["endLine"], read["truncated"]) == (2, 3, True)
        with pytest.raises(ValueError, match="start_line"):
            await tools["read_source"]("main.py", start_line=5)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("*", ["main.py"]),
        ("**", ["main.py", "src/a.py", "src/x/a.py"]),
        ("**/*", ["main.py", "src/a.py", "src/x/a.py"]),
    ],
)
def test_single_star_matches_only_top_level_files(
    tmp_path: Path, pattern: str, expected: list[str]
) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed(
            "inputs",
            "source",
            "application/zip",
            make_zip({"main.py": "hit\n", "src/a.py": "hit\n", "src/x/a.py": "hit\n"}),
        )
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        listed = await tools["list_source_files"](pattern)
        assert [item["path"] for item in listed["files"]] == expected
        found = await tools["search_source"]("hit", pattern)
        assert [item["path"] for item in found["matches"]] == expected

    asyncio.run(scenario())


def test_search_deadline_is_checked_between_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", make_zip({"big.py": "hit\n" * 50}))
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        session = tools["search_source"]._session
        ticks = iter([0.0, 0.0, 0.0])
        monkeypatch.setattr(
            source_tools_module,
            "time",
            SimpleNamespace(monotonic=lambda: next(ticks, 1e9)),
        )
        result = session._search_files(tuple(session._files.values()), "hit", False, False, 100)
        assert len(result["matches"]) == 1
        assert result["truncated"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("**/*.py", ["main.py", "src/a.py", "src/x/a.py"]),
        ("src/**/*.py", ["src/a.py", "src/x/a.py"]),
        ("src/*.py", ["src/a.py"]),
    ],
)
def test_source_globs_treat_double_star_as_recursive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str, expected: list[str]
) -> None:
    compiled = 0
    normalize = paths_module.normalize_project_glob

    def counted(value: str) -> str:
        nonlocal compiled
        compiled += 1
        return normalize(value)

    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed(
            "inputs",
            "source",
            "application/zip",
            make_zip({"main.py": "hit\n", "src/a.py": "hit\n", "src/x/a.py": "hit\n"}),
        )
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        monkeypatch.setattr(paths_module, "normalize_project_glob", counted)
        listed = await tools["list_source_files"](pattern)
        assert [item["path"] for item in listed["files"]] == expected
        found = await tools["search_source"]("hit", pattern)
        assert [item["path"] for item in found["matches"]] == expected

    asyncio.run(scenario())
    # Each call compiles its glob once instead of once per member path.
    assert compiled == 2


# 41 "**" components between 41 "d" components, then "**/*.py": 256 characters,
# the most _validate_pattern admits. Only the deep fixture paths match it.
MULTI_STAR_GLOB = "/".join(["**", "d?*"] * 3 + ["**", "d*"] * 38) + "/**/*.py"
SELECTION_PATTERNS = ("**/*", "**/f0002?.*", MULTI_STAR_GLOB)


@pytest.mark.parametrize("entries", [MAX_ARCHIVE_ENTRIES, 50_000])
def test_source_path_selection_runs_off_the_event_loop_at_the_entry_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: int
) -> None:
    assert len(MULTI_STAR_GLOB) == 256
    paths = [archive_path(index) for index in range(entries)]
    assert {len(path.encode("utf-8")) for path in paths} == {100, MAX_PATH_UTF8_BYTES}
    archive = make_zip({path: f"line {index}\n" for index, path in enumerate(paths)})
    settings = runtime_settings()
    # The operation deadline is wall-clock time. This test checks where the
    # matching runs, not how fast, so a loaded machine must not expire it.
    settings.request_timeout_seconds = 600
    if entries > MAX_ARCHIVE_ENTRIES:
        # LIM-GIT-16: the Server may raise the entry bound to 50,000.
        settings.source_archive_limits = default_source_archive_limits().model_copy(
            update={"max_entries": entries}
        )

    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", archive)
        tools = await make_tools(tmp_path, client, WorkerState(), settings)
        opened = await tools["open_source_archive"]("inputs", "source", ref.revision)
        assert opened["fileCount"] == entries
        session = tools["list_source_files"]._session
        # The inline implementation this change replaced, computed before the
        # probe records matching threads.
        expected = {
            pattern: former_selection(tuple(session._files.values()), pattern)
            for pattern in SELECTION_PATTERNS
        }
        assert len(expected["**/*"]) == entries
        assert len(expected["**/f0002?.*"]) == 10
        assert 0 < len(expected[MULTI_STAR_GLOB]) < MAX_LIST_RESULTS
        loop_thread = threading.get_ident()
        try:
            for pattern, limit in zip(SELECTION_PATTERNS, (MAX_LIST_RESULTS, 4, 100), strict=True):
                matched = expected[pattern]
                # The probe records the first page and the search; later pages
                # run the same code without its per-path overhead.
                with monkeypatch.context() as patch:
                    probe = OffLoopProbe(patch)
                    page = await probe.held(tools["list_source_files"](pattern, 0, limit))
                    assert probe.matched == entries
                    found = await probe.held(tools["search_source"]("line", pattern))
                    assert probe.matched == entries
                assert probe.match_threads and probe.order_threads
                assert loop_thread not in probe.match_threads | probe.order_threads

                offset = 0
                while True:
                    assert page == former_page(matched, offset, limit)
                    if not page["truncated"]:
                        break
                    offset += limit
                    page = await tools["list_source_files"](pattern, offset, limit)
                assert offset + len(page["files"]) == len(matched)
                assert found == session._search_files(matched, "line", False, False, 100)
                assert found["scannedFiles"] == min(len(matched), 100)
        finally:
            await tools["list_source_files"].close()

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["list_source_files", "search_source"])
def test_cancelled_path_selection_keeps_the_session_until_its_thread_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed(
            "inputs",
            "source",
            "application/zip",
            make_zip({"main.py": "hit\n", "src/a.py": "hit\n"}),
        )
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        source = tmp_path / "source"
        probe = OffLoopProbe(monkeypatch)
        removed_after_selection: list[bool] = []
        remove = source_tools_module._remove_path

        def recorded_remove(path: Path) -> None:
            removed_after_selection.append(probe.finished.is_set())
            remove(path)

        monkeypatch.setattr(source_tools_module, "_remove_path", recorded_remove)
        call = (
            tools["list_source_files"]("**/*")
            if name == "list_source_files"
            else tools["search_source"]("hit")
        )
        probe.arm()
        task = asyncio.create_task(call)
        try:
            assert await asyncio.to_thread(probe.entered.wait, 5)
            task.cancel()
            closing = asyncio.create_task(tools[name].close())
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.02)
            # close() waits for the guard, which owns the still running thread.
            assert not closing.done()
            assert source.is_dir()
            probe.release.set()
            await closing
        finally:
            probe.release.set()
        assert probe.finished.is_set()
        assert removed_after_selection == [True]
        assert not source.exists()
        assert probe.match_threads
        assert threading.get_ident() not in probe.match_threads

    asyncio.run(scenario())


def test_invalid_selection_is_rejected_before_a_worker_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", make_zip({"main.py": "hit\n"}))
        tools = await make_tools(tmp_path, client, WorkerState())
        await tools["open_source_archive"]("inputs", "source", ref.revision)
        probe = OffLoopProbe(monkeypatch)
        listing, search = tools["list_source_files"], tools["search_source"]
        rejected: tuple[Callable[[], Awaitable[object]], ...] = (
            lambda: listing("../main.py"),
            lambda: listing("src/**.py"),
            lambda: listing("**/*", -1),
            lambda: listing("**/*", 0, 0),
            lambda: listing("**/*", 0, MAX_LIST_RESULTS + 1),
            lambda: search("hit", "x" * 257),
            lambda: search("hit", "**/*", max_results=0),
        )
        for call in rejected:
            with pytest.raises(ToolInputError) as error:
                await call()
            assert error.value.code == "tool_input_invalid"
        assert probe.matched == 0 and not probe.finished.is_set()
        # The session stays usable: validation failed before the guard ran.
        assert (await tools["list_source_files"]())["total"] == 1

    asyncio.run(scenario())


def test_source_tools_enforce_shared_memory_binding_visibility(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        hidden = client.seed(
            "analysis", "memory.hidden", "application/zip", make_zip({"hidden.py": "x"})
        )
        allowed = client.seed(
            "inputs",
            "memory.workflow_input",
            "application/zip",
            make_zip({"main.py": "print('ok')\n"}),
        )
        # Simulate provenance accumulated by another trusted wrapper sharing
        # this allocation client.
        await client.read_artifact(hidden)
        tools = await make_tools(tmp_path, client, WorkerState())
        calls_before = client.read_calls

        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["open_source_archive"](hidden.namespace, hidden.name, hidden.revision)
        assert client.read_calls == calls_before
        assert not (tmp_path / "source").exists()

        opened = await tools["open_source_archive"](
            allowed.namespace, allowed.name, allowed.revision
        )
        assert opened["fileCount"] == 1
        observed = {
            (ref.namespace, ref.name) for tool in tools.values() for ref in tool.known_exact_refs
        }
        assert (hidden.namespace, hidden.name) not in observed
        assert (allowed.namespace, allowed.name) in observed

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
            resources=WorkerState().resources,
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
                resources=WorkerState().resources,
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
    settings: RuntimeSettings | None = None,
) -> dict[str, object]:
    factory = SourceAnalysisToolsetFactory(lambda _allocation, _settings: client)
    selected = await factory.create_selected(
        selected=["open_source_archive", "list_source_files", "search_source", "read_source"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="analysis",
        runtime_settings=settings or runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        resources=state.resources,
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


def archive_path(index: int) -> str:
    """Return a distinct member path of a repository-like archive.

    Every 97th path nests over 41 short "d" directories (enough for
    MULTI_STAR_GLOB) and every 7th nests eight long directories; both fill
    MAX_PATH_UTF8_BYTES exactly. The others are 100 bytes long. Paths of one
    shape share their directories, which keeps extraction and cleanup cheap.
    """

    leaf = f"g{index // 100:04d}/f{index:05d}.{('py', 'txt', 'js', 'go')[index % 4]}"
    if index % 97 == 0:
        prefix, width, size = "d", 10, MAX_PATH_UTF8_BYTES
    elif index % 7 == 0:
        prefix, width, size = "s", 60, MAX_PATH_UTF8_BYTES
    else:
        prefix, width, size = "s", 30, 100
    stem = f"{prefix}{width:02d}"
    component = 0
    while len(stem) + 1 + width + 2 + len(leaf) < size:
        stem += f"/{prefix}{component:02d}" + "x" * (width - 3)
        component += 1
    pad = size - len(stem) - 2 - len(leaf)
    return f"{stem}/{'p' * pad}/{leaf}"


def former_selection(files: tuple[Any, ...], pattern: str) -> tuple[Any, ...]:
    """The inline list_files/search selection that ran on the event loop before V414-005."""

    matches = source_tools_module._path_matcher(pattern)
    matched = [item for item in files if matches(item.path)]
    matched.sort(key=lambda item: item.path)
    return tuple(matched)


def former_page(matched: tuple[Any, ...], offset: int, limit: int) -> dict[str, Any]:
    page = matched[offset : offset + limit]
    return {
        "files": [{"path": item.path, "size": item.size} for item in page],
        "offset": offset,
        "limit": limit,
        "total": len(matched),
        "truncated": offset + len(page) < len(matched),
    }


class OffLoopProbe:
    """Record the threads that match and order archive paths; hold one match.

    The order and selection hooks are installed with raising=False, so an
    implementation without them fails the thread assertions, not the setup.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.match_threads: set[int] = set()
        self.order_threads: set[int] = set()
        self.matched = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self._hold = False
        matcher = source_tools_module._path_matcher
        select = getattr(source_tools_module, "_select_files", None)

        def recorded_matcher(pattern: str) -> Callable[[str], bool]:
            matches = matcher(pattern)

            def recorded(path: str) -> bool:
                self.match_threads.add(threading.get_ident())
                self.matched += 1
                if self._hold:
                    self._hold = False
                    self.entered.set()
                    if not self.release.wait(2):
                        raise AssertionError("the held path match was not released")
                return matches(path)

            return recorded

        def recorded_order(item: Any) -> str:
            self.order_threads.add(threading.get_ident())
            return item.path

        def recorded_select(*args: Any) -> Any:
            if select is None:
                raise AssertionError("_select_files is absent")
            try:
                return select(*args)
            finally:
                self.finished.set()

        monkeypatch.setattr(source_tools_module, "_path_matcher", recorded_matcher)
        monkeypatch.setattr(source_tools_module, "_path_order", recorded_order, raising=False)
        monkeypatch.setattr(source_tools_module, "_select_files", recorded_select, raising=False)

    def reset(self) -> None:
        self.matched = 0
        self.entered.clear()
        self.release.clear()
        self.finished.clear()

    def arm(self) -> None:
        self.reset()
        self._hold = True

    async def held[T](self, call: Awaitable[T]) -> T:
        """Await ``call`` while its first path match waits and the loop keeps running."""

        self.arm()
        task = asyncio.ensure_future(call)
        try:
            if not await asyncio.to_thread(self.entered.wait, 5):
                raise AssertionError("the call never matched a path")
            beats = 0

            async def heartbeat() -> None:
                nonlocal beats
                for _ in range(3):
                    await asyncio.sleep(0)
                    beats += 1

            await asyncio.create_task(heartbeat())
            assert beats == 3
            assert not task.done()
        finally:
            self.release.set()
        return await task


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
        return ArtifactValue(
            artifact=exact,
            media_type=stored.media_type,
            data=stored.data,
            binding_created_at=datetime.now(UTC),
            revision_created_at=datetime.now(UTC),
        )


def test_server_source_limits_reach_factory_and_extraction(tmp_path: Path) -> None:
    from contractor_runtime.contracts.settings import SourceArchiveLimits

    payload = b"# source line\n" * ((5 * 1024 * 1024) // len(b"# source line\n") + 1)
    archive = make_zip({"large.py": payload})
    with pytest.raises(ValueError, match="file limit"):
        source_tools_module._extract_archive(archive, tmp_path / "default")

    async def scenario() -> None:
        client = ReadOnlyArtifactClient()
        ref = client.seed("inputs", "source", "application/zip", archive)
        settings = runtime_settings()
        settings.source_archive_limits = SourceArchiveLimits(
            max_expanded_bytes=128 * 1024 * 1024,
            max_file_bytes=16 * 1024 * 1024,
            max_entries=10000,
        )
        work = tmp_path / "configured"
        work.mkdir()
        factory = SourceAnalysisToolsetFactory(lambda _allocation, _settings: client)
        selected = await factory.create_selected(
            selected=["open_source_archive", "read_source"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="analysis",
            runtime_settings=settings,
            workspace=AllocationWorkspace(root=tmp_path, path=work),
            resources=WorkerState().resources,
        )
        result = await selected["open_source_archive"](ref.namespace, ref.name, ref.revision)
        assert result["fileCount"] == 1
        assert result["totalUncompressedBytes"] == len(payload)
        assert (work / "source" / "large.py").read_bytes() == payload
        result = await selected["read_source"]("large.py", max_lines=2)
        assert result["text"] == "# source line\n" * 2
        assert result["truncated"] is True

    asyncio.run(scenario())


def test_configured_source_declared_and_streamed_budgets() -> None:
    from contractor_runtime.contracts.settings import SourceArchiveLimits

    limits = SourceArchiveLimits(max_expanded_bytes=70, max_file_bytes=5, max_entries=14)
    members = [zipfile.ZipInfo(f"part-{index}.txt") for index in range(14)]
    for member in members:
        member.file_size = 5
    _validate_entries(members, limits)
    with pytest.raises(ValueError, match="uncompressed limit"):
        _validate_entries(members, limits.model_copy(update={"max_expanded_bytes": 69}))
    with pytest.raises(ValueError, match="entry limit"):
        _validate_entries(members, limits.model_copy(update={"max_entries": 13}))
    with (
        zipfile.ZipFile(io.BytesIO(make_zip({"part.txt": "123456"}))) as archive,
        pytest.raises(ValueError, match="streamed limit"),
    ):
        source_tools_module._read_member_bounded(archive, archive.infolist()[0], limits)
