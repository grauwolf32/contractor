from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import jcs
import pytest
from test_projectfs_zip import archive, workspace_inputs
from test_projectfs_zip import settings as workspace_settings

from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.metrics import MetricsState
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    ManagedWorkspaceTree,
    MemoryWorkspaceProvider,
    WorkspaceSnapshot,
    WorkspaceTextFile,
    hydrate_workspace,
)
from contractor_runtime.toolsets import code_analysis, code_analysis_languages
from contractor_runtime.toolsets.code_analysis import (
    MAX_PREVIEW_BYTES,
    MAX_PREVIEW_LINES,
    MAX_RESULT_BYTES,
    CodeAnalysisError,
    CodeAnalysisToolsetFactory,
)
from contractor_runtime.toolsets.code_analysis_languages import (
    EXTENSION_LANGUAGES,
    Language,
    load_parser,
    parse_symbols,
)
from contractor_runtime.workspace import AllocationWorkspace

LANGUAGE_SAMPLES = {
    Language.PYTHON: ("sample.py", "def target():\n    pass\n", "target"),
    Language.JAVASCRIPT: ("sample.js", "function target() {}\n", "target"),
    Language.TYPESCRIPT: ("sample.ts", "function target(): void {}\n", "target"),
    Language.TSX: ("sample.tsx", "function Target() { return <div/>; }\n", "Target"),
    Language.GO: ("sample.go", "package sample\nfunc Target() {}\n", "Target"),
    Language.RUST: ("sample.rs", "fn target() {}\n", "target"),
    Language.JAVA: ("Sample.java", "class Target {}\n", "Target"),
    Language.KOTLIN: ("sample.kt", "fun target() {}\n", "target"),
    Language.C: ("sample.c", "void target(void) {}\n", "target"),
    Language.CPP: ("sample.cpp", "void target() {}\n", "target"),
    Language.C_SHARP: ("Sample.cs", "class Target {}\n", "Target"),
    Language.RUBY: ("sample.rb", "def target\nend\n", "target"),
    Language.PHP: ("sample.php", "<?php function target() {}\n", "target"),
    Language.SCALA: ("sample.scala", "def target(): Unit = {}\n", "target"),
    Language.SWIFT: ("sample.swift", "func target() {}\n", "target"),
    Language.LUA: ("sample.lua", "function target() end\n", "target"),
    Language.ELIXIR: (
        "sample.ex",
        "defmodule Sample do\n  def target do\n    :ok\n  end\nend\n",
        "target",
    ),
    Language.HASKELL: ("sample.hs", "target x = x\n", "target"),
    Language.BASH: ("sample.sh", "target() { :; }\n", "target"),
}


@pytest.mark.parametrize("language", list(Language))
def test_every_fixed_language_parser_extracts_a_structural_definition(
    language: Language,
) -> None:
    path, text, expected = LANGUAGE_SAMPLES[language]
    parsed = parse_symbols(load_parser(language), text.encode(), path, language, 100)
    assert expected in {item.name for item in parsed.symbols}
    assert not parsed.parse_error


def test_extension_registry_retains_the_v1_surface() -> None:
    expected = {
        ".bash",
        ".c",
        ".cc",
        ".cjs",
        ".cpp",
        ".cs",
        ".cxx",
        ".ex",
        ".exs",
        ".go",
        ".h",
        ".hpp",
        ".hs",
        ".hxx",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".lhs",
        ".lua",
        ".mjs",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".sc",
        ".scala",
        ".sh",
        ".swift",
        ".ts",
        ".tsx",
    }
    assert expected == set(EXTENSION_LANGUAGES)


def test_factory_rejects_missing_workspace_and_unadvertised_graph_selection(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        factory = CodeAnalysisToolsetFactory()
        metrics = MetricsState()
        common = {
            "allocation_id": "allocation-code-analysis",
            "run_id": "run-code-analysis",
            "namespace": "analysis",
            "runtime_settings": RuntimeSettings(
                llm_gateway_url="https://llm.example/v1",
                llm_gateway_token="temporary-token",
                artifact_api_url="https://server.example/private/v1/artifacts",
                request_timeout_seconds=10,
            ),
            "workspace": AllocationWorkspace(root=tmp_path, path=tmp_path),
            "state": SimpleNamespace(metrics=metrics),
        }
        with pytest.raises(CodeAnalysisError) as missing:
            await factory.create_selected(selected=("search_def",), **common)
        assert missing.value.code == "workspace_required"
        with pytest.raises(ValueError, match="unavailable selected code-analysis tools"):
            await factory.create_selected(
                selected=("find_callers",),
                project_workspace=MutableReader({"a.py": "def a(): pass\n"}),
                **common,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("storage", ["local", "memory"])
@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_shallow_tools_match_across_real_workspace_providers_and_modes(
    tmp_path: Path,
    storage: str,
    mode: str,
) -> None:
    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        source_spec, reader = workspace_inputs(
            [
                (
                    "source",
                    "",
                    archive(
                        {
                            "src/app.py": b"def Target():\n    return 1\n",
                            "src/lib.go": b"package src\nfunc Helper() {}\n",
                            "src/contract.sol": b"contract Hidden {}\n",
                            "src/logo.bin": b"\x00\xff",
                            "README.md": b"Target is documented here.\n",
                        }
                    ),
                )
            ]
        )
        source_spec.mode = mode  # type: ignore[assignment]
        provider = (
            LocalWorkspaceProvider(workspace_settings("local", tmp_path / "local"))
            if storage == "local"
            else MemoryWorkspaceProvider(workspace_settings("memory"))
        )
        session = await hydrate_workspace(
            provider=provider,
            spec=source_spec,
            artifact_reader=reader,
            allocation_id=f"{storage}-{mode}",
            timeout_seconds=5,
        )
        tools, _ = await _tools(session.reader_view(), tmp_path)
        listed = await tools["list_symbols"](limit=20)
        found = await tools["search_def"]("target", language="python")
        await tools["list_symbols"].close()
        await tools["search_def"].close()
        await session.close()
        await provider.cleanup(session.storage)
        return listed, found

    listed, found = asyncio.run(scenario())
    assert [
        (
            item["name"],
            item["path"],
            item["line"],
            item["nodeType"],
            item["language"],
        )
        for item in listed["items"]
    ] == [
        ("Target", "src/app.py", 1, "function_definition", "python"),
        ("Helper", "src/lib.go", 2, "function_declaration", "go"),
    ]
    assert listed["coverage"] == {
        "analyzedFiles": 2,
        "analyzedBytes": 56,
        "binaryFiles": 1,
        "unsupportedSourceFiles": 1,
        "oversizedFiles": 0,
        "parseErrors": 0,
        "incomplete": False,
        "reasons": [],
    }
    assert [item["name"] for item in found["items"]] == ["Target"]
    assert found["items"][0]["path"] == "src/app.py"
    assert found["items"][0]["preview"].startswith("def Target")


def test_pagination_is_deterministic_integrity_protected_and_query_bound(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        reader = MutableReader(
            {
                "b.py": "def second():\n    pass\n",
                "a.py": "def first():\n    pass\n",
            }
        )
        tools, _ = await _tools(reader, tmp_path)
        first = await tools["list_symbols"](limit=1)
        assert [item["name"] for item in first["items"]] == ["first"]
        assert first["truncated"] and first["observedTotal"] == 2
        assert isinstance(first["nextCursor"], str)
        second = await tools["list_symbols"](cursor=first["nextCursor"], limit=1)
        assert [item["name"] for item in second["items"]] == ["second"]
        assert not second["truncated"] and second["nextCursor"] is None

        tampered = first["nextCursor"][:-1] + ("A" if first["nextCursor"][-1] != "A" else "B")
        with pytest.raises(CodeAnalysisError, match="code_analysis_cursor_invalid"):
            await tools["list_symbols"](cursor=tampered, limit=1)
        with pytest.raises(CodeAnalysisError, match="code_analysis_cursor_invalid"):
            await tools["search_def"]("first", cursor=first["nextCursor"], limit=1)

        other, _ = await _tools(reader, tmp_path)
        with pytest.raises(CodeAnalysisError, match="code_analysis_cursor_invalid"):
            await other["list_symbols"](cursor=first["nextCursor"], limit=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_edit_invalidates_cache_and_stale_cursor_but_fresh_call_sees_change(
    tmp_path: Path,
    mode: str,
) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [("source", "", archive({"a.py": b"def first():\n    pass\n"}))]
        )
        spec.mode = mode  # type: ignore[assignment]
        provider = MemoryWorkspaceProvider(workspace_settings("memory"))
        workspace = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id=mode,
            timeout_seconds=5,
        )
        tools, metrics = await _tools(workspace.reader_view(), tmp_path)
        first = await tools["list_symbols"](limit=1)
        assert first["nextCursor"] is None
        await workspace.write_text("b.py", "def second():\n    pass\n")
        fresh = await tools["list_symbols"](limit=1)
        assert fresh["nextCursor"] is not None
        await workspace.write_text("c.py", "def third():\n    pass\n")
        with pytest.raises(CodeAnalysisError) as changed:
            await tools["list_symbols"](cursor=fresh["nextCursor"], limit=1)
        assert changed.value.code == "code_analysis_workspace_changed"
        assert changed.value.retryable
        latest = await tools["search_def"]("third")
        assert [item["name"] for item in latest["items"]] == ["third"]
        rendered = metrics.tool_calls[-2].model_dump_json()
        assert "b.py" not in rendered and "second" not in rendered
        await tools["list_symbols"].close()
        await workspace.close()
        await provider.cleanup(workspace.storage)

    asyncio.run(scenario())


def test_search_def_has_no_textual_fallback_and_filters_structural_rows(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        reader = MutableReader(
            {
                "src/calls.py": "# Missing is only mentioned\nMissing()\n",
                "src/defs.py": "class Present:\n    pass\n",
                "outside.py": "def Present():\n    pass\n",
            }
        )
        tools, _ = await _tools(reader, tmp_path)
        absent = await tools["search_def"]("Missing")
        assert absent["items"] == [] and absent["observedTotal"] == 0
        present = await tools["search_def"]("pkg.present", path="src", language="python")
        assert [(item["name"], item["path"]) for item in present["items"]] == [
            ("Present", "src/defs.py")
        ]
        classes = await tools["list_symbols"](
            path="src", language="python", node_type="class_definition"
        )
        assert [item["name"] for item in classes["items"]] == ["Present"]

    asyncio.run(scenario())


def test_coverage_reports_binary_unsupported_oversized_and_parse_errors(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        huge = WorkspaceTextFile(
            path="huge.py",
            text="def Huge(): pass\n",
            size=code_analysis.MAX_SOURCE_FILE_BYTES + 1,
        )
        malformed_text = "def broken(\n"
        malformed = WorkspaceTextFile(
            path="broken.py",
            text=malformed_text,
            size=len(malformed_text.encode()),
        )
        snapshot = WorkspaceSnapshot(
            directories=(),
            files=(
                malformed,
                huge,
                WorkspaceTextFile("contract.sol", "contract X {}", 13),
                WorkspaceTextFile("README.md", "not source", 10),
            ),
            binary_paths=("image.bin",),
            digest="sha256:" + "1" * 64,
        )
        tools, _ = await _tools(StaticReader(snapshot), tmp_path)
        result = await tools["list_symbols"]()
        coverage = result["coverage"]
        assert coverage["binaryFiles"] == 1
        assert coverage["unsupportedSourceFiles"] == 1
        assert coverage["oversizedFiles"] == 1
        assert coverage["parseErrors"] == 1
        assert coverage["incomplete"]
        assert coverage["reasons"] == ["parse_errors"]

    asyncio.run(scenario())


def test_file_byte_symbol_and_result_limits_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        reader = MutableReader(
            {
                "a.py": "def a(): pass\ndef b(): pass\ndef c(): pass\n",
                "b.py": "def d(): pass\n",
            }
        )

        monkeypatch.setattr(code_analysis, "MAX_SOURCE_FILES", 1)
        tools, _ = await _tools(reader, tmp_path)
        file_limited = await tools["list_symbols"]()
        assert "file_limit" in file_limited["coverage"]["reasons"]

        monkeypatch.setattr(code_analysis, "MAX_SOURCE_FILES", 20_000)
        monkeypatch.setattr(code_analysis, "MAX_SOURCE_BYTES", 1)
        byte_tools, _ = await _tools(reader, tmp_path)
        byte_limited = await byte_tools["list_symbols"]()
        assert byte_limited["items"] == []
        assert byte_limited["coverage"]["reasons"] == ["byte_limit"]

        monkeypatch.setattr(code_analysis, "MAX_SOURCE_BYTES", 128 * 1024 * 1024)
        monkeypatch.setattr(code_analysis, "MAX_COMPACT_SYMBOLS", 1)
        symbol_tools, _ = await _tools(reader, tmp_path)
        symbol_limited = await symbol_tools["list_symbols"]()
        assert len(symbol_limited["items"]) == 1
        assert symbol_limited["coverage"]["reasons"] == ["symbol_limit"]
        assert len(jcs.canonicalize(symbol_limited)) <= MAX_RESULT_BYTES

    asyncio.run(scenario())


def test_encoded_result_limit_reduces_page_without_silent_truncation(
    tmp_path: Path,
) -> None:
    body = "\n".join("    value = '" + ("x" * 350) + "'" for _ in range(12))
    source = "\n".join(f"def target():\n{body}" for _ in range(100)) + "\n"

    async def scenario() -> None:
        tools, _ = await _tools(MutableReader({"many.py": source}), tmp_path)
        result = await tools["search_def"]("target", limit=200)
        assert result["observedTotal"] == 100
        assert 0 < len(result["items"]) < result["observedTotal"]
        assert result["truncated"] and result["nextCursor"]
        assert len(jcs.canonicalize(result)) <= MAX_RESULT_BYTES
        continued = await tools["search_def"]("target", cursor=result["nextCursor"], limit=200)
        assert continued["items"]

    asyncio.run(scenario())


def test_slow_parser_runs_off_loop_and_deadline_is_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = code_analysis_languages.parse_symbols

    def slow_parse(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original(*args, **kwargs)

    monkeypatch.setattr(code_analysis_languages, "parse_symbols", slow_parse)
    monkeypatch.setattr(code_analysis, "MAX_SCAN_SECONDS", 0.02)

    async def scenario() -> None:
        reader = MutableReader({"a.py": "def a(): pass\n", "b.py": "def b(): pass\n"})
        tools, _ = await _tools(reader, tmp_path)
        ticks = 0
        running = True

        async def heartbeat() -> None:
            nonlocal ticks
            while running:
                await asyncio.sleep(0.005)
                ticks += 1

        heartbeat_task = asyncio.create_task(heartbeat())
        result = await tools["list_symbols"]()
        ticks_before_completion = ticks
        running = False
        await heartbeat_task
        assert ticks_before_completion >= 2
        assert result["coverage"]["reasons"] == ["deadline"]
        assert result["coverage"]["analyzedFiles"] == 1

    asyncio.run(scenario())


def test_preview_metrics_and_close_keep_content_out_of_retained_reports(
    tmp_path: Path,
) -> None:
    canary = "PRIVATE_SYMBOL_CANARY"
    lines = [f"def {canary}():"] + [f"    value_{index} = {index}" for index in range(30)]

    async def scenario() -> None:
        reader = MutableReader({"private/secret.py": "\n".join(lines) + "\n"})
        tools, metrics = await _tools(reader, tmp_path)
        result = await tools["search_def"](canary)
        preview = result["items"][0]["preview"]
        assert len(preview.splitlines()) == MAX_PREVIEW_LINES
        assert len(preview.encode()) <= MAX_PREVIEW_BYTES
        assert len(jcs.canonicalize(result)) <= MAX_RESULT_BYTES

        report = metrics.tool_calls[-1].model_dump_json()
        assert canary not in report
        assert "private/secret.py" not in report
        assert json.loads(report)["arguments"] == {}

        session = tools["search_def"]._session
        assert session._file_cache
        assert all(
            not hasattr(value, "tree")
            and not hasattr(value, "source")
            and not hasattr(value, "text")
            for value in session._file_cache.values()
        )
        await tools["search_def"].close()
        assert session._file_cache == {}
        assert session._parsers == {}
        assert set(session._cursor_key) == {0}
        with pytest.raises(CodeAnalysisError) as closing:
            await tools["list_symbols"]()
        assert closing.value.code == "code_analysis_closing"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (lambda tools: tools["search_def"](""), "code_analysis_input_invalid"),
        (
            lambda tools: tools["search_def"]("x", path="../escape"),
            "code_analysis_input_invalid",
        ),
        (
            lambda tools: tools["search_def"]("x", language="brainfuck"),
            "code_analysis_input_invalid",
        ),
        (
            lambda tools: tools["list_symbols"](node_type="Bad Type"),
            "code_analysis_input_invalid",
        ),
        (lambda tools: tools["list_symbols"](limit=201), "code_analysis_input_invalid"),
    ],
)
def test_invalid_inputs_have_only_stable_errors(
    tmp_path: Path,
    call: Any,
    code: str,
) -> None:
    async def scenario() -> None:
        tools, _ = await _tools(MutableReader({"a.py": "def x(): pass\n"}), tmp_path)
        with pytest.raises(CodeAnalysisError) as rejected:
            await call(tools)
        assert rejected.value.code == code
        assert str(rejected.value) == f"Code analysis operation failed ({code})"

    asyncio.run(scenario())


class StaticReader:
    def __init__(self, snapshot: WorkspaceSnapshot) -> None:
        self.snapshot_value = snapshot

    async def snapshot(self) -> WorkspaceSnapshot:
        return self.snapshot_value

    async def read_text(self, path: str) -> str:
        for item in self.snapshot_value.files:
            if item.path == path:
                return item.text
        raise FileNotFoundError


class MutableReader:
    def __init__(self, files: dict[str, str]) -> None:
        self._tree = ManagedWorkspaceTree(
            directories=_directories(files),
            text_files=dict(files),
        )

    async def snapshot(self) -> WorkspaceSnapshot:
        return self._tree.snapshot()

    async def read_text(self, path: str) -> str:
        return self._tree.text_files[path]

    def write(self, path: str, text: str) -> None:
        self._tree.directories.update(_directories({path: text}))
        self._tree.text_files[path] = text


def _directories(files: dict[str, str]) -> set[str]:
    result: set[str] = set()
    for path in files:
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            result.add(str(parent))
            parent = parent.parent
    return result


async def _tools(
    reader: Any,
    tmp_path: Path,
) -> tuple[dict[str, Any], MetricsState]:
    metrics = MetricsState()
    tools = await CodeAnalysisToolsetFactory().create_selected(
        selected=("list_symbols", "search_def"),
        allocation_id="allocation-code-analysis",
        run_id="run-code-analysis",
        namespace="analysis",
        runtime_settings=RuntimeSettings(
            llm_gateway_url="https://llm.example/v1",
            llm_gateway_token="temporary-token",
            artifact_api_url="https://server.example/private/v1/artifacts",
            request_timeout_seconds=10,
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        state=SimpleNamespace(metrics=metrics),
        project_workspace=reader,
    )
    return dict(tools), metrics
