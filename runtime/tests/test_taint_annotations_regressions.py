from __future__ import annotations

import asyncio
import io
import stat
import tokenize
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from zipfile import ZipInfo

import pytest
from google.adk.tools import FunctionTool
from test_projectfs_zip import archive_infos, settings, workspace_inputs
from test_taint_annotations import LANGUAGE_CASES, make_tools

from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
    hydrate_workspace,
)
from contractor_runtime.toolsets.code_analysis.languages import Language, detect_language
from contractor_runtime.toolsets.taint_annotations.tools import (
    MAX_SOURCE_FILE_BYTES,
    TaintAnnotationError,
    _parse_target_file,
)

PROVIDERS = pytest.mark.parametrize("storage", ["local", "memory"])
MODES = pytest.mark.parametrize("mode", ["direct", "overlay"])


@asynccontextmanager
async def workspace_tools(
    tmp_path: Path, path: str, source: str, storage: str, mode: str
) -> AsyncIterator[tuple[DirectWorkspaceSession | OverlayWorkspaceSession, dict[str, Any]]]:
    # Stored ZIP members keep the 4 MiB repeated-text fixture within the
    # workspace importer's independent compression-ratio bound.
    member = ZipInfo(path)
    member.external_attr = (stat.S_IFREG | 0o644) << 16
    payload = archive_infos([(member, source.encode("utf-8"))])
    spec, reader = workspace_inputs([("source", "", payload)])
    spec.mode = mode  # type: ignore[assignment]
    provider = (
        LocalWorkspaceProvider(settings("local", tmp_path / "workspace"))
        if storage == "local"
        else MemoryWorkspaceProvider(settings("memory"))
    )
    workspace = await hydrate_workspace(
        provider=provider,
        spec=spec,
        artifact_reader=reader,
        allocation_id="annotation-regression",
        timeout_seconds=5,
    )
    tools, _ = await make_tools(tmp_path, workspace.writer_view())
    try:
        yield workspace, tools
    finally:
        await tools["annotate_trace"].close()
        await workspace.close()
        await provider.cleanup(workspace.storage)


@PROVIDERS
@MODES
@pytest.mark.parametrize("extension", ["js", "ts", "tsx"])
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("annotate_trace", {"target": "test"}),
        ("annotate_validate", {"arg": "value", "kind": "schema"}),
        ("annotate_sink", {"kind": "db.query"}),
    ],
)
def test_template_expression_annotation_never_changes_literal_content(
    tmp_path: Path,
    storage: str,
    mode: str,
    extension: str,
    tool_name: str,
    arguments: dict[str, str],
) -> None:
    source = "const text = `hello\n${function handler(value) { return value; }}\n`;\n"

    async def scenario() -> None:
        path = f"app.{extension}"
        async with workspace_tools(tmp_path, path, source, storage, mode) as (workspace, tools):
            with pytest.raises(TaintAnnotationError) as failure:
                await tools[tool_name](path, "handler", **arguments)
            assert failure.value.code == "taint_annotation_input_invalid"
            assert failure.value.retryable is False
            assert await workspace.read_text(path) == source

    asyncio.run(scenario())


@PROVIDERS
@MODES
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("annotate_trace", {"target": "?>"}),
        ("annotate_trace", {"args": "value?>:tainted"}),
        ("annotate_trace", {"calls": "callee?>"}),
        ("annotate_validate", {"arg": "value?>", "kind": "schema"}),
        ("annotate_validate", {"arg": "value", "kind": "?>"}),
        ("annotate_sink", {"kind": "?>"}),
        ("annotate_sink", {"kind": "db.query", "arg": "value?>"}),
    ],
)
def test_php_close_tags_are_rejected_in_every_emitted_field(
    tmp_path: Path, storage: str, mode: str, tool_name: str, arguments: dict[str, str]
) -> None:
    source = "<?php\nfunction handler($value) { return $value; }\n"

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.php", source, storage, mode) as (
            workspace,
            tools,
        ):
            with pytest.raises(TaintAnnotationError) as failure:
                await tools[tool_name]("app.php", "handler", **arguments)
            assert failure.value.code == "taint_annotation_input_invalid"
            assert not failure.value.retryable
            assert await workspace.read_text("app.php") == source

    asyncio.run(scenario())


@PROVIDERS
@MODES
@pytest.mark.parametrize("prefix", ["", "#!/usr/bin/env python3\n"])
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("annotate_trace", {"target": "coding:latin-1"}),
        ("annotate_trace", {"args": "coding:tainted"}),
        ("annotate_trace", {"calls": "encoding:latin-1"}),
        ("annotate_validate", {"arg": "value", "kind": "coding:latin-1"}),
        ("annotate_sink", {"kind": "coding:latin-1"}),
    ],
)
def test_python_annotation_cannot_introduce_an_encoding_cookie(
    tmp_path: Path,
    storage: str,
    mode: str,
    prefix: str,
    tool_name: str,
    arguments: dict[str, str],
) -> None:
    source = prefix + "def handler():\n    return 'привет'\n"

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.py", source, storage, mode) as (workspace, tools):
            with pytest.raises(TaintAnnotationError) as failure:
                await tools[tool_name]("app.py", "handler", **arguments)
            assert failure.value.code == "taint_annotation_input_invalid"
            assert not failure.value.retryable
            updated = await workspace.read_text("app.py")
            assert updated == source
            assert tokenize.detect_encoding(io.BytesIO(updated.encode()).readline)[0] == "utf-8"
            namespace: dict[str, Any] = {}
            exec(compile(updated.encode(), "app.py", "exec"), namespace)
            assert namespace["handler"]() == "привет"

    asyncio.run(scenario())


@PROVIDERS
@MODES
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("annotate_trace", {"target": "original"}),
        ("annotate_validate", {"arg": "value", "kind": "schema"}),
        ("annotate_sink", {"kind": "db.query"}),
    ],
)
@pytest.mark.parametrize(
    ("source", "line"),
    [
        ("def handler(): pass\n", 1),
        ("@first\n@second\ndef handler(): pass\n", 1),
        ("@first\n@second\ndef handler(): pass\n", 3),
        ("def handler(): return 1\n\ndef handler(): return 2\n", 3),
    ],
)
def test_explicit_selector_replay_survives_insertion_and_allocation_replacement(
    tmp_path: Path,
    storage: str,
    mode: str,
    newline: str,
    source: str,
    line: int,
    tool_name: str,
    arguments: dict[str, str],
) -> None:
    source = source.replace("\n", newline)

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.py", source, storage, mode) as (workspace, tools):
            request = {**arguments, "definition_line": line}
            first = await tools[tool_name]("app.py", "handler", **request)
            await tools["annotate_trace"].close()
            replacement, _ = await make_tools(tmp_path, workspace.writer_view())
            try:
                replay = await replacement[tool_name]("app.py", "handler", **request)
                assert replay == {**first, "changed": False}
                current_request = {**arguments, "definition_line": first["definitionLine"]}
                replay = await replacement[tool_name]("app.py", "handler", **current_request)
                assert not replay["changed"]
                await replacement["annotate_trace"](
                    "app.py", "handler", target="later", definition_line=first["definitionLine"]
                )
                current = await workspace.read_text("app.py")
                replay = await replacement[tool_name]("app.py", "handler", **request)
                assert replay == {
                    **first,
                    "definitionLine": first["definitionLine"] + 1,
                    "changed": False,
                }
                replay_of_noop = await replacement[tool_name](
                    "app.py", "handler", **current_request
                )
                assert replay_of_noop == replay
                assert await workspace.read_text("app.py") == current
            finally:
                await replacement["annotate_trace"].close()

    asyncio.run(scenario())


@PROVIDERS
@MODES
@pytest.mark.parametrize(
    ("tool_name", "arguments", "annotation"),
    [
        ("annotate_trace", {"target": "fit"}, "# @trace target=fit\n"),
        ("annotate_validate", {"arg": "v", "kind": "schema"}, "# @validate arg=v kind=schema\n"),
        ("annotate_sink", {"kind": "db.query"}, "# @sink kind=db.query arg=unknown\n"),
    ],
)
def test_exact_replay_at_file_limit_does_not_reserve_another_line(
    tmp_path: Path,
    storage: str,
    mode: str,
    tool_name: str,
    arguments: dict[str, str],
    annotation: str,
) -> None:
    definition = "def handler(): pass\n"
    filler_size = MAX_SOURCE_FILE_BYTES - len(annotation.encode()) - len(definition.encode())
    source = "#" + "x" * (filler_size - 2) + "\n" + definition

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.py", source, storage, mode) as (workspace, tools):
            first = await tools[tool_name]("app.py", "handler", **arguments)
            current = await workspace.read_text("app.py")
            assert len(current.encode()) == MAX_SOURCE_FILE_BYTES
            replay = await tools[tool_name]("app.py", "handler", **arguments)
            assert replay == {**first, "changed": False}
            assert await workspace.read_text("app.py") == current
            with pytest.raises(TaintAnnotationError) as failure:
                await tools["annotate_trace"]("app.py", "handler", target="extra")
            assert failure.value.code == "taint_annotation_capacity_exceeded"
            assert await workspace.read_text("app.py") == current

    asyncio.run(scenario())


@PROVIDERS
@MODES
@pytest.mark.parametrize(
    ("value", "callable_value"),
    [
        ("{handle = function(value) return value end}", False),
        ("wrap(function(value) return value end)", False),
        ("function(value) return value end", True),
        ("(function(value) return value end)", True),
    ],
)
def test_lua_only_direct_callable_assignments_are_annotation_targets(
    tmp_path: Path, storage: str, mode: str, value: str, callable_value: bool
) -> None:
    source = f"local handler = {value}\n"

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.lua", source, storage, mode) as (
            workspace,
            tools,
        ):
            if callable_value:
                result = await tools["annotate_trace"]("app.lua", "handler", target="test")
                assert result["changed"]
                assert await workspace.read_text("app.lua") == "-- @trace target=test\n" + source
            else:
                with pytest.raises(TaintAnnotationError) as failure:
                    await tools["annotate_trace"]("app.lua", "handler", target="test")
                assert failure.value.code == "taint_annotation_target_not_found"
                assert not failure.value.retryable
                assert await workspace.read_text("app.lua") == source

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("prefix", "target"),
    [
        ("", "test"),
        ("#!/usr/bin/env python3\n", "test"),
        ("# coding: utf-8\n", "test"),
        ("# first\n# second\n", "coding:latin-1"),
    ],
)
def test_safe_python_comments_preserve_encoding_and_non_ascii_values(
    tmp_path: Path, prefix: str, target: str
) -> None:
    source = prefix + "def handler():\n    return 'привет'\n"

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.py", source, "memory", "overlay") as (
            workspace,
            tools,
        ):
            await tools["annotate_trace"]("app.py", "handler", target=target)
            updated = await workspace.read_text("app.py")
            assert tokenize.detect_encoding(io.BytesIO(updated.encode()).readline)[0] == "utf-8"
            namespace: dict[str, Any] = {}
            exec(compile(updated.encode(), "app.py", "exec"), namespace)
            assert namespace["handler"]() == "привет"

    asyncio.run(scenario())


def test_replay_keeps_duplicate_names_ambiguous_without_a_selector(tmp_path: Path) -> None:
    source = (
        "# @trace target=test\ndef handler(): return 1\n\n"
        "# @trace target=test\ndef handler(): return 2\n"
    )

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.py", source, "memory", "overlay") as (
            workspace,
            tools,
        ):
            with pytest.raises(TaintAnnotationError) as failure:
                await tools["annotate_trace"]("app.py", "handler", target="test")
            assert failure.value.code == "taint_annotation_target_ambiguous"
            assert await workspace.read_text("app.py") == source

    asyncio.run(scenario())


def test_many_same_name_methods_keep_resolution_and_replay_unambiguous(tmp_path: Path) -> None:
    count = 1000
    source = "".join(
        f"class Handler{index}:\n    def handle(self): return {index}\n" for index in range(count)
    )

    async def scenario() -> None:
        async with workspace_tools(tmp_path, "app.py", source, "memory", "overlay") as (
            workspace,
            tools,
        ):
            with pytest.raises(TaintAnnotationError) as ambiguous:
                await tools["annotate_trace"]("app.py", "handle", target="last")
            assert ambiguous.value.code == "taint_annotation_target_ambiguous"
            arguments = {"target": "last", "definition_line": count * 2}
            first = await tools["annotate_trace"]("app.py", "handle", **arguments)
            current = await workspace.read_text("app.py")
            assert current.count("@trace") == 1
            replay = await tools["annotate_trace"]("app.py", "handle", **arguments)
            assert replay == {**first, "changed": False}
            assert await workspace.read_text("app.py") == current
            with pytest.raises(TaintAnnotationError) as stale:
                await tools["annotate_trace"](
                    "app.py", "handle", target="different", definition_line=count * 2
                )
            assert stale.value.code == "taint_annotation_target_not_found"
            assert await workspace.read_text("app.py") == current

    asyncio.run(scenario())


@pytest.mark.parametrize(("path", "source", "symbol", "_expected"), LANGUAGE_CASES)
def test_safe_annotations_preserve_parsed_declarations_in_every_language(
    tmp_path: Path, path: str, source: str, symbol: str, _expected: str
) -> None:
    async def scenario() -> None:
        async with workspace_tools(tmp_path, path, source, "memory", "overlay") as (
            workspace,
            tools,
        ):
            language = detect_language(path, header=Language.C)
            assert language is not None
            before = _parse_target_file(source.encode(), language)
            await tools["annotate_trace"](path, symbol, target="test")
            after = _parse_target_file((await workspace.read_text(path)).encode(), language)
            assert after.parse_error == before.parse_error
            assert [(item.name, item.node_type) for item in after.targets] == [
                (item.name, item.node_type) for item in before.targets
            ]

    asyncio.run(scenario())


def test_sink_tool_adk_description_uses_the_trace_sink_vocabulary(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with workspace_tools(
            tmp_path, "app.py", "def handler(): pass\n", "memory", "overlay"
        ) as (_, tools):
            declaration = FunctionTool(tools["annotate_sink"])._get_declaration()
            description = declaration.description or ""
            for label in ("db.query", "shell.exec", "http.request"):
                assert f'"{label}"' in description
            for obsolete in ("sql", "shell", "ssrf", "deserialize"):
                assert f'"{obsolete}"' not in description

    asyncio.run(scenario())
