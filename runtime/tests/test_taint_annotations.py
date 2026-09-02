from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.tools import FunctionTool
from test_projectfs_zip import archive, settings, workspace_inputs

from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.metrics import MetricsState
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    hydrate_workspace,
)
from contractor_runtime.toolsets.taint_annotations import (
    EXPORTED_TOOLS,
    MAX_SOURCE_FILE_BYTES,
    TaintAnnotationError,
    TaintAnnotationsToolsetFactory,
)
from contractor_runtime.workspace import AllocationWorkspace

LANGUAGE_CASES = (
    (
        "handler.py",
        '@app.get("/x")\n@auth\nasync def handler(value):\n    return value\n',
        "handler",
        "# @trace target=test\n@app.get",
    ),
    (
        "handler.js",
        "export const handler = async (value) => value;\n",
        "handler",
        "// @trace target=test\nexport const handler",
    ),
    (
        "handler.ts",
        "class C {\n  @get()\n  async handler(value: string) { return value; }\n}\n",
        "handler",
        "  // @trace target=test\n  @get()",
    ),
    (
        "handler.tsx",
        "export const Handler = () => <div />;\n",
        "Handler",
        "// @trace target=test\nexport const Handler",
    ),
    (
        "handler.go",
        "package sample\nfunc handler(value string) string { return value }\n",
        "handler",
        "// @trace target=test\nfunc handler",
    ),
    (
        "handler.rs",
        '#[get("/x")]\npub async fn handler(value: String) -> String { value }\n',
        "handler",
        '// @trace target=test\n#[get("/x")]',
    ),
    (
        "Handler.java",
        "class C {\n  @Override\n  public String handler(String value) { return value; }\n}\n",
        "handler",
        "  // @trace target=test\n  @Override",
    ),
    (
        "handler.kt",
        "fun handler(value: String): String { return value }\n",
        "handler",
        "// @trace target=test\nfun handler",
    ),
    (
        "handler.c",
        "const char *handler(const char *value) { return value; }\n",
        "handler",
        "// @trace target=test\nconst char *handler",
    ),
    (
        "handler.cpp",
        "template <typename T>\nT handler(T value) { return value; }\n",
        "handler",
        "// @trace target=test\ntemplate <typename T>",
    ),
    (
        "Handler.cs",
        "class C {\n  [HttpGet]\n  public string Handler(string value) { return value; }\n}\n",
        "Handler",
        "  // @trace target=test\n  [HttpGet]",
    ),
    (
        "handler.rb",
        "def handler(value)\n  value\nend\n",
        "handler",
        "# @trace target=test\ndef handler",
    ),
    (
        "handler.php",
        '<?php\n#[Route("/x")]\nfunction handler($value) { return $value; }\n',
        "handler",
        '// @trace target=test\n#[Route("/x")]',
    ),
    (
        "handler.scala",
        "def handler(value: String): String = value\n",
        "handler",
        "// @trace target=test\ndef handler",
    ),
    (
        "handler.swift",
        "func handler(_ value: String) -> String { value }\n",
        "handler",
        "// @trace target=test\nfunc handler",
    ),
    (
        "handler.lua",
        "local function handler(value) return value end\n",
        "handler",
        "-- @trace target=test\nlocal function handler",
    ),
    (
        "handler.ex",
        "defmodule Sample do\n  def handler(value) do\n    value\n  end\nend\n",
        "handler",
        "  # @trace target=test\n  def handler",
    ),
    (
        "handler.hs",
        "handler value = value\n",
        "handler",
        "-- @trace target=test\nhandler value",
    ),
    (
        "handler.sh",
        "handler() { printf '%s' \"$1\"; }\n",
        "handler",
        "# @trace target=test\nhandler()",
    ),
)


@pytest.mark.parametrize(("path", "source", "symbol", "expected"), LANGUAGE_CASES)
def test_all_nineteen_languages_insert_one_syntax_safe_line(
    tmp_path: Path,
    path: str,
    source: str,
    symbol: str,
    expected: str,
) -> None:
    async def scenario() -> None:
        writer = MemoryWriter({path: source})
        tools, _ = await make_tools(tmp_path, writer)
        result = await tools["annotate_trace"](path, symbol, target="test")
        updated = await writer.read_text(path)
        assert expected in updated
        assert updated.replace(expected.split("\n", 1)[0] + "\n", "", 1) == source
        assert result == {
            "path": path,
            "symbol": symbol,
            "kind": "trace",
            "annotationLine": updated[: updated.index(expected)].count("\n") + 1,
            "definitionLine": source[: source.index(symbol)].count("\n") + 2,
            "changed": True,
        }

    asyncio.run(scenario())


def test_canonical_order_replay_and_trace_target_conflict(tmp_path: Path) -> None:
    async def scenario() -> None:
        writer = MemoryWriter({"app.py": "def handler(req):\n    return req\n"})
        tools, metrics = await make_tools(tmp_path, writer)
        first = await tools["annotate_trace"](
            "app.py",
            "handler",
            target="GET_/items/{id}",
            args="req:tainted, item:derived",
            calls="validate, repo.find",
        )
        replay = await tools["annotate_trace"](
            "app.py",
            "handler",
            target="GET_/items/{id}",
            args="req:tainted,item:derived",
            calls="validate,repo.find",
        )
        second = await tools["annotate_trace"](
            "app.py", "handler", target="POST_/items", args="req:validated"
        )
        validation = await tools["annotate_validate"]("app.py", "handler", arg="req", kind="schema")
        sink = await tools["annotate_sink"]("app.py", "handler", kind="db.query", arg="item")
        sink_replay = await tools["annotate_sink"]("app.py", "handler", kind="db.query", arg="item")
        before_conflict = await writer.read_text("app.py")
        with pytest.raises(TaintAnnotationError) as conflict:
            await tools["annotate_trace"](
                "app.py",
                "handler",
                target="GET_/items/{id}",
                args="req:clean",
            )
        assert conflict.value.code == "taint_annotation_conflict"
        assert await writer.read_text("app.py") == before_conflict
        assert [first["changed"], replay["changed"], second["changed"]] == [
            True,
            False,
            True,
        ]
        assert validation["changed"] and sink["changed"] and not sink_replay["changed"]
        assert before_conflict == (
            "# @trace target=GET_/items/{id} args=req:tainted,item:derived "
            "calls=validate,repo.find\n"
            "# @trace target=POST_/items args=req:validated\n"
            "# @validate arg=req kind=schema\n"
            "# @sink kind=db.query arg=item\n"
            "def handler(req):\n"
            "    return req\n"
        )
        snapshot = metrics.snapshot()
        rendered = str(snapshot)
        assert "GET_/items" not in rendered
        assert "repo.find" not in rendered
        assert "app.py" not in rendered

    asyncio.run(scenario())


def test_structural_resolution_rejects_calls_noncallables_and_ambiguity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = (
            "handler = 1\n"
            "handler()\n"
            "def handler(value):\n"
            "    return value\n"
            "\n"
            "def handler(value):\n"
            "    return value + 1\n"
        )
        writer = MemoryWriter({"duplicate.py": source, "call.py": "missing()\n"})
        tools, _ = await make_tools(tmp_path, writer)
        with pytest.raises(TaintAnnotationError) as ambiguous:
            await tools["annotate_trace"]("duplicate.py", "handler")
        assert ambiguous.value.code == "taint_annotation_target_ambiguous"

        selected = await tools["annotate_trace"]("duplicate.py", "handler", definition_line=6)
        assert selected["annotationLine"] == 6
        assert "\n# @trace target=unknown\ndef handler(value):\n    return value + 1" in (
            await writer.read_text("duplicate.py")
        )

        for path, symbol in (("call.py", "missing"), ("duplicate.py", "Handler")):
            with pytest.raises(TaintAnnotationError) as missing:
                await tools["annotate_sink"](path, symbol, kind="db.query")
            assert missing.value.code == "taint_annotation_target_not_found"

    asyncio.run(scenario())


def test_named_php_javascript_and_lua_closures_are_real_targets(tmp_path: Path) -> None:
    async def scenario() -> None:
        writer = MemoryWriter(
            {
                "a.js": "const value = 1, handler = (req) => req;\n",
                "a.php": "<?php\n$handler = fn($req) => $req;\n",
                "a.lua": "local handler = function(req) return req end\n",
            }
        )
        tools, _ = await make_tools(tmp_path, writer)
        for path in writer.files:
            result = await tools["annotate_trace"](path, "handler")
            assert result["changed"] is True
            assert "@trace target=unknown" in await writer.read_text(path)

        with pytest.raises(TaintAnnotationError) as variable:
            await tools["annotate_trace"]("a.js", "value")
        assert variable.value.code == "taint_annotation_target_not_found"

    asyncio.run(scenario())


@pytest.mark.parametrize("storage", ["local", "memory"])
@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_real_workspace_providers_have_equal_annotation_behavior(
    tmp_path: Path,
    storage: str,
    mode: str,
) -> None:
    async def scenario() -> str:
        spec, reader = workspace_inputs(
            [("source", "", archive({"src/app.py": b"def handler(req):\n    return req\n"}))]
        )
        spec.mode = mode  # type: ignore[assignment]
        provider = (
            LocalWorkspaceProvider(settings("local", tmp_path / f"local-{mode}"))
            if storage == "local"
            else MemoryWorkspaceProvider(settings("memory"))
        )
        session = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id=f"{storage}-{mode}",
            timeout_seconds=5,
        )
        tools, _ = await make_tools(tmp_path, session.writer_view())
        await tools["annotate_trace"]("src/app.py", "handler", args="req:tainted")
        result = await session.read_text("src/app.py")
        await tools["annotate_trace"].close()
        await session.close()
        await provider.cleanup(session.storage)
        return result

    assert asyncio.run(scenario()) == (
        "# @trace target=unknown args=req:tainted\ndef handler(req):\n    return req\n"
    )


def test_compare_inside_atomic_update_preserves_a_competing_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = "def handler(req):\n    return req\n"
        writer = RacingWriter({"app.py": source})
        tools, _ = await make_tools(tmp_path, writer)
        with pytest.raises(TaintAnnotationError) as changed:
            await tools["annotate_trace"]("app.py", "handler")
        assert changed.value.code == "taint_annotation_workspace_changed"
        assert changed.value.retryable
        assert await writer.read_text("app.py") == source + "# competing-write\n"

    asyncio.run(scenario())


def test_crlf_and_input_failures_are_bounded_and_non_mutating(tmp_path: Path) -> None:
    async def scenario() -> None:
        crlf = "class C:\r\n    def handler(self, req):\r\n        return req\r\n"
        writer = MemoryWriter(
            {
                "app.py": crlf,
                "notes.txt": "def handler():\n    pass\n",
                "large.py": "x" * (MAX_SOURCE_FILE_BYTES + 1),
            }
        )
        tools, _ = await make_tools(tmp_path, writer)
        await tools["annotate_validate"]("app.py", "handler", "req", "schema")
        assert await writer.read_text("app.py") == (
            "class C:\r\n"
            "    # @validate arg=req kind=schema\r\n"
            "    def handler(self, req):\r\n"
            "        return req\r\n"
        )

        invalid_calls = (
            lambda: tools["annotate_trace"]("../app.py", "handler"),
            lambda: tools["annotate_trace"]("app.py", "handler", target=""),
            lambda: tools["annotate_trace"]("app.py", "handler", target="bad\nline"),
            lambda: tools["annotate_trace"]("app.py", "handler", args="req:unknown"),
            lambda: tools["annotate_trace"]("app.py", "handler", calls="same,same"),
            lambda: tools["annotate_trace"]("app.py", "handler", definition_line=True),
        )
        stable = await writer.read_text("app.py")
        for invoke in invalid_calls:
            with pytest.raises(TaintAnnotationError) as invalid:
                await invoke()
            assert invalid.value.code == "taint_annotation_input_invalid"
            assert await writer.read_text("app.py") == stable

        with pytest.raises(TaintAnnotationError) as unsupported:
            await tools["annotate_trace"]("notes.txt", "handler")
        assert unsupported.value.code == "taint_annotation_language_unsupported"
        with pytest.raises(TaintAnnotationError) as capacity:
            await tools["annotate_trace"]("large.py", "handler")
        assert capacity.value.code == "taint_annotation_capacity_exceeded"

    asyncio.run(scenario())


def test_factory_requires_a_writer_and_successful_positive_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        common = {
            "selected": ("annotate_trace",),
            "allocation_id": "annotation-allocation",
            "run_id": "annotation-run",
            "namespace": "analysis",
            "runtime_settings": RuntimeSettings(
                llmGatewayUrl="https://llm.example/v1",
                llmGatewayToken="secret-token",
                artifactApiUrl="https://server.example/private/v1",
                requestTimeoutSeconds=30,
            ),
            "workspace": AllocationWorkspace(root=tmp_path, path=tmp_path),
            "state": SimpleNamespace(metrics=MetricsState()),
        }
        with pytest.raises(TaintAnnotationError) as missing:
            await TaintAnnotationsToolsetFactory().create_selected(**common)
        assert missing.value.code == "workspace_required"

        factory = TaintAnnotationsToolsetFactory()
        monkeypatch.setattr(
            "contractor_runtime.toolsets.taint_annotations.language_support.probe_all_parsers",
            lambda: False,
        )
        assert await factory.probe() == frozenset()
        with pytest.raises(ValueError, match="unavailable selected taint annotation tools"):
            await factory.create_selected(
                project_workspace=MemoryWriter({"app.py": "def handler(): pass\n"}),
                **common,
            )

    asyncio.run(scenario())


def test_all_annotation_tools_generate_adk_function_declarations(tmp_path: Path) -> None:
    async def scenario() -> None:
        tools, _ = await make_tools(
            tmp_path,
            MemoryWriter({"app.py": "def handler(value):\n    return value\n"}),
        )
        declarations = {
            name: FunctionTool(tool)._get_declaration().model_dump(mode="json")
            for name, tool in tools.items()
        }
        assert sorted(declarations) == sorted(EXPORTED_TOOLS)
        assert declarations["annotate_trace"]["parameters_json_schema"] == {
            "properties": {
                "path": {"title": "Path", "type": "string"},
                "symbol": {"title": "Symbol", "type": "string"},
                "target": {"default": "unknown", "title": "Target", "type": "string"},
                "args": {"default": "", "title": "Args", "type": "string"},
                "calls": {"default": "", "title": "Calls", "type": "string"},
                "definition_line": {
                    "default": 0,
                    "title": "Definition Line",
                    "type": "integer",
                },
            },
            "required": ["path", "symbol"],
            "title": "annotate_traceParams",
            "type": "object",
        }
        assert declarations["annotate_validate"]["name"] == "annotate_validate"
        assert declarations["annotate_sink"]["name"] == "annotate_sink"

    asyncio.run(scenario())


async def make_tools(
    tmp_path: Path,
    writer: Any,
    selected: tuple[str, ...] = tuple(sorted(EXPORTED_TOOLS)),
) -> tuple[dict[str, Any], MetricsState]:
    metrics = MetricsState()
    result = await TaintAnnotationsToolsetFactory().create_selected(
        selected=selected,
        allocation_id="annotation-allocation",
        run_id="annotation-run",
        namespace="analysis",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken="secret-token",
            artifactApiUrl="https://server.example/private/v1",
            requestTimeoutSeconds=30,
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        state=SimpleNamespace(metrics=metrics),
        project_workspace=writer,
    )
    return dict(result), metrics


class MemoryWriter:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = dict(files)
        self.lock = asyncio.Lock()

    async def read_text(self, path: str) -> str:
        async with self.lock:
            try:
                return self.files[path]
            except KeyError:
                from contractor_runtime.projectfs.storage import WorkspaceStorageError

                raise WorkspaceStorageError("workspace_not_found") from None

    async def update_text(self, path: str, transform: Any) -> None:
        async with self.lock:
            self.files[path] = transform(self.files[path])


class RacingWriter(MemoryWriter):
    async def update_text(self, path: str, transform: Any) -> None:
        async with self.lock:
            self.files[path] += "# competing-write\n"
            self.files[path] = transform(self.files[path])
