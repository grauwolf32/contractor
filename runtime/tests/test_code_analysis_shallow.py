from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import jcs
import pytest
from test_projectfs_zip import archive, workspace_inputs
from test_projectfs_zip import settings as workspace_settings

import contractor_runtime.toolsets.code_analysis.languages as code_analysis_languages
import contractor_runtime.toolsets.code_analysis.tools as code_analysis
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    ManagedWorkspaceTree,
    MemoryWorkspaceProvider,
    WorkspaceSnapshot,
    WorkspaceTextFile,
    hydrate_workspace,
)
from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.toolsets.code_analysis.languages import (
    EXTENSION_LANGUAGES,
    Language,
    load_parser,
    parse_symbols,
)
from contractor_runtime.toolsets.code_analysis.tools import (
    MAX_PREVIEW_BYTES,
    MAX_PREVIEW_LINES,
    MAX_RESULT_BYTES,
    CodeAnalysisError,
    CodeAnalysisToolsetFactory,
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


def test_list_symbols_description_uses_valid_exact_node_types() -> None:
    description = code_analysis.ListSymbolsTool.description
    languages = {
        "Python/C/C++": (Language.PYTHON, Language.C, Language.CPP),
        "Go/JavaScript/TypeScript": (Language.GO, Language.JAVASCRIPT, Language.TYPESCRIPT),
        "Java": (Language.JAVA,),
        "Rust": (Language.RUST,),
    }
    examples = re.findall(r'"([a-z_]+)" \(([^)]+)\)', description)
    assert {label for _, label in examples} == set(languages)
    for node_type, label in examples:
        for language in languages[label]:
            assert node_type in {
                spec.node_type for spec in code_analysis_languages.NODE_SPECS[language]
            }
    assert "exact, language-specific parser node type" in description.lower()
    assert "row's nodeType is a valid filter value" in description
    assert '"function" or "class"' not in description


@pytest.mark.parametrize("language", list(Language))
def test_every_fixed_language_parser_extracts_a_structural_definition(
    language: Language,
) -> None:
    path, text, expected = LANGUAGE_SAMPLES[language]
    parsed = parse_symbols(load_parser(language), text.encode(), path, language, 100)
    assert expected in {item.name for item in parsed.symbols}
    assert not parsed.parse_error


@pytest.mark.parametrize("language", [Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX])
def test_arrow_callbacks_are_not_named_after_parameters_or_bodies(language: Language) -> None:
    source = (
        b"export const double = x => x * 2;\n"
        b"const ids = items.map(item => item.id).filter(id => id > 0);\n"
        b"const fallback = () => value;\n"
        b"const cb = function(x) { return x; };\n"
        b"const named = function helper(y) { return y; };\n"
    )
    parsed = parse_symbols(load_parser(language), source, "src/app.js", language, 100)
    assert not parsed.parse_error
    names = {symbol.name for symbol in parsed.symbols}
    assert {"double", "ids", "fallback", "cb", "named", "helper"} <= names
    assert not {"x", "y", "item", "id", "value"} & names
    assert not any(symbol.node_type == "arrow_function" for symbol in parsed.symbols)


@pytest.mark.parametrize(
    ("language", "source", "expected"),
    [
        (
            Language.C,
            'static int counter = 0;\nstatic const char *greeting = "hello";\n'
            "int counts[3] = {1, 2, 3};\n",
            {"counter", "greeting", "counts"},
        ),
        (
            Language.CPP,
            'std::string s = "x";\nauto lam = [](int q){ return q; };\nint &r = counter;\n',
            {"s", "lam", "r"},
        ),
    ],
)
def test_initialized_c_family_declarations_use_identifier_names(
    language: Language, source: str, expected: set[str]
) -> None:
    parsed = parse_symbols(load_parser(language), source.encode(), "src/sample.c", language, 100)
    assert not parsed.parse_error
    names = {symbol.name for symbol in parsed.symbols}
    assert expected <= names
    assert all("=" not in name for name in names)


CPP_REFERENCE_DECLARATIONS = (
    ("const std::string& Handler::name() const { return name_; }\n", "Handler::name", "name"),
    ("int &counter() { static int c; return c; }\n", "counter", "counter"),
    ("template <typename T>\nT&& take(T&& value) { return value; }\n", "take", "take"),
    ("Foo& Foo::operator=(const Foo& other) { return *this; }\n", "Foo::operator=", "operator="),
    (
        "struct Foo {\n  Foo& operator=(Foo&& other) { return *this; }\n};\n",
        "operator=",
        "operator=",
    ),
    (
        "struct Widget {\n  inline const T& label() const { return label_; }\n};\n",
        "label",
        "label",
    ),
)


@pytest.mark.parametrize(
    ("source", "expected", "bare"),
    CPP_REFERENCE_DECLARATIONS,
    ids=["qualified", "lvalue", "rvalue", "operator", "member-operator", "inline-method"],
)
def test_cpp_functions_returning_references_use_declarator_names(
    source: str, expected: str, bare: str
) -> None:
    parsed = parse_symbols(
        load_parser(Language.CPP), source.encode(), "src/a.cpp", Language.CPP, 100
    )
    assert not parsed.parse_error
    functions = [item for item in parsed.symbols if item.node_type == "function_definition"]
    assert [item.name for item in functions] == [expected]
    assert functions[0].line == source[: source.index(f"{bare}(")].count("\n") + 1


@pytest.mark.parametrize(
    ("language", "source", "expected"),
    [
        (
            Language.CPP,
            "struct Box {\n  bool operator()(int x) const { return x; }\n"
            "  int& operator[](int i) { return v[i]; }\n"
            "  bool operator<(const Box& o) const { return false; }\n};\n"
            "bool Box::operator()(int x) const { return x; }\n",
            ["operator()", "operator[]", "operator<", "Box::operator()"],
        ),
        (
            Language.CPP,
            "template <typename T> T& Box<T>::get() { return value; }\n"
            "[[nodiscard]] int& Box<int>::cached() { return value; }\n",
            ["Box::get", "Box::cached"],
        ),
        (
            Language.CPP,
            "int& (*ref_fn_ptr)(int);\nconst char* const& name_ref() { return n; }\n"
            "typedef int& (*ref_callback)(int);\nint (&arr_ref())[3] { return values; }\n",
            ["ref_fn_ptr", "name_ref", "ref_callback", "arr_ref"],
        ),
        (
            Language.C,
            "int (*handler_ptr)(int);\nint (*handler_table[4])(int);\n"
            "typedef int (*callback)(int);\n",
            ["handler_ptr", "handler_table", "callback"],
        ),
    ],
    ids=["operators", "template-scopes", "cpp-wrapped", "c-function-pointers"],
)
def test_c_family_declarators_name_operators_qualified_and_pointer_forms(
    language: Language, source: str, expected: list[str]
) -> None:
    parsed = parse_symbols(load_parser(language), source.encode(), "src/a.c", language, 100)
    assert not parsed.parse_error
    declared = {"declaration", "function_definition", "type_definition"}
    assert [item.name for item in parsed.symbols if item.node_type in declared] == expected


def test_search_def_and_list_symbols_find_cpp_reference_returning_definitions(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        files = {
            f"src/form{index}.cpp": source
            for index, (source, _, _) in enumerate(CPP_REFERENCE_DECLARATIONS)
        }
        tools, _ = await _tools(MutableReader(files), tmp_path)
        listed = await tools["list_symbols"](node_type="function_definition", limit=200)
        assert sorted(item["name"] for item in listed["items"]) == sorted(
            expected for _, expected, _ in CPP_REFERENCE_DECLARATIONS
        )
        for index, (_, expected, bare) in enumerate(CPP_REFERENCE_DECLARATIONS):
            found = await tools["search_def"](bare, language="cpp")
            assert (f"src/form{index}.cpp", expected) in {
                (item["path"], item["name"]) for item in found["items"]
            }
        for return_type in ("T", "Foo", "string"):
            found = await tools["search_def"](return_type)
            assert all(item["nodeType"] != "function_definition" for item in found["items"])

    asyncio.run(scenario())


def test_cpp_declarator_names_search_identically_with_and_without_cache(tmp_path: Path) -> None:
    source = (
        "template <typename T> T& Box<T>::get() { return value; }\n"
        "struct Box { bool operator ()(int x) const { return x; } };\n"
    )

    async def scenario() -> None:
        tools, _ = await _tools(MutableReader({"src/box.cpp": source}), tmp_path)
        queries = {"get": "Box::get", "Box::get": "Box::get", "operator ()": "operator ()"}
        uncached = {query: await tools["search_def"](query) for query in queries}
        await tools["list_symbols"]()
        for query, expected in queries.items():
            assert [item["name"] for item in uncached[query]["items"]] == [expected]
            assert await tools["search_def"](query) == uncached[query]

    asyncio.run(scenario())


def test_search_def_uses_arrow_bindings_and_initialized_c_names(tmp_path: Path) -> None:
    async def scenario() -> None:
        tools, _ = await _tools(
            MutableReader(
                {
                    "src/app.js": (
                        "export const double = x => x * 2;\n"
                        "const ids = items.map(item => item.id).filter(id => id > 0);\n"
                        "const fallback = () => value;\n"
                    ),
                    "src/state.c": (
                        'static int counter = 0;\nstatic const char *greeting = "hello";\n'
                    ),
                }
            ),
            tmp_path,
        )
        assert (await tools["search_def"]("double"))["items"]
        assert (await tools["search_def"]("counter"))["items"]
        for callback_name in ("x", "item", "id", "value"):
            assert not (await tools["search_def"](callback_name))["items"]
        listed = await tools["list_symbols"](limit=100)
        assert all("=" not in item["name"] for item in listed["items"])

    asyncio.run(scenario())


def test_c_family_headers_keep_the_snapshot_language(tmp_path: Path) -> None:
    async def scenario() -> None:
        cpp_header = (
            "namespace app {\nclass Handler {\npublic:\n  int count() const { return 1; }\n};\n}\n"
        )
        cpp_tools, _ = await _tools(
            MutableReader(
                {
                    "include/handler.h": cpp_header,
                    "src/main.cpp": "int main() { return app::Handler{}.count(); }\n",
                }
            ),
            tmp_path / "cpp",
        )
        listed = await cpp_tools["list_symbols"](limit=100)
        assert listed["coverage"]["parseErrors"] == 0
        assert any(
            item["name"] == "count"
            and item["path"] == "include/handler.h"
            and item["language"] == "cpp"
            for item in listed["items"]
        )
        found = await cpp_tools["search_def"]("count", language="cpp")
        assert [item["path"] for item in found["items"]] == ["include/handler.h"]

        c_tools, _ = await _tools(
            MutableReader(
                {
                    "include/util.h": "static inline int helper(int x) { return x + 1; }\n",
                    "src/main.c": "int main(void) { return helper(2); }\n",
                }
            ),
            tmp_path / "c",
        )
        c_listed = await c_tools["list_symbols"](limit=100)
        assert c_listed["coverage"]["parseErrors"] == 0
        assert any(
            item["name"] == "helper"
            and item["path"] == "include/util.h"
            and item["language"] == "c"
            for item in c_listed["items"]
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        ((), Language.C),
        (("include/a.h", "README.md"), Language.C),
        (("src/main.c", "include/a.h"), Language.C),
        (("src/main.cpp", "include/a.h"), Language.CPP),
        (("src/a.c", "src/b.c", "src/c.c", "vendor/lib.cc"), Language.C),
        (("src/a.c", "src/b.cc"), Language.CPP),
        (("src/a.c", "include/b.hpp", "include/c.hh"), Language.CPP),
        (("SRC/MAIN.CPP", "src/x.c"), Language.CPP),
    ],
)
def test_header_language_follows_the_snapshot_c_family_majority(
    paths: tuple[str, ...], expected: Language
) -> None:
    assert code_analysis_languages.header_language(paths) is expected
    assert code_analysis_languages.detect_language("a/b.H", header=expected) is expected


def test_c_project_with_a_vendored_cpp_file_keeps_c_headers(tmp_path: Path) -> None:
    async def scenario() -> None:
        tools, _ = await _tools(
            MutableReader(
                {
                    # typeof is valid C but not C++ syntax.
                    "include/util.h": (
                        "static inline int twice(int value) {\n"
                        "  typeof(value) doubled = value * 2;\n  return doubled;\n}\n"
                    ),
                    "src/a.c": "int main(void) { return twice(2); }\n",
                    "src/b.c": "int other(void) { return 1; }\n",
                    "third_party/lib.cc": "int lib() { return 3; }\n",
                }
            ),
            tmp_path,
        )
        listed = await tools["list_symbols"](path="include", node_type="function_definition")
        assert listed["coverage"]["parseErrors"] == 0
        assert [(item["name"], item["language"]) for item in listed["items"]] == [("twice", "c")]

    asyncio.run(scenario())


def test_case_variant_extensions_are_parsed_or_reported(tmp_path: Path) -> None:
    async def scenario() -> None:
        tools, _ = await _tools(
            MutableReader(
                {
                    "Tool.PY": "def upper_tool():\n    return 1\n",
                    "native/impl.C": "int upper_c(void) { return 3; }\n",
                    "native/impl.H": "int upper_h(void);\n",
                    "contracts/Token.SOL": "contract Token {}\n",
                }
            ),
            tmp_path,
        )
        listed = await tools["list_symbols"]()
        assert {(item["path"], item["name"], item["language"]) for item in listed["items"]} == {
            ("Tool.PY", "upper_tool", "python"),
            ("native/impl.C", "upper_c", "c"),
            ("native/impl.H", "upper_h", "c"),
        }
        assert listed["coverage"]["unsupportedSourceFiles"] == 1

    asyncio.run(scenario())


def test_php_arrow_functions_are_not_structural_definitions() -> None:
    assert all(
        spec.node_type != "arrow_function"
        for specs in code_analysis_languages.NODE_SPECS.values()
        for spec in specs
    )
    parsed = parse_symbols(
        load_parser(Language.PHP),
        b"<?php\n$double = fn($x) => $x * 2;\nfunction named($y) { return $y; }\n",
        "src/app.php",
        Language.PHP,
        100,
    )
    assert not parsed.parse_error
    assert [item.name for item in parsed.symbols] == ["named"]


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


def test_search_def_builds_previews_only_for_the_returned_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        reader = MutableReader(
            {
                "a.py": "def Target():\n    pass\n" * 10,
                "b.py": "def Target():\n    pass\n" * 10,
            }
        )
        previews: list[int] = []
        original = code_analysis._preview

        def counting(source: bytes, start_byte: int, end_byte: int) -> str:
            previews.append(start_byte)
            return original(source, start_byte, end_byte)

        monkeypatch.setattr(code_analysis, "_preview", counting)
        tools, _ = await _tools(reader, tmp_path)
        first = await tools["search_def"]("Target", limit=3)
        assert len(first["items"]) == len(previews) == 3
        assert first["observedTotal"] == 20 and first["truncated"]
        assert all(item["preview"].startswith("def Target") for item in first["items"])
        second = await tools["search_def"]("Target", cursor=first["nextCursor"], limit=3)
        assert [(item["path"], item["line"]) for item in second["items"]] == [
            ("a.py", 7),
            ("a.py", 9),
            ("a.py", 11),
        ]
        assert len(previews) == 6

    asyncio.run(scenario())


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

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        final_index = alphabet.index(first["nextCursor"][-1])
        assert final_index % 4 == 0
        tampered = first["nextCursor"][:-1] + alphabet[final_index + 1]
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


def test_search_def_prefilters_only_uncached_or_flagged_file_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanned: list[int] = []
    original = code_analysis._contains_casefold

    def counting(text: str, needle: str) -> bool:
        scanned.append(len(text))
        return original(text, needle)

    monkeypatch.setattr(code_analysis, "_contains_casefold", counting)

    async def scenario() -> None:
        reader = MutableReader(
            {
                "a.py": "def target():\n    pass\n",
                "b.py": "def caller():\n    return target()\n",
                "broken.py": "def broken(\n",
                "broken_target.py": "def target(\n",
                "c.py": "def unrelated():\n    pass\n",
            }
        )
        tools, _ = await _tools(reader, tmp_path)
        cold = await tools["search_def"]("target")
        assert len(scanned) == 5

        scanned.clear()
        warm = await tools["search_def"]("target")
        # a.py and b.py were parsed and cached; the flagged parse-error files
        # and never-parsed files still need their text.
        assert len(scanned) == 3

        await tools["list_symbols"]()
        scanned.clear()
        cached = await tools["search_def"]("target")
        assert len(scanned) == 2  # only the two cached parse-error files
        assert cold == warm == cached
        assert cold["coverage"]["parseErrors"] == 1
        assert [item["path"] for item in cold["items"]] == ["a.py"]

    asyncio.run(scenario())


def test_search_def_counts_only_matching_definitions_toward_symbol_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(code_analysis, "MAX_COMPACT_SYMBOLS", 3)

    async def scenario() -> None:
        reader = MutableReader(
            {
                "a.py": "def target():\n    pass\n",
                "b.py": "# target\ndef x1(): pass\ndef x2(): pass\ndef x3(): pass\n",
                "c.py": "def target():\n    pass\n",
            }
        )
        fresh_tools, _ = await _tools(reader, tmp_path / "fresh")
        fresh = await fresh_tools["search_def"]("target")
        assert [item["path"] for item in fresh["items"]] == ["a.py", "c.py"]
        assert fresh["coverage"]["reasons"] == []

        warmed_tools, _ = await _tools(reader, tmp_path / "warmed")
        listed = await warmed_tools["list_symbols"]()
        assert listed["coverage"]["reasons"] == ["symbol_limit"]
        assert await warmed_tools["search_def"]("target") == fresh

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


def test_overlong_symbol_names_are_skipped_without_stopping_the_scan(tmp_path: Path) -> None:
    async def scenario() -> None:
        long_name = "x" * 300
        reader = MutableReader(
            {
                "a.py": f"def {long_name}(): pass\ndef kept(): pass\n",
                "b.py": "def later(): pass\n",
            }
        )
        tools, _ = await _tools(reader, tmp_path)
        for _ in range(2):
            listed = await tools["list_symbols"]()
            assert sorted(item["name"] for item in listed["items"]) == ["kept", "later"]
            assert listed["coverage"]["reasons"] == ["symbol_name_limit"]

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


def test_large_search_page_fits_off_loop_and_encodes_each_row_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Structural checks instead of an event-loop gap bound: a worker-thread
    # garbage collection or Tree-sitter parse holds the GIL however page
    # fitting is scheduled, so wall-clock gaps are not deterministic.
    body = "\n".join("    value = '" + ("x" * 350) + "'" for _ in range(12))
    source = "\n".join(f"def target():\n{body}" for _ in range(200)) + "\n"

    async def scenario() -> None:
        tools, _ = await _tools(MutableReader({"many.py": source}), tmp_path)
        loop_thread = threading.get_ident()
        original_canonicalize = jcs.canonicalize
        original_fit = code_analysis._fit_result_page
        row_encodings: Counter[int] = Counter()
        encoding_threads: set[int] = set()
        fit_threads: list[int] = []

        def count_row_encodings(value: Any) -> bytes:
            # Encoding an envelope encodes each of its items as well.
            rows = value.get("items", ()) if isinstance(value, dict) else ()
            if isinstance(value, dict) and "nodeType" in value:
                rows = (value,)
            for row in rows:
                row_encodings[row["line"]] += 1
                encoding_threads.add(threading.get_ident())
            return original_canonicalize(value)

        def record_fit_thread(rows: list[Any], build_result: Any) -> dict[str, Any]:
            fit_threads.append(threading.get_ident())
            return original_fit(rows, build_result)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(jcs, "canonicalize", count_row_encodings)
                patch.setattr(code_analysis, "_fit_result_page", record_fit_thread)
                result = await tools["search_def"]("target", limit=200)
        finally:
            await tools["search_def"].close()
        assert result["observedTotal"] == 200
        assert 0 < len(result["items"]) < 200
        assert result["truncated"] and result["nextCursor"]
        assert len(original_canonicalize(result)) <= MAX_RESULT_BYTES
        assert len(fit_threads) == 1 and loop_thread not in fit_threads
        assert encoding_threads and loop_thread not in encoding_threads
        assert sorted(row_encodings) == [1 + 13 * index for index in range(200)]
        assert set(row_encodings.values()) == {1}

    asyncio.run(scenario())


def test_result_page_fitting_rejects_item_dependent_envelopes() -> None:
    rows = [{"number": 1}, {"number": 2}]
    for build in (
        lambda _count, items: {"items": list(items)},
        lambda _count, items: {"items": items, "first": items[:1]},
    ):
        with pytest.raises(CodeAnalysisError, match="code_analysis_engine_failed"):
            code_analysis._fit_result_page(rows, build)


@pytest.mark.parametrize("shape", ["shallow", "graph", "path"])
def test_result_page_fitting_matches_drop_one_row_reference(
    monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    rows = [{"preview": "x" * 300, "number": number} for number in range(20)]

    def build(count: int, items: list[Any]) -> dict[str, Any]:
        if shape == "path":
            return {"items": items, "coverage": {"analyzedFiles": 1}, "truncated": count < 20}
        return {
            "items": items,
            "nextCursor": f"cursor-{count}" if count < 20 else None,
            "truncated": count < 20,
            "observedTotal": 20,
            "coverage": {"analyzedFiles": 1} if shape == "graph" else {"reasons": []},
        }

    with monkeypatch.context() as patch:
        patch.setattr(code_analysis, "MAX_RESULT_BYTES", 1400)
        expected_count = len(rows)
        while len(jcs.canonicalize(build(expected_count, rows[:expected_count]))) > 1400:
            expected_count -= 1
        expected = build(expected_count, rows[:expected_count])
        assert code_analysis._fit_result_page(rows, build) == expected
        assert len(jcs.canonicalize(expected)) <= 1400


def test_result_page_fitting_rejects_oversized_empty_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(code_analysis, "MAX_RESULT_BYTES", 2)
    with pytest.raises(CodeAnalysisError, match="code_analysis_capacity_exceeded"):
        code_analysis._fit_result_page([], lambda _count, items: {"items": items})


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


def test_real_four_mib_parser_keeps_event_loop_gaps_below_100_ms() -> None:
    # Many syntax nodes exercise the native parser rather than a sleeping fake.
    unit = b'def handler():\n    return "a moderately long value"\n'
    source = unit * ((4 << 20) // len(unit))
    source += b"#" * ((4 << 20) - len(source))

    async def scenario() -> None:
        ticks: list[float] = []
        running = True

        async def heartbeat() -> None:
            while running:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.001)

        task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.01)
        result = await asyncio.to_thread(
            code_analysis_languages.parse_symbols,
            code_analysis_languages.load_parser(code_analysis_languages.Language.PYTHON),
            source,
            "large.py",
            code_analysis_languages.Language.PYTHON,
            1,
        )
        await asyncio.sleep(0.01)
        running = False
        await task
        gap = max(b - a for a, b in pairwise(ticks))
        assert gap < 0.1, f"native parsing stalled the event loop for {gap:.3f}s"
        assert result.symbol_limit_reached and not result.parse_error
        assert len(result.symbols) == 1 and result.symbols[0].name == "handler"

    asyncio.run(scenario())
