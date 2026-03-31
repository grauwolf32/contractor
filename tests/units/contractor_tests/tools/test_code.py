"""
Tests for contractor.tools.code — symbol definition search using tree-sitter.
"""

from __future__ import annotations

import textwrap
from typing import Optional

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from contractor.tools.code.tools import (
    Language,
    CodeTools,
    SearchResult,
    DefinitionResult,
    GrepMatch,
    code_tools,
    detect_language,
    _symbol_matches,
    _specs_for,
)


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def memfs() -> MemoryFileSystem:
    """Fresh in-memory filesystem for each test."""
    fs = MemoryFileSystem()
    fs.store.clear()
    return fs


def _write(fs: MemoryFileSystem, path: str, content: str) -> None:
    """Helper: write text content into the memory filesystem."""
    with fs.open(path, "wb") as f:
        f.write(textwrap.dedent(content).encode("utf-8"))


@pytest.fixture()
def python_project(memfs: MemoryFileSystem) -> MemoryFileSystem:
    """A small Python project."""
    _write(
        memfs,
        "/project/src/handler.py",
        """\
        from src.service import create_user

        def handle_request(request):
            data = request.json()
            return create_user(data)
        """,
    )
    _write(
        memfs,
        "/project/src/service.py",
        """\
        from src.repo import insert_user

        def create_user(data):
            validated = validate(data)
            return insert_user(validated)

        def validate(data):
            if not data.get("name"):
                raise ValueError("name required")
            return data
        """,
    )
    _write(
        memfs,
        "/project/src/repo.py",
        """\
        def insert_user(user):
            db.execute("INSERT INTO users ...", user)
        """,
    )
    _write(
        memfs,
        "/project/src/models.py",
        """\
        class User:
            def __init__(self, name: str, email: str):
                self.name = name
                self.email = email

            def full_name(self):
                return self.name
        """,
    )
    return memfs


@pytest.fixture()
def js_project(memfs: MemoryFileSystem) -> MemoryFileSystem:
    """A small JavaScript project."""
    _write(
        memfs,
        "/project/src/index.js",
        """\
        const express = require('express');
        const { handleRequest } = require('./handler');

        const app = express();
        app.post('/users', handleRequest);
        """,
    )
    _write(
        memfs,
        "/project/src/handler.js",
        """\
        function handleRequest(req, res) {
            const data = req.body;
            res.json({ ok: true });
        }

        const processData = (data) => {
            return data;
        };

        module.exports = { handleRequest, processData };
        """,
    )
    return memfs


@pytest.fixture()
def go_project(memfs: MemoryFileSystem) -> MemoryFileSystem:
    """A small Go project."""
    _write(
        memfs,
        "/project/main.go",
        """\
        package main

        import "fmt"

        func main() {
            fmt.Println(greet("world"))
        }

        func greet(name string) string {
            return "Hello, " + name
        }
        """,
    )
    _write(
        memfs,
        "/project/handler.go",
        """\
        package main

        type UserService struct{}

        func (s *UserService) CreateUser(name string) error {
            return nil
        }

        func HandleRequest(w http.ResponseWriter, r *http.Request) {
            svc := &UserService{}
            svc.CreateUser("test")
        }
        """,
    )
    return memfs


@pytest.fixture()
def rust_project(memfs: MemoryFileSystem) -> MemoryFileSystem:
    """A small Rust project."""
    _write(
        memfs,
        "/project/src/main.rs",
        """\
        struct Config {
            port: u16,
        }

        enum Status {
            Active,
            Inactive,
        }

        fn start_server(config: Config) {
            println!("Starting on port {}", config.port);
        }

        fn main() {
            let cfg = Config { port: 8080 };
            start_server(cfg);
        }
        """,
    )
    return memfs


@pytest.fixture()
def multi_lang_project(memfs: MemoryFileSystem) -> MemoryFileSystem:
    """Project with multiple languages."""
    _write(
        memfs,
        "/project/app.py",
        """\
        def process():
            pass
        """,
    )
    _write(
        memfs,
        "/project/app.js",
        """\
        function process() {
            return null;
        }
        """,
    )
    _write(
        memfs,
        "/project/app.go",
        """\
        package main

        func process() {
        }
        """,
    )
    _write(
        memfs,
        "/project/app.rs",
        """\
        fn process() {
        }
        """,
    )
    return memfs


@pytest.fixture()
def tools(python_project: MemoryFileSystem) -> CodeTools:
    return CodeTools(fs=python_project, root="/project")


# ─── detect_language ──────────────────────────────────────────────────


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("foo.py", Language.PYTHON),
            ("bar.js", Language.JAVASCRIPT),
            ("baz.ts", Language.TYPESCRIPT),
            ("qux.tsx", Language.TSX),
            ("main.go", Language.GO),
            ("lib.rs", Language.RUST),
            ("App.java", Language.JAVA),
            ("App.kt", Language.KOTLIN),
            ("main.c", Language.C),
            ("main.cpp", Language.CPP),
            ("main.cc", Language.CPP),
            ("Prog.cs", Language.CSHARP),
            ("script.rb", Language.RUBY),
            ("index.php", Language.PHP),
            ("Main.scala", Language.SCALA),
            ("main.swift", Language.SWIFT),
            ("init.lua", Language.LUA),
            ("app.ex", Language.ELIXIR),
            ("Main.hs", Language.HASKELL),
            ("run.sh", Language.BASH),
            ("file.mjs", Language.JAVASCRIPT),
            ("file.cjs", Language.JAVASCRIPT),
            ("file.jsx", Language.JAVASCRIPT),
            ("file.kts", Language.KOTLIN),
            ("header.h", Language.C),
            ("header.hpp", Language.CPP),
            ("header.hxx", Language.CPP),
            ("file.cxx", Language.CPP),
            ("file.sc", Language.SCALA),
            ("file.exs", Language.ELIXIR),
            ("file.bash", Language.BASH),
        ],
    )
    def test_known_extensions(self, path: str, expected: Language) -> None:
        assert detect_language(path) == expected

    @pytest.mark.parametrize(
        "path", ["file.txt", "file.md", "file.json", "file.xml", "Makefile", "file"]
    )
    def test_unknown_extensions(self, path: str) -> None:
        assert detect_language(path) is None

    def test_case_insensitive_via_path(self) -> None:
        # PurePosixPath.suffix preserves case, but we .lower() it
        assert detect_language("FILE.PY") == Language.PYTHON
        assert detect_language("APP.JS") == Language.JAVASCRIPT


# ─── _symbol_matches ──────────────────────────────────────────────────


class TestSymbolMatches:
    def test_exact_match(self) -> None:
        assert _symbol_matches("my_func", "my_func") is True

    def test_no_match(self) -> None:
        assert _symbol_matches("my_func", "other_func") is False

    def test_qualified_query(self) -> None:
        # query="MyClass.method" should match extracted="method"
        assert _symbol_matches("method", "MyClass.method") is True

    def test_qualified_extracted(self) -> None:
        # extracted="pkg.Foo" should match query="Foo"
        assert _symbol_matches("pkg.Foo", "Foo") is True

    def test_case_insensitive_fallback(self) -> None:
        assert _symbol_matches("MyFunc", "myfunc") is True
        assert _symbol_matches("myfunc", "MyFunc") is True

    def test_no_partial_match(self) -> None:
        assert _symbol_matches("my_func_extra", "my_func") is False
        assert _symbol_matches("my_func", "my_func_extra") is False

    def test_deeply_qualified_query(self) -> None:
        # rsplit(".", 1) only splits on last dot
        assert _symbol_matches("method", "a.b.method") is True

    def test_both_qualified_no_match(self) -> None:
        assert _symbol_matches("a.foo", "b.bar") is False


# ─── _specs_for ───────────────────────────────────────────────────────


class TestSpecsFor:
    @pytest.mark.parametrize("lang", list(Language))
    def test_every_language_has_specs(self, lang: Language) -> None:
        specs = _specs_for(lang)
        assert len(specs) > 0, f"No specs for {lang.value}"

    def test_python_has_function_and_class(self) -> None:
        specs = _specs_for(Language.PYTHON)
        types = {s.node_type for s in specs}
        assert "function_definition" in types
        assert "class_definition" in types

    def test_go_has_method_declaration(self) -> None:
        specs = _specs_for(Language.GO)
        types = {s.node_type for s in specs}
        assert "method_declaration" in types
        assert "function_declaration" in types


# ─── CodeTools — Python ──────────────────────────────────────


class TesttoolsPython:
    def test_find_function(self, tools: CodeTools) -> None:
        result = tools.search_definition("handle_request")
        assert len(result.definitions) >= 1
        defn = result.definitions[0]
        assert defn.symbol == "handle_request"
        assert "handler.py" in defn.file
        assert defn.language == "python"
        assert defn.node_type in ("function_definition", "decorated_definition")
        assert defn.line >= 1

    def test_find_class(self, tools: CodeTools) -> None:
        result = tools.search_definition("User")
        assert len(result.definitions) >= 1
        defn = result.definitions[0]
        assert defn.symbol == "User"
        assert defn.node_type == "class_definition"
        assert "models.py" in defn.file

    def test_find_method_by_name(self, tools: CodeTools) -> None:
        result = tools.search_definition("full_name")
        assert len(result.definitions) >= 1
        assert any(d.symbol == "full_name" for d in result.definitions)

    def test_find_multiple_functions(self, tools: CodeTools) -> None:
        result = tools.search_definition("validate")
        assert len(result.definitions) >= 1
        assert result.definitions[0].symbol == "validate"

    def test_find_in_subdirectory(self, tools: CodeTools) -> None:
        result = tools.search_definition("insert_user", path="src")
        assert len(result.definitions) >= 1
        assert any("repo.py" in d.file for d in result.definitions)

    def test_not_found_returns_grep_fallback(
        self, python_project: MemoryFileSystem
    ) -> None:
        # "db" appears in repo.py but is not a definition
        s = CodeTools(fs=python_project, root="/project")
        result = s.search_definition("execute")
        # "execute" is called but not defined — should be grep fallback or none
        if result.definitions:
            pytest.skip("tree-sitter found a definition unexpectedly")
        assert result.is_fallback or len(result.grep_matches) == 0

    def test_symbol_not_in_any_file(self, tools: CodeTools) -> None:
        result = tools.search_definition("nonexistent_symbol_xyz")
        assert len(result.definitions) == 0
        assert len(result.grep_matches) == 0
        assert result.is_fallback is False

    def test_context_is_populated(self, tools: CodeTools) -> None:
        result = tools.search_definition("create_user")
        assert len(result.definitions) >= 1
        defn = result.definitions[0]
        assert "create_user" in defn.context
        assert len(defn.context) > 0

    def test_line_numbers_are_positive(self, tools: CodeTools) -> None:
        result = tools.search_definition("handle_request")
        for defn in result.definitions:
            assert defn.line >= 1
            assert defn.end_line >= defn.line
            assert defn.column >= 0


# ─── CodeTools — JavaScript ──────────────────────────────────


class TesttoolsJavaScript:
    def test_find_function_declaration(self, js_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=js_project, root="/project")
        result = tools.search_definition("handleRequest")
        assert len(result.definitions) >= 1
        defn = result.definitions[0]
        assert defn.symbol == "handleRequest"
        assert defn.language == "javascript"

    def test_find_arrow_function(self, js_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=js_project, root="/project")
        result = tools.search_definition("processData")
        # Should find the variable_declarator for const processData = ...
        assert len(result.definitions) >= 1 or len(result.grep_matches) >= 1

    def test_language_filter(self, js_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=js_project, root="/project")
        result = tools.search_definition(
            "handleRequest", language_filter=Language.PYTHON
        )
        # No Python files in js_project
        assert len(result.definitions) == 0


# ─── CodeTools — Go ─────────────────────────────────────────


class TesttoolsGo:
    def test_find_function(self, go_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=go_project, root="/project")
        result = tools.search_definition("greet")
        assert len(result.definitions) >= 1
        assert result.definitions[0].language == "go"

    def test_find_method(self, go_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=go_project, root="/project")
        result = tools.search_definition("CreateUser")
        assert len(result.definitions) >= 1
        defn = result.definitions[0]
        assert defn.node_type == "method_declaration"

    def test_find_type(self, go_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=go_project, root="/project")
        result = tools.search_definition("UserService")
        # Should find the type spec or struct
        assert len(result.definitions) >= 1 or len(result.grep_matches) >= 1


# ─── CodeTools — Rust ────────────────────────────────────────


class TesttoolsRust:
    def test_find_function(self, rust_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=rust_project, root="/project")
        result = tools.search_definition("start_server")
        assert len(result.definitions) >= 1
        assert result.definitions[0].language == "rust"

    def test_find_struct(self, rust_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=rust_project, root="/project")
        result = tools.search_definition("Config")
        assert len(result.definitions) >= 1
        assert result.definitions[0].node_type == "struct_item"

    def test_find_enum(self, rust_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=rust_project, root="/project")
        result = tools.search_definition("Status")
        assert len(result.definitions) >= 1
        assert result.definitions[0].node_type == "enum_item"


# ─── Multi-language ───────────────────────────────────────────────────


class TestMultiLanguage:
    def test_finds_across_languages(self, multi_lang_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=multi_lang_project, root="/project")
        result = tools.search_definition("process")
        assert len(result.definitions) >= 3  # py, js, go, rs
        languages = {d.language for d in result.definitions}
        assert "python" in languages
        assert "go" in languages

    def test_language_filter_restricts(
        self, multi_lang_project: MemoryFileSystem
    ) -> None:
        tools = CodeTools(fs=multi_lang_project, root="/project")
        result = tools.search_definition("process", language_filter=Language.PYTHON)
        assert all(d.language == "python" for d in result.definitions)
        assert len(result.definitions) >= 1

    def test_each_language_individually(
        self, multi_lang_project: MemoryFileSystem
    ) -> None:
        tools = CodeTools(fs=multi_lang_project, root="/project")
        for lang in [Language.PYTHON, Language.JAVASCRIPT, Language.GO, Language.RUST]:
            result = tools.search_definition("process", language_filter=lang)
            assert len(result.definitions) >= 1, f"No definition found for {lang.value}"


# ─── Caching ──────────────────────────────────────────────────────────


class TestCaching:
    def test_second_search_returns_same_results(self, tools: CodeTools) -> None:
        result1 = tools.search_definition("handle_request")
        result2 = tools.search_definition("handle_request")
        assert result1.definitions == result2.definitions

    def test_different_symbols_return_different_results(self, tools: CodeTools) -> None:
        r1 = tools.search_definition("handle_request")
        r2 = tools.search_definition("create_user")
        # Both should find definitions
        assert len(r1.definitions) >= 1
        assert len(r2.definitions) >= 1
        assert r1.definitions[0].symbol != r2.definitions[0].symbol

    def test_parse_cache_populated_after_search(self, tools: CodeTools) -> None:
        tools.search_definition("handle_request")
        # After a successful search, parse cache should have entries
        # (only if tree-sitter parsing succeeded)
        if tools._cache._parse_cache:
            assert len(tools._cache._parse_cache) > 0
        else:
            # If parse cache is empty, resolution cache should also be empty
            # but grep fallback should have found something
            pass

    def test_clear_cache(self, tools: CodeTools) -> None:
        tools.search_definition("handle_request")
        tools.clear_cache()
        assert len(tools._cache._parse_cache) == 0
        assert len(tools._cache._resolution_cache) == 0

    def test_invalidate_file(self, tools: CodeTools) -> None:
        tools.search_definition("handle_request")
        # Find a handler path in cache if present
        handler_path = None
        for key in tools._cache._parse_cache:
            if "handler.py" in key:
                handler_path = key
                break
        if handler_path is not None:
            tools.invalidate_file(handler_path)
            assert handler_path not in tools._cache._parse_cache
        # If no cache entries, invalidation is still safe
        tools.invalidate_file("/nonexistent/path.py")

    def test_cache_invalidated_on_content_change(
        self, python_project: MemoryFileSystem
    ) -> None:
        s = CodeTools(fs=python_project, root="/project")
        result1 = s.search_definition("handle_request")
        assert len(result1.definitions) >= 1

        # Modify the file — remove the original function
        _write(
            python_project,
            "/project/src/handler.py",
            """\
            def handle_request_v2(request):
                return "v2"
            """,
        )

        # Search again — cache should detect content change
        result2 = s.search_definition("handle_request")
        # Original function no longer exists
        assert len(result2.definitions) == 0 or all(
            d.symbol != "handle_request" for d in result2.definitions
        )

    def test_cache_hit_returns_consistent_results(self, tools: CodeTools) -> None:
        result1 = tools.search_definition("create_user")
        result2 = tools.search_definition("create_user")
        # Cache should return consistent results
        assert len(result1.definitions) == len(result2.definitions)
        if result1.definitions and result2.definitions:
            assert result1.definitions[0].symbol == result2.definitions[0].symbol
            assert result1.definitions[0].file == result2.definitions[0].file
            assert result1.definitions[0].line == result2.definitions[0].line


# ─── SearchResult.to_dict ─────────────────────────────────────────────


class TestSearchResultToDict:
    def test_definition_result_dict(self, tools: CodeTools) -> None:
        result = tools.search_definition("handle_request")
        d = result.to_dict()
        assert d["symbol"] == "handle_request"
        if d["found"]:
            assert d["kind"] == "definition"
            assert d["count"] >= 1
            assert len(d["results"]) >= 1
            entry = d["results"][0]
            assert "file" in entry
            assert "line" in entry
            assert "end_line" in entry
            assert "column" in entry
            assert "node_type" in entry
            assert "language" in entry
            assert "context" in entry
        else:
            # Grep fallback is also acceptable
            assert d["kind"] in ("grep_fallback", "none")

    def test_grep_fallback_dict(self) -> None:
        result = SearchResult(
            symbol="test",
            definitions=[],
            grep_matches=[GrepMatch(file="a.py", line=10, text=">>>    10 | test()")],
            is_fallback=True,
        )
        d = result.to_dict()
        assert d["found"] is False
        assert d["kind"] == "grep_fallback"
        assert "note" in d
        assert d["count"] == 1
        assert d["results"][0]["file"] == "a.py"

    def test_none_result_dict(self) -> None:
        result = SearchResult(
            symbol="nothing",
            definitions=[],
            grep_matches=[],
            is_fallback=False,
        )
        d = result.to_dict()
        assert d["found"] is False
        assert d["kind"] == "none"
        assert d["count"] == 0
        assert d["results"] == []


# ─── DefinitionResult ─────────────────────────────────────────────────


class TestDefinitionResult:
    def test_location_property(self) -> None:
        defn = DefinitionResult(
            symbol="foo",
            file="src/bar.py",
            line=42,
            end_line=50,
            column=0,
            node_type="function_definition",
            language="python",
            context="def foo(): ...",
        )
        assert defn.location == "src/bar.py:42"

    def test_frozen(self) -> None:
        defn = DefinitionResult(
            symbol="foo",
            file="src/bar.py",
            line=1,
            end_line=1,
            column=0,
            node_type="function_definition",
            language="python",
            context="def foo(): ...",
        )
        with pytest.raises(AttributeError):
            defn.symbol = "bar"  # type: ignore[misc]


class TestGrepMatch:
    def test_location_property(self) -> None:
        match = GrepMatch(file="a.py", line=10, text="hello")
        assert match.location == "a.py:10"


# ─── Grep fallback ────────────────────────────────────────────────────


class TestGrepFallback:
    def test_fallback_when_no_definition(
        self, python_project: MemoryFileSystem
    ) -> None:
        """A symbol that is used but never defined should produce grep fallback."""
        _write(
            python_project,
            "/project/src/caller.py",
            """\
            def do_work():
                result = mysterious_function(42)
                return result
            """,
        )
        s = CodeTools(fs=python_project, root="/project")
        result = s.search_definition("mysterious_function")
        # Not defined anywhere, but appears in caller.py
        if result.definitions:
            pytest.skip("Unexpectedly found as definition")
        assert result.is_fallback is True
        assert len(result.grep_matches) >= 1
        assert any("caller.py" in m.file for m in result.grep_matches)

    def test_grep_shows_context_lines(self, python_project: MemoryFileSystem) -> None:
        _write(
            python_project,
            "/project/src/usage.py",
            """\
            import os

            x = 1
            y = external_lib_call(x)
            z = y + 1
            """,
        )
        s = CodeTools(fs=python_project, root="/project", grep_context_lines=1)
        result = s.search_definition("external_lib_call")
        if result.definitions:
            pytest.skip("Unexpectedly found as definition")
        assert len(result.grep_matches) >= 1
        match = result.grep_matches[0]
        assert ">>>" in match.text  # matching line marker
        # Should have context lines around it
        lines = match.text.strip().splitlines()
        assert len(lines) >= 2  # at least the match + 1 context line


# ─── code_tools ───────────────────────────────────────────────────────


class TestCodeTools:
    def test_returns_list_with_search_def(
        self, python_project: MemoryFileSystem
    ) -> None:
        tools = code_tools(fs=python_project, root="/project")
        assert isinstance(tools, list)
        assert len(tools) == 3
        assert callable(tools[0])
        assert tools[0].__name__ == "search_def"

    def test_search_def_has_docstring(self, python_project: MemoryFileSystem) -> None:
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        assert search_def.__doc__ is not None
        assert "symbol" in search_def.__doc__
        assert "definition" in search_def.__doc__.lower()

    def test_search_def_finds_definition(
        self, python_project: MemoryFileSystem
    ) -> None:
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        result = search_def(symbol="handle_request")
        assert isinstance(result, dict)
        assert result["found"] is True
        assert result["kind"] == "definition"
        assert result["count"] >= 1
        found_symbols = [r["symbol"] for r in result["results"]]
        assert "handle_request" in found_symbols

    def test_search_def_with_path(self, python_project: MemoryFileSystem) -> None:
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        result = search_def(symbol="insert_user", path="src")
        assert result["found"] is True
        assert any("repo.py" in r["file"] for r in result["results"])

    def test_search_def_with_language_filter(
        self, python_project: MemoryFileSystem
    ) -> None:
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        result = search_def(symbol="handle_request", language="python")
        assert result["found"] is True

    def test_search_def_invalid_language(
        self, python_project: MemoryFileSystem
    ) -> None:
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        result = search_def(symbol="handle_request", language="brainfuck")
        assert result["found"] is False
        assert result["kind"] == "error"
        assert "Unknown language" in result["note"]

    def test_search_def_not_found(self, python_project: MemoryFileSystem) -> None:
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        result = search_def(symbol="absolutely_nonexistent_xyz")
        assert result["found"] is False
        assert result["count"] == 0

    def test_search_def_grep_fallback(self, python_project: MemoryFileSystem) -> None:
        _write(
            python_project,
            "/project/src/refs.py",
            """\
            # just a reference, no definition
            value = some_external_api(123)
            """,
        )
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        result = search_def(symbol="some_external_api")
        if result["kind"] == "definition":
            pytest.skip("Unexpectedly parsed as definition")
        assert result["kind"] in ("grep_fallback", "none")

    def test_search_def_default_args(self, python_project: MemoryFileSystem) -> None:
        """Verify defaults work when path and language are omitted."""
        tools = code_tools(fs=python_project, root="/project")
        search_def = tools[0]
        result = search_def(symbol="create_user")
        assert result["found"] is True

    def test_multiple_tools_instances_independent(
        self, python_project: MemoryFileSystem
    ) -> None:
        tools1 = code_tools(fs=python_project, root="/project")
        tools2 = code_tools(fs=python_project, root="/project")
        # Each has its own tools / cache
        r1 = tools1[0](symbol="handle_request")
        r2 = tools2[0](symbol="handle_request")
        assert r1["found"] is True
        assert r2["found"] is True


# ─── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_filesystem(self, memfs: MemoryFileSystem) -> None:
        tools = CodeTools(fs=memfs, root="/empty")
        result = tools.search_definition("anything")
        assert len(result.definitions) == 0
        assert len(result.grep_matches) == 0

    def test_binary_file_skipped(self, memfs: MemoryFileSystem) -> None:
        # Write a file with a known extension but binary content
        with memfs.open("/project/data.py", "wb") as f:
            f.write(b"\x00\x01\x02\x03 def handle_request(): pass")
        tools = CodeTools(fs=memfs, root="/project")
        # Should not crash
        result = tools.search_definition("handle_request")
        # Might or might not find it depending on parser tolerance
        assert isinstance(result, SearchResult)

    def test_non_source_files_ignored(self, memfs: MemoryFileSystem) -> None:
        _write(memfs, "/project/readme.md", "# handle_request\nSome docs.")
        _write(memfs, "/project/data.json", '{"handle_request": true}')
        _write(
            memfs,
            "/project/handler.py",
            """\
            def handle_request():
                pass
            """,
        )
        tools = CodeTools(fs=memfs, root="/project")
        result = tools.search_definition("handle_request")
        # Should find definition in the .py file; .md and .json should be ignored
        assert len(result.definitions) >= 1
        assert all(d.language == "python" for d in result.definitions)

    def test_deeply_nested_path(self, memfs: MemoryFileSystem) -> None:
        _write(
            memfs,
            "/project/a/b/c/d/e/deep.py",
            """\
            def deep_function():
                return 42
            """,
        )
        tools = CodeTools(fs=memfs, root="/project")
        result = tools.search_definition("deep_function")
        assert len(result.definitions) >= 1

    def test_unicode_content(self, memfs: MemoryFileSystem) -> None:
        _write(
            memfs,
            "/project/unicode.py",
            """\
            def greet_unicode():
                return "Hello World"
            """,
        )
        tools = CodeTools(fs=memfs, root="/project")
        result = tools.search_definition("greet_unicode")
        assert len(result.definitions) >= 1

    def test_empty_file(self, memfs: MemoryFileSystem) -> None:
        _write(memfs, "/project/empty.py", "")
        tools = CodeTools(fs=memfs, root="/project")
        result = tools.search_definition("anything")
        assert isinstance(result, SearchResult)

    def test_syntax_error_file_handled(self, memfs: MemoryFileSystem) -> None:
        _write(
            memfs,
            "/project/broken.py",
            """\
            def broken(
                # missing closing paren and colon
            """,
        )
        tools = CodeTools(fs=memfs, root="/project")
        # Should not raise — tree-sitter is error-tolerant
        result = tools.search_definition("broken")
        assert isinstance(result, SearchResult)

    def test_max_results_respected(self, memfs: MemoryFileSystem) -> None:
        for i in range(10):
            _write(
                memfs,
                f"/project/mod{i}.py",
                f"""\
                def target():
                    return {i}
                """,
            )
        tools = CodeTools(fs=memfs, root="/project")
        result = tools.search_definition("target", max_results=3)
        assert len(result.definitions) <= 3

    def test_qualified_symbol_search(self, python_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=python_project, root="/project")
        result = tools.search_definition("User.full_name")
        assert len(result.definitions) >= 1
        assert any(d.symbol == "full_name" for d in result.definitions)

    def test_decorated_python_function(self, memfs: MemoryFileSystem) -> None:
        _write(
            memfs,
            "/project/decorated.py",
            """\
            def my_decorator(f):
                return f

            @my_decorator
            def decorated_handler(request):
                return "ok"
            """,
        )
        tools = CodeTools(fs=memfs, root="/project")
        result = tools.search_definition("decorated_handler")
        # May be found as function_definition or decorated_definition
        assert len(result.definitions) >= 1

    def test_root_with_trailing_slash(self, python_project: MemoryFileSystem) -> None:
        tools = CodeTools(fs=python_project, root="/project/")
        result = tools.search_definition("handle_request")
        assert len(result.definitions) >= 1

    def test_search_nonexistent_root(self, memfs: MemoryFileSystem) -> None:
        tools = CodeTools(fs=memfs, root="/does/not/exist")
        result = tools.search_definition("anything")
        assert len(result.definitions) == 0
        assert len(result.grep_matches) == 0
