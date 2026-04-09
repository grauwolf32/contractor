"""
Comprehensive tests for the code_tools module.
Run with:
    pytest test_code_tools.py -v
"""

from __future__ import annotations
from unittest.mock import MagicMock
import pytest

# ─── Import the module under test ────────────────────────────────────────────
from contractor.tools.code.tools import (
    Language,
    _EXT_TO_LANG,
    detect_language,
    _NodeSpec,
    _specs_for,
    SymbolEntry,
    DefinitionResult,
    GrepMatch,
    SearchResult,
    _ParsedFile,
    _CacheEntry,
    _context_snippet,
    _symbol_matches,
    _grep_file_lines,
    CodeTools,
    _parse_language,
    code_tools,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def make_mock_fs(files: dict[str, str]) -> MagicMock:
    """Create a mock AbstractFileSystem with given path→content mapping.

    Wires up both legacy (``cat_file`` / ``find``) and modern
    (``read_text`` / ``walk``) interfaces so tests work with
    production code that uses either path.
    """
    fs = MagicMock()

    # ── derive directory structure for walk() ──
    from pathlib import PurePosixPath

    dir_contents: dict[str, tuple[list[str], list[str]]] = {}
    all_dirs: set[str] = set()

    for fpath in files:
        parent = str(PurePosixPath(fpath).parent)
        fname = PurePosixPath(fpath).name
        # register every ancestor as a directory
        p = PurePosixPath(fpath).parent
        while str(p) != p.root and str(p) != ".":
            all_dirs.add(str(p))
            p = p.parent
        if str(PurePosixPath(fpath).parent.parent) != str(
            PurePosixPath(fpath).parent
        ):
            all_dirs.add(str(PurePosixPath(fpath).parent.parent))

        if parent not in dir_contents:
            dir_contents[parent] = ([], [])
        dir_contents[parent][1].append(fname)

    # fill in sub-directory lists
    for d in list(dir_contents.keys()):
        for d2 in list(dir_contents.keys()):
            parent_of_d2 = str(PurePosixPath(d2).parent)
            if parent_of_d2 == d and d2 != d:
                dirname = PurePosixPath(d2).name
                if dirname not in dir_contents[d][0]:
                    dir_contents[d][0].append(dirname)

    # also register dirs that have no files directly
    for d in all_dirs:
        if d not in dir_contents:
            dir_contents[d] = ([], [])
            # check if any existing dir is a child
            for d2 in list(dir_contents.keys()):
                if str(PurePosixPath(d2).parent) == d and d2 != d:
                    dirname = PurePosixPath(d2).name
                    if dirname not in dir_contents[d][0]:
                        dir_contents[d][0].append(dirname)

    def walk(root, *args, **kwargs):
        """Mock fs.walk that ignores maxdepth (not needed for shallow test trees)."""
        root = root.rstrip("/")
        results = []
        for dirpath in sorted(dir_contents.keys()):
            if dirpath == root or dirpath.startswith(root + "/"):
                subdirs, filenames = dir_contents[dirpath]
                results.append((dirpath, subdirs, filenames))
        return iter(results)

    fs.walk.side_effect = walk

    # ── read_text: returns str ──
    def read_text(path, encoding="utf-8", errors="strict"):
        if path in files:
            content = files[path]
            if isinstance(content, bytes):
                return content.decode(encoding, errors=errors)
            return content
        raise FileNotFoundError(f"No such file: {path}")

    fs.read_text.side_effect = read_text

    # ── cat_file: returns bytes (legacy, some tests use directly) ──
    def cat_file(path):
        if path in files:
            content = files[path]
            if isinstance(content, str):
                return content.encode("utf-8")
            return content
        raise FileNotFoundError(f"No such file: {path}")

    fs.cat_file.side_effect = cat_file

    # ── find: returns list of paths (legacy, some tests use directly) ──
    fs.find.return_value = list(files.keys())

    # ── exists / isfile ──
    def exists(path):
        path = path.rstrip("/")
        return path in files or path in all_dirs

    def isfile(path):
        return path in files

    fs.exists.side_effect = exists
    fs.isfile.side_effect = isfile

    return fs


SIMPLE_PYTHON = """\
def hello(name):
    return f"Hello, {name}"

class Greeter:
    def greet(self):
        pass
"""

SIMPLE_JS = """\
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(x, y) {
        return x * y;
    }
}
"""

SIMPLE_GO = """\
package main

func Add(a, b int) int {
    return a + b
}

type Point struct {
    X, Y int
}
"""

SIMPLE_RUST = """\
fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}

struct Config {
    debug: bool,
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Language Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectLanguage:
    """Tests for detect_language()."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("foo.py", Language.PYTHON),
            ("foo.js", Language.JAVASCRIPT),
            ("foo.mjs", Language.JAVASCRIPT),
            ("foo.cjs", Language.JAVASCRIPT),
            ("foo.jsx", Language.JAVASCRIPT),
            ("foo.ts", Language.TYPESCRIPT),
            ("foo.tsx", Language.TSX),
            ("foo.go", Language.GO),
            ("foo.rs", Language.RUST),
            ("foo.java", Language.JAVA),
            ("foo.kt", Language.KOTLIN),
            ("foo.kts", Language.KOTLIN),
            ("foo.c", Language.C),
            ("foo.h", Language.C),
            ("foo.cpp", Language.CPP),
            ("foo.cc", Language.CPP),
            ("foo.cxx", Language.CPP),
            ("foo.hpp", Language.CPP),
            ("foo.hxx", Language.CPP),
            ("foo.cs", Language.CSHARP),
            ("foo.rb", Language.RUBY),
            ("foo.php", Language.PHP),
            ("foo.scala", Language.SCALA),
            ("foo.sc", Language.SCALA),
            ("foo.swift", Language.SWIFT),
            ("foo.lua", Language.LUA),
            ("foo.ex", Language.ELIXIR),
            ("foo.exs", Language.ELIXIR),
            ("foo.hs", Language.HASKELL),
            ("foo.lhs", Language.HASKELL),
            ("foo.sh", Language.BASH),
            ("foo.bash", Language.BASH),
        ],
    )
    def test_known_extensions(self, path, expected):
        assert detect_language(path) == expected

    @pytest.mark.parametrize(
        "path",
        ["foo.txt", "foo.md", "foo.json", "foo.yaml", "foo", ""],
    )
    def test_unknown_extension_returns_none(self, path):
        assert detect_language(path) is None

    def test_case_insensitive(self):
        assert detect_language("foo.PY") == Language.PYTHON
        assert detect_language("foo.JS") == Language.JAVASCRIPT

    def test_nested_path(self):
        assert detect_language("src/lib/utils.py") == Language.PYTHON

    def test_posix_path_handling(self):
        assert detect_language("a/b/c/module.ts") == Language.TYPESCRIPT

    def test_ext_to_lang_coverage(self):
        """Every entry in _EXT_TO_LANG must be reachable via detect_language."""
        for ext, lang in _EXT_TO_LANG.items():
            assert detect_language(f"dummy{ext}") == lang


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  _specs_for
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpecsFor:
    """Tests for _specs_for()."""

    def test_python_has_function_and_class(self):
        specs = _specs_for(Language.PYTHON)
        types = {s.node_type for s in specs}
        assert "function_definition" in types
        assert "class_definition" in types

    def test_javascript_includes_arrow_function(self):
        specs = _specs_for(Language.JAVASCRIPT)
        types = {s.node_type for s in specs}
        assert "arrow_function" in types

    def test_go_has_function_and_method(self):
        specs = _specs_for(Language.GO)
        types = {s.node_type for s in specs}
        assert "function_declaration" in types
        assert "method_declaration" in types

    def test_rust_has_function_and_struct(self):
        specs = _specs_for(Language.RUST)
        types = {s.node_type for s in specs}
        assert "function_item" in types
        assert "struct_item" in types

    def test_all_languages_return_list(self):
        for lang in Language:
            result = _specs_for(lang)
            assert isinstance(result, list)

    def test_all_languages_nonempty(self):
        """Every supported language must have at least one spec."""
        for lang in Language:
            specs = _specs_for(lang)
            assert len(specs) > 0, f"{lang} has no specs"

    def test_node_spec_fields(self):
        specs = _specs_for(Language.PYTHON)
        for spec in specs:
            assert isinstance(spec, _NodeSpec)
            assert isinstance(spec.node_type, str)
            assert isinstance(spec.name_field, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataclasses:
    def test_symbol_entry_fields(self):
        entry = SymbolEntry(
            name="foo",
            file="a.py",
            line=1,
            end_line=5,
            node_type="function_definition",
            language="python",
        )
        assert entry.name == "foo"
        assert entry.file == "a.py"
        assert entry.line == 1
        assert entry.end_line == 5
        assert entry.node_type == "function_definition"
        assert entry.language == "python"

    def test_definition_result_location(self):
        dr = DefinitionResult(
            symbol="bar",
            file="b.py",
            line=10,
            end_line=20,
            column=4,
            node_type="class_definition",
            language="python",
            context="class bar:\n    pass",
        )
        assert dr.location == "b.py:10"

    def test_grep_match_location(self):
        gm = GrepMatch(file="c.py", line=7, text="7: something")
        assert gm.location == "c.py:7"

    def test_symbol_entry_is_frozen(self):
        entry = SymbolEntry("x", "f.py", 1, 2, "t", "python")
        with pytest.raises((AttributeError, TypeError)):
            entry.name = "y"  # type: ignore

    def test_definition_result_is_frozen(self):
        dr = DefinitionResult("x", "f.py", 1, 2, 0, "t", "python", "ctx")
        with pytest.raises((AttributeError, TypeError)):
            dr.symbol = "z"  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SearchResult.to_dict
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchResultToDict:
    def _make_def(self, symbol="fn", file="a.py", line=1) -> DefinitionResult:
        return DefinitionResult(
            symbol=symbol,
            file=file,
            line=line,
            end_line=line + 5,
            column=0,
            node_type="function_definition",
            language="python",
            context="def fn(): pass",
        )

    def _make_grep(self, file="a.py", line=2) -> GrepMatch:
        return GrepMatch(file=file, line=line, text="2: fn()")

    def test_with_definitions(self):
        sr = SearchResult(
            symbol="fn",
            definitions=[self._make_def()],
            grep_matches=[],
            is_fallback=False,
        )
        d = sr.to_dict()
        assert d["kind"] == "definition"
        assert d["total_items"] == 1
        assert d["result"][0]["symbol"] == "fn"
        assert d["result"][0]["file"] == "a.py"
        assert d["result"][0]["line"] == 1

    def test_with_grep_matches(self):
        sr = SearchResult(
            symbol="fn",
            definitions=[],
            grep_matches=[self._make_grep()],
            is_fallback=True,
        )
        d = sr.to_dict()
        assert d["kind"] == "grep_fallback"
        assert d["total_items"] == 1
        assert "note" in d
        assert d["result"][0]["file"] == "a.py"

    def test_empty(self):
        sr = SearchResult(
            symbol="fn",
            definitions=[],
            grep_matches=[],
            is_fallback=False,
        )
        d = sr.to_dict()
        assert d["kind"] == "none"
        assert d["total_items"] == 0
        assert "note" in d

    def test_definitions_take_priority_over_grep(self):
        """When both definitions and grep_matches exist, definitions win."""
        sr = SearchResult(
            symbol="fn",
            definitions=[self._make_def()],
            grep_matches=[self._make_grep()],
            is_fallback=False,
        )
        d = sr.to_dict()
        assert d["kind"] == "definition"

    def test_multiple_definitions(self):
        sr = SearchResult(
            symbol="fn",
            definitions=[self._make_def(file="a.py"), self._make_def(file="b.py")],
            grep_matches=[],
            is_fallback=False,
        )
        d = sr.to_dict()
        assert d["total_items"] == 2
        files = {r["file"] for r in d["result"]}
        assert files == {"a.py", "b.py"}


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  _symbol_matches
# ═══════════════════════════════════════════════════════════════════════════════


class TestSymbolMatches:
    def test_exact_match(self):
        assert _symbol_matches("foo", "foo") is True

    def test_case_insensitive(self):
        assert _symbol_matches("Foo", "foo") is True
        assert _symbol_matches("foo", "FOO") is True

    def test_qualified_query_matches_leaf(self):
        # query = "MyClass.my_method", extracted = "my_method"
        assert _symbol_matches("my_method", "MyClass.my_method") is True

    def test_qualified_extracted_matches_query(self):
        # extracted = "MyClass.my_method", query = "my_method"
        assert _symbol_matches("MyClass.my_method", "my_method") is True

    def test_no_match(self):
        assert _symbol_matches("bar", "foo") is False

    def test_partial_not_enough(self):
        # Substring but not dot-separated leaf
        assert _symbol_matches("foobar", "foo") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  _grep_file_lines  (now takes content: str, not fs)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGrepFileLines:
    def test_finds_matching_line(self):
        content = "line one\nhello world\nline three"
        matches = _grep_file_lines(content, "a.py", "hello")
        assert len(matches) == 1
        assert matches[0].line == 2
        assert "hello world" in matches[0].text

    def test_no_match(self):
        content = "nothing here"
        matches = _grep_file_lines(content, "a.py", "xyz")
        assert matches == []

    def test_context_lines_included(self):
        content = "\n".join(f"line {i}" for i in range(10))
        matches = _grep_file_lines(content, "a.py", "line 5", context_lines=2)
        assert len(matches) == 1
        # context: lines 3,4,5,6,7 (0-indexed 3–7 → displayed 4–8 in snippet)
        assert "line 3" in matches[0].text or "line 4" in matches[0].text

    def test_multiple_matches_in_file(self):
        content = "foo\nbar\nfoo\nbaz"
        matches = _grep_file_lines(content, "a.py", "foo", context_lines=0)
        assert len(matches) == 2
        lines = {m.line for m in matches}
        assert lines == {1, 3}

    def test_empty_content_returns_empty(self):
        matches = _grep_file_lines("", "a.py", "foo")
        assert matches == []

    def test_returns_grep_match_type(self):
        content = "hello"
        matches = _grep_file_lines(content, "a.py", "hello")
        assert all(isinstance(m, GrepMatch) for m in matches)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CodeTools — internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeToolsInternals:
    def _make_tools(self, files: dict[str, str]) -> CodeTools:
        fs = make_mock_fs(files)
        return CodeTools(fs=fs, root="/repo")

    # ── _content_hash ────────────────────────────────────────────────────────

    def test_content_hash_deterministic(self):
        ct = self._make_tools({})
        data = b"hello world"
        assert ct._content_hash(data) == ct._content_hash(data)

    def test_content_hash_changes_with_content(self):
        ct = self._make_tools({})
        assert ct._content_hash(b"foo") != ct._content_hash(b"bar")

    def test_content_hash_length(self):
        ct = self._make_tools({})
        h = ct._content_hash(b"data")
        assert len(h) == 16

    # ── _read_file ───────────────────────────────────────────────────────────

    def test_read_file_returns_bytes(self):
        ct = self._make_tools({"/repo/a.py": "hello"})
        content = ct._read_file("/repo/a.py")
        assert content == b"hello"

    def test_read_file_missing_returns_none(self):
        fs = MagicMock()
        fs.read_text.side_effect = FileNotFoundError()
        fs.cat_file.side_effect = FileNotFoundError()
        ct = CodeTools(fs=fs, root="/repo")
        assert ct._read_file("/repo/missing.py") is None

    # ── _parse_file ──────────────────────────────────────────────────────────

    def test_parse_file_caches_result(self):
        ct = self._make_tools({})
        content = SIMPLE_PYTHON.encode()
        parsed1 = ct._parse_file("/repo/a.py", content, Language.PYTHON)
        parsed2 = ct._parse_file("/repo/a.py", content, Language.PYTHON)
        assert parsed1 is parsed2  # same object from cache

    def test_parse_file_invalidates_on_content_change(self):
        ct = self._make_tools({})
        content1 = b"def foo(): pass"
        content2 = b"def bar(): pass"
        parsed1 = ct._parse_file("/repo/a.py", content1, Language.PYTHON)
        parsed2 = ct._parse_file("/repo/a.py", content2, Language.PYTHON)
        assert parsed1 is not parsed2

    def test_parse_file_returns_parsed_file(self):
        ct = self._make_tools({})
        content = SIMPLE_PYTHON.encode()
        parsed = ct._parse_file("/repo/a.py", content, Language.PYTHON)
        assert parsed is not None
        assert isinstance(parsed, _ParsedFile)
        assert parsed.language == Language.PYTHON

    # ── _grep_files ──────────────────────────────────────────────────────────

    # ── grep pre-filter (was _grep_files, now inlined in search_definition) ──

    def test_grep_prefilter_finds_matching(self):
        """search_definition only considers files containing the symbol."""
        ct = self._make_tools(
            {
                "/repo/a.py": "def hello(): pass",
                "/repo/b.py": "def world(): pass",
            }
        )
        result = ct.search_definition("hello")
        # "hello" is defined in a.py only
        found_files = {d.file for d in result.definitions}
        assert "/repo/a.py" in found_files
        assert "/repo/b.py" not in found_files

    def test_grep_prefilter_skips_non_source(self):
        """Non-source files (no known extension) are never searched."""
        ct = self._make_tools(
            {
                "/repo/a.py": "def hello(): pass",
                "/repo/readme.md": "hello is great",
            }
        )
        result = ct.search_definition("hello")
        found_files = {d.file for d in result.definitions}
        assert "/repo/readme.md" not in found_files

    def test_grep_prefilter_finds_across_multiple_files(self):
        """When multiple files contain the symbol, all are considered."""
        files = {
            "/repo/c.py": "def hello(): pass",
            "/repo/a.py": "def hello(): pass",
            "/repo/b.py": "def hello(): pass",
        }
        ct = self._make_tools(files)
        result = ct.search_definition("hello")
        found_files = sorted(d.file for d in result.definitions)
        assert len(found_files) == 3
        # Results should be sorted by file
        assert found_files == sorted(found_files)

    def test_grep_prefilter_empty_when_root_not_found(self):
        """When root doesn't exist, no files are found."""
        fs = MagicMock()
        fs.exists.return_value = False
        fs.isfile.return_value = False
        fs.walk.return_value = iter([])
        ct = CodeTools(fs=fs, root="/missing")
        result = ct.search_definition("anything")
        assert result.definitions == []
        assert result.grep_matches == []

    # ── clear_cache / invalidate_file ────────────────────────────────────────

    def test_clear_cache(self):
        ct = self._make_tools({})
        content = b"def foo(): pass"
        ct._parse_file("/repo/a.py", content, Language.PYTHON)
        assert len(ct._parse_cache) == 1
        ct.clear_cache()
        assert len(ct._parse_cache) == 0
        assert len(ct._resolution_cache) == 0

    def test_invalidate_file_removes_from_caches(self):
        ct = self._make_tools({})
        content = b"def foo(): pass"
        ct._parse_file("/repo/a.py", content, Language.PYTHON)
        ct._resolution_cache[("foo", "/repo/a.py")] = _CacheEntry(
            content_hash="abc", results=[]
        )
        ct.invalidate_file("/repo/a.py")
        assert "/repo/a.py" not in ct._parse_cache
        assert ("foo", "/repo/a.py") not in ct._resolution_cache

    def test_invalidate_nonexistent_file_no_error(self):
        ct = self._make_tools({})
        ct.invalidate_file("/repo/not_there.py")  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CodeTools.search_definition — integration tests (real parser)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchDefinition:
    def _make_tools(self, files: dict[str, str]) -> CodeTools:
        fs = make_mock_fs(files)
        return CodeTools(fs=fs, root="/repo")

    def test_finds_python_function(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        result = ct.search_definition("hello")
        assert result.definitions
        assert result.definitions[0].symbol == "hello"
        assert result.definitions[0].node_type == "function_definition"

    def test_finds_python_class(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        result = ct.search_definition("Greeter")
        assert result.definitions
        assert result.definitions[0].node_type == "class_definition"

    def test_not_found_returns_grep_fallback(self):
        content = "x = hello()\n"
        ct = self._make_tools({"/repo/a.py": content})
        result = ct.search_definition("hello")
        # No tree-sitter def, but grep finds the usage
        assert result.is_fallback or not result.definitions

    def test_not_found_at_all(self):
        ct = self._make_tools({"/repo/a.py": "x = 1\n"})
        result = ct.search_definition("nonexistent_symbol_xyz")
        assert result.definitions == []

    def test_language_filter_excludes_other_langs(self):
        ct = self._make_tools(
            {
                "/repo/a.py": SIMPLE_PYTHON,
                "/repo/b.js": "function hello() {}",
            }
        )
        result = ct.search_definition("hello", language_filter=Language.PYTHON)
        assert all(d.language == "python" for d in result.definitions)

    def test_definitions_sorted_by_file_and_line(self):
        ct = self._make_tools(
            {
                "/repo/b.py": "def hello(): pass\n",
                "/repo/a.py": "def hello(): pass\n",
            }
        )
        result = ct.search_definition("hello")
        if len(result.definitions) > 1:
            files = [d.file for d in result.definitions]
            assert files == sorted(files)

    def test_max_results_respected(self):
        files = {f"/repo/{chr(97+i)}.py": "def hello(): pass\n" for i in range(10)}
        ct = self._make_tools(files)
        result = ct.search_definition("hello", max_results=3)
        assert len(result.definitions) <= 3

    def test_path_restriction(self):
        files = {
            "/repo/src/a.py": "def hello(): pass\n",
            "/repo/tests/b.py": "def hello(): pass\n",
        }
        fs = make_mock_fs(files)
        ct = CodeTools(fs=fs, root="/repo")
        result = ct.search_definition("hello", path="src")
        assert all("src" in d.file for d in result.definitions)

    def test_empty_candidate_list_returns_empty(self):
        fs = MagicMock()
        fs.exists.return_value = False
        fs.isfile.return_value = False
        fs.walk.return_value = iter([])
        ct = CodeTools(fs=fs, root="/repo")
        result = ct.search_definition("hello")
        assert result.definitions == []
        assert result.grep_matches == []

    def test_definition_result_fields_complete(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        result = ct.search_definition("hello")
        assert result.definitions
        dr = result.definitions[0]
        assert dr.file == "/repo/a.py"
        assert dr.line >= 1
        assert dr.end_line >= dr.line
        assert dr.column >= 0
        assert dr.context  # non-empty snippet
        assert dr.language == "python"

    def test_resolution_cache_used_on_second_call(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        ct.search_definition("hello")
        cache_size_after_first = len(ct._resolution_cache)
        ct.search_definition("hello")
        # Cache should not grow on second identical call
        assert len(ct._resolution_cache) == cache_size_after_first


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CodeTools.list_symbols
# ═══════════════════════════════════════════════════════════════════════════════


class TestListSymbols:
    def _make_tools(self, files: dict[str, str]) -> CodeTools:
        fs = make_mock_fs(files)
        return CodeTools(fs=fs, root="/repo")

    def test_lists_python_symbols(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        symbols = ct.list_symbols()
        names = {s.name for s in symbols}
        assert "hello" in names
        assert "Greeter" in names

    def test_language_filter(self):
        ct = self._make_tools(
            {
                "/repo/a.py": SIMPLE_PYTHON,
                "/repo/b.js": SIMPLE_JS,
            }
        )
        symbols = ct.list_symbols(language_filter=Language.PYTHON)
        assert all(s.language == "python" for s in symbols)

    def test_node_type_filter(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        symbols = ct.list_symbols(node_type_filter="function_definition")
        assert all(s.node_type == "function_definition" for s in symbols)

    def test_empty_when_no_source_files(self):
        fs = make_mock_fs({"/repo/readme.md": "# Hello"})
        ct = CodeTools(fs=fs, root="/repo")
        symbols = ct.list_symbols()
        assert symbols == []

    def test_returns_symbol_entry_types(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        symbols = ct.list_symbols()
        assert all(isinstance(s, SymbolEntry) for s in symbols)

    def test_root_not_found_returns_empty(self):
        fs = MagicMock()
        fs.exists.return_value = False
        fs.isfile.return_value = False
        fs.walk.return_value = iter([])
        ct = CodeTools(fs=fs, root="/missing")
        assert ct.list_symbols() == []

    def test_symbol_entry_line_numbers_positive(self):
        ct = self._make_tools({"/repo/a.py": SIMPLE_PYTHON})
        symbols = ct.list_symbols()
        assert all(s.line >= 1 for s in symbols)
        assert all(s.end_line >= s.line for s in symbols)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. _context_snippet
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextSnippet:
    def _make_node(self, source: bytes, start: int, end: int) -> MagicMock:
        node = MagicMock()
        node.start_byte = start
        node.end_byte = end
        return node

    def test_short_function_returned_in_full(self):
        src = b"def foo():\n    pass\n"
        node = self._make_node(src, 0, len(src))
        snippet = _context_snippet(src, node, max_lines=15)
        assert "def foo():" in snippet
        assert "pass" in snippet

    def test_truncates_long_function(self):
        lines = [f"line_{i} = {i}" for i in range(30)]
        src = "\n".join(lines).encode()
        node = self._make_node(src, 0, len(src))
        snippet = _context_snippet(src, node, max_lines=5)
        assert "more lines" in snippet

    def test_max_lines_respected(self):
        lines = [f"x = {i}" for i in range(20)]
        src = "\n".join(lines).encode()
        node = self._make_node(src, 0, len(src))
        snippet = _context_snippet(src, node, max_lines=10)
        snippet_lines = snippet.splitlines()
        # 10 content lines + 1 ellipsis line
        assert len(snippet_lines) == 11


# ═══════════════════════════════════════════════════════════════════════════════
# 12. _parse_language
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseLanguage:
    def test_valid_language(self):
        assert _parse_language("python") == Language.PYTHON
        assert _parse_language("javascript") == Language.JAVASCRIPT

    def test_empty_string_returns_none(self):
        assert _parse_language("") is None

    def test_invalid_returns_error_dict(self):
        result = _parse_language("cobol")
        assert isinstance(result, dict)
        assert "error" in result
        assert "cobol" in result["error"]
        assert result["result"] == []
        assert result["total_items"] == 0

    def test_all_language_values_parseable(self):
        for lang in Language:
            parsed = _parse_language(lang.value)
            assert parsed == lang


# ═══════════════════════════════════════════════════════════════════════════════
# 13. code_tools factory — search_def tool
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeToolsFactory:
    def _make_tools_list(self, files: dict[str, str]):
        fs = make_mock_fs(files)
        return code_tools(fs=fs, root="/repo")

    def test_returns_two_tools(self):
        tools = self._make_tools_list({})
        assert len(tools) == 2

    def test_tools_are_callable(self):
        tools = self._make_tools_list({})
        for t in tools:
            assert callable(t)

    def test_search_def_finds_python(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        search_def, _ = tools
        result = search_def("hello")
        assert result["kind"] in ("definition", "grep_fallback", "none")

    def test_search_def_definition_kind(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        search_def, _ = tools
        result = search_def("hello")
        assert result["kind"] == "definition"
        assert result["total_items"] >= 1

    def test_search_def_invalid_language_returns_error(self):
        tools = self._make_tools_list({})
        search_def, _ = tools
        result = search_def("foo", language="cobol")
        assert "error" in result

    def test_search_def_not_found_returns_none_kind(self):
        tools = self._make_tools_list({"/repo/a.py": "x = 1\n"})
        search_def, _ = tools
        result = search_def("totally_missing_xyz")
        assert result["kind"] == "none"

    def test_list_symbols_returns_expected_structure(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        _, list_syms = tools
        result = list_syms()
        assert "result" in result
        assert "total_items" in result
        assert "offset" in result
        assert "limit" in result

    def test_list_symbols_pagination(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        _, list_syms = tools
        result_all = list_syms()
        total = result_all["total_items"]
        result_page = list_syms(offset=0, limit=1)
        assert len(result_page["result"]) == min(1, total)
        assert result_page["total_items"] == total

    def test_list_symbols_offset(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        _, list_syms = tools
        result_all = list_syms()
        result_offset = list_syms(offset=1)
        if result_all["total_items"] > 1:
            assert result_offset["result"][0] != result_all["result"][0]

    def test_list_symbols_language_filter(self):
        tools = self._make_tools_list(
            {
                "/repo/a.py": SIMPLE_PYTHON,
                "/repo/b.js": SIMPLE_JS,
            }
        )
        _, list_syms = tools
        result = list_syms(language="python")
        assert all(s["language"] == "python" for s in result["result"])

    def test_list_symbols_invalid_language(self):
        tools = self._make_tools_list({})
        _, list_syms = tools
        result = list_syms(language="cobol")
        assert "error" in result

    def test_list_symbols_node_type_filter(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        _, list_syms = tools
        result = list_syms(node_type="function_definition")
        assert all(
            s["node_type"] == "function_definition" for s in result["result"]
        )

    def test_list_symbols_default_limit(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        _, list_syms = tools
        result = list_syms()
        assert result["limit"] == 300

    def test_search_def_result_dict_structure(self):
        tools = self._make_tools_list({"/repo/a.py": SIMPLE_PYTHON})
        search_def, _ = tools
        result = search_def("hello")
        assert "result" in result
        assert "kind" in result
        assert "total_items" in result
        if result["kind"] == "definition":
            item = result["result"][0]
            for key in (
                "symbol",
                "file",
                "line",
                "end_line",
                "column",
                "node_type",
                "language",
                "context",
            ):
                assert key in item


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Multi-language integration tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiLanguageIntegration:
    def _ct(self, files: dict[str, str]) -> CodeTools:
        fs = make_mock_fs(files)
        return CodeTools(fs=fs, root="/repo")

    def test_go_function_found(self):
        ct = self._ct({"/repo/main.go": SIMPLE_GO})
        result = ct.search_definition("Add")
        assert result.definitions
        assert result.definitions[0].language == "go"

    def test_rust_function_found(self):
        ct = self._ct({"/repo/lib.rs": SIMPLE_RUST})
        result = ct.search_definition("greet")
        assert result.definitions
        assert result.definitions[0].language == "rust"

    def test_rust_struct_found(self):
        ct = self._ct({"/repo/lib.rs": SIMPLE_RUST})
        result = ct.search_definition("Config")
        assert result.definitions
        assert result.definitions[0].node_type == "struct_item"

    def test_javascript_function_found(self):
        ct = self._ct({"/repo/app.js": SIMPLE_JS})
        result = ct.search_definition("add")
        assert result.definitions
        assert result.definitions[0].language == "javascript"

    def test_javascript_class_found(self):
        ct = self._ct({"/repo/app.js": SIMPLE_JS})
        result = ct.search_definition("Calculator")
        assert result.definitions

    def test_list_symbols_go(self):
        ct = self._ct({"/repo/main.go": SIMPLE_GO})
        symbols = ct.list_symbols()
        names = {s.name for s in symbols}
        assert "Add" in names

    def test_list_symbols_rust(self):
        ct = self._ct({"/repo/lib.rs": SIMPLE_RUST})
        symbols = ct.list_symbols()
        names = {s.name for s in symbols}
        assert "greet" in names
        assert "Config" in names

    def test_mixed_repo_all_langs(self):
        ct = self._ct(
            {
                "/repo/a.py": SIMPLE_PYTHON,
                "/repo/b.js": SIMPLE_JS,
                "/repo/c.go": SIMPLE_GO,
                "/repo/d.rs": SIMPLE_RUST,
            }
        )
        symbols = ct.list_symbols()
        langs = {s.language for s in symbols}
        assert "python" in langs
        assert "javascript" in langs
        assert "go" in langs
        assert "rust" in langs


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_file(self):
        fs = make_mock_fs({"/repo/a.py": ""})
        ct = CodeTools(fs=fs, root="/repo")
        symbols = ct.list_symbols()
        assert symbols == []

    def test_binary_like_content_does_not_crash(self):
        # Create a mock that returns binary content via read_text (decoded)
        content = bytes(range(256))
        fs = make_mock_fs({"/repo/a.py": content.decode("utf-8", errors="ignore")})
        ct = CodeTools(fs=fs, root="/repo")
        # Should not raise
        result = ct.search_definition("anything")
        assert isinstance(result, SearchResult)

    def test_very_long_function_context_truncated(self):
        many_lines = (
            "def big():\n"
            + "\n".join(f"    x_{i} = {i}" for i in range(100))
            + "\n"
        )
        fs = make_mock_fs({"/repo/a.py": many_lines})
        ct = CodeTools(fs=fs, root="/repo", max_context_lines=15)
        result = ct.search_definition("big")
        if result.definitions:
            ctx_lines = result.definitions[0].context.splitlines()
            assert len(ctx_lines) <= 16  # 15 + ellipsis

    def test_symbol_with_dot_notation(self):
        fs = make_mock_fs({"/repo/a.py": SIMPLE_PYTHON})
        ct = CodeTools(fs=fs, root="/repo")
        # "Greeter.greet" should find "greet" via leaf matching
        result = ct.search_definition("Greeter.greet")
        # May or may not find depending on extraction; should not crash
        assert isinstance(result, SearchResult)

    def test_search_with_unicode_content(self):
        content = 'def héllo():\n    return "Héllo"\n'
        fs = make_mock_fs({"/repo/a.py": content})
        ct = CodeTools(fs=fs, root="/repo")
        result = ct.search_definition("héllo")
        assert isinstance(result, SearchResult)

    def test_repeated_search_stable(self):
        """Searching the same symbol twice returns identical results."""
        files = {f"/repo/{i}.py": f"def fn_{i}(): pass\n" for i in range(20)}
        fs = make_mock_fs(files)
        ct = CodeTools(fs=fs, root="/repo")
        result1 = ct.search_definition("fn_0")
        result2 = ct.search_definition("fn_0")
        assert len(result1.definitions) == len(result2.definitions)
        assert [d.file for d in result1.definitions] == [
            d.file for d in result2.definitions
        ]

    def test_file_read_error_during_search(self):
        """When read_text fails for a file, search should skip it gracefully."""
        fs = MagicMock()
        fs.exists.return_value = True
        fs.isfile.return_value = False
        # walk returns one file
        fs.walk.return_value = iter([("/repo", [], ["a.py"])])
        # read_text always fails — the file can't be read at all
        fs.read_text.side_effect = IOError("read error")
        ct = CodeTools(fs=fs, root="/repo")
        # Should not raise; file is simply skipped
        result = ct.search_definition("hello")
        assert isinstance(result, SearchResult)
        assert result.definitions == []
        assert result.grep_matches == []

    def test_search_def_grep_fallback_respects_max(self):
        content = "hello\n" * 30
        files = {f"/repo/{i}.py": content for i in range(5)}
        fs = make_mock_fs(files)
        ct = CodeTools(fs=fs, root="/repo")
        result = ct.search_definition("hello", max_grep_results=5)
        assert len(result.grep_matches) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# 16. _resolve_root path handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveRoot:
    """Tests for CodeTools._resolve_root to prevent the double-slash hang."""

    def _ct(self, root: str = "/repo") -> CodeTools:
        fs = MagicMock()
        return CodeTools(fs=fs, root=root)

    def test_empty_path_returns_root(self):
        ct = self._ct("/repo")
        assert ct._resolve_root("") == "/repo"

    def test_relative_path_joined(self):
        ct = self._ct("/repo")
        resolved = ct._resolve_root("src")
        assert resolved == "/repo/src"

    def test_absolute_path_used_directly(self):
        ct = self._ct("/repo")
        resolved = ct._resolve_root("/online/wsd")
        assert resolved == "/online/wsd"

    def test_no_double_slashes(self):
        ct = self._ct("/")
        resolved = ct._resolve_root("/online/wsd")
        assert "//" not in resolved

    def test_no_double_slashes_with_trailing_root(self):
        ct = self._ct("/repo/")
        resolved = ct._resolve_root("src/")
        assert "//" not in resolved

    def test_bare_slash_root_empty_path(self):
        ct = self._ct("/")
        resolved = ct._resolve_root("")
        assert resolved == "/"

    def test_bare_slash_root_relative_path(self):
        ct = self._ct("/")
        resolved = ct._resolve_root("src")
        assert resolved == "/src"

    def test_relative_path_with_leading_slash_stripped(self):
        """Even if path looks absolute, no double slashes appear."""
        ct = self._ct("/repo")
        resolved = ct._resolve_root("/sub/dir")
        assert "//" not in resolved