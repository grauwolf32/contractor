from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Iterator, Optional

from tree_sitter import Node, Parser, Tree
from tree_sitter_language_pack import get_parser
from fsspec import AbstractFileSystem

from contractor.utils.formatting import (
    norm_unicode,
    normalize_slashes,
)

logger = logging.getLogger(__name__)

# Guard against infinite directory traversal.
_MAX_WALK_DEPTH = 50
# Maximum number of files to visit in a single traversal to prevent runaway scans.
_MAX_FILES_PER_WALK = 100_000


# ─── Language Detection ───────────────────────────────────────────────
class Language(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    KOTLIN = "kotlin"
    C = "c"
    CPP = "cpp"
    CSHARP = "c_sharp"
    RUBY = "ruby"
    PHP = "php"
    SCALA = "scala"
    SWIFT = "swift"
    LUA = "lua"
    ELIXIR = "elixir"
    HASKELL = "haskell"
    BASH = "bash"


_EXT_TO_LANG: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".hxx": Language.CPP,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".scala": Language.SCALA,
    ".sc": Language.SCALA,
    ".swift": Language.SWIFT,
    ".lua": Language.LUA,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    ".hs": Language.HASKELL,
    ".lhs": Language.HASKELL,
    ".sh": Language.BASH,
    ".bash": Language.BASH,
}


def detect_language(path: str) -> Optional[Language]:
    """Detect language from file extension."""
    suffix = PurePosixPath(path).suffix.lower()
    return _EXT_TO_LANG.get(suffix)


# ─── Node type tables per language ────────────────────────────────────
@dataclass(frozen=True)
class _NodeSpec:
    node_type: str
    name_field: str


def _specs_for(lang: Language) -> list[_NodeSpec]:
    match lang:
        case Language.PYTHON:
            return [
                _NodeSpec("function_definition", "name"),
                _NodeSpec("async_function_definition", "name"),
                _NodeSpec("class_definition", "name"),
                _NodeSpec("decorated_definition", "definition"),
            ]
        case Language.JAVASCRIPT | Language.TYPESCRIPT | Language.TSX:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("function_expression", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("class_expression", "name"),
                _NodeSpec("method_definition", "name"),
                _NodeSpec("arrow_function", ""),
                _NodeSpec("variable_declarator", "name"),
                _NodeSpec("export_statement", ""),
                _NodeSpec("lexical_declaration", ""),
                _NodeSpec("variable_declaration", ""),
                _NodeSpec("interface_declaration", "name"),
                _NodeSpec("type_alias_declaration", "name"),
                _NodeSpec("enum_declaration", "name"),
                _NodeSpec("abstract_class_declaration", "name"),
            ]
        case Language.GO:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("method_declaration", "name"),
                _NodeSpec("type_declaration", ""),
                _NodeSpec("type_spec", "name"),
                _NodeSpec("var_declaration", ""),
                _NodeSpec("const_declaration", ""),
                _NodeSpec("var_spec", "name"),
                _NodeSpec("const_spec", "name"),
            ]
        case Language.RUST:
            return [
                _NodeSpec("function_item", "name"),
                _NodeSpec("struct_item", "name"),
                _NodeSpec("enum_item", "name"),
                _NodeSpec("trait_item", "name"),
                _NodeSpec("impl_item", ""),
                _NodeSpec("type_item", "name"),
                _NodeSpec("const_item", "name"),
                _NodeSpec("static_item", "name"),
                _NodeSpec("macro_definition", "name"),
                _NodeSpec("mod_item", "name"),
            ]
        case Language.JAVA:
            return [
                _NodeSpec("method_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("interface_declaration", "name"),
                _NodeSpec("constructor_declaration", "name"),
                _NodeSpec("enum_declaration", "name"),
                _NodeSpec("annotation_type_declaration", "name"),
                _NodeSpec("record_declaration", "name"),
            ]
        case Language.KOTLIN:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("object_declaration", "name"),
                _NodeSpec("interface_declaration", "name"),
                _NodeSpec("secondary_constructor", ""),
                _NodeSpec("companion_object", "name"),
                _NodeSpec("property_declaration", ""),
            ]
        case Language.C:
            return [
                _NodeSpec("function_definition", "declarator"),
                _NodeSpec("declaration", "declarator"),
                _NodeSpec("struct_specifier", "name"),
                _NodeSpec("union_specifier", "name"),
                _NodeSpec("enum_specifier", "name"),
                _NodeSpec("type_definition", "declarator"),
            ]
        case Language.CPP:
            return [
                _NodeSpec("function_definition", "declarator"),
                _NodeSpec("declaration", "declarator"),
                _NodeSpec("class_specifier", "name"),
                _NodeSpec("struct_specifier", "name"),
                _NodeSpec("union_specifier", "name"),
                _NodeSpec("enum_specifier", "name"),
                _NodeSpec("template_declaration", ""),
                _NodeSpec("namespace_definition", "name"),
                _NodeSpec("type_definition", "declarator"),
                _NodeSpec("alias_declaration", "name"),
            ]
        case Language.CSHARP:
            return [
                _NodeSpec("method_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("interface_declaration", "name"),
                _NodeSpec("struct_declaration", "name"),
                _NodeSpec("enum_declaration", "name"),
                _NodeSpec("constructor_declaration", "name"),
                _NodeSpec("property_declaration", "name"),
                _NodeSpec("delegate_declaration", "name"),
                _NodeSpec("record_declaration", "name"),
                _NodeSpec("namespace_declaration", "name"),
                _NodeSpec("local_function_statement", "name"),
            ]
        case Language.RUBY:
            return [
                _NodeSpec("method", "name"),
                _NodeSpec("singleton_method", "name"),
                _NodeSpec("class", "name"),
                _NodeSpec("module", "name"),
                _NodeSpec("do_block", ""),
            ]
        case Language.PHP:
            return [
                _NodeSpec("function_definition", "name"),
                _NodeSpec("method_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("interface_declaration", "name"),
                _NodeSpec("trait_declaration", "name"),
                _NodeSpec("enum_declaration", "name"),
                _NodeSpec("arrow_function", ""),
            ]
        case Language.SCALA:
            return [
                _NodeSpec("function_definition", "name"),
                _NodeSpec("class_definition", "name"),
                _NodeSpec("object_definition", "name"),
                _NodeSpec("trait_definition", "name"),
                _NodeSpec("val_definition", "pattern"),
                _NodeSpec("var_definition", "pattern"),
                _NodeSpec("type_definition", "name"),
            ]
        case Language.SWIFT:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("struct_declaration", "name"),
                _NodeSpec("enum_declaration", "name"),
                _NodeSpec("protocol_declaration", "name"),
                _NodeSpec("extension_declaration", ""),
                _NodeSpec("typealias_declaration", "name"),
                _NodeSpec("init_declaration", ""),
                _NodeSpec("subscript_declaration", ""),
                _NodeSpec("computed_property", ""),
            ]
        case Language.LUA:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("local_function_declaration", "name"),
                _NodeSpec("function_definition", ""),
                _NodeSpec("assignment_statement", ""),
                _NodeSpec("local_variable_declaration", ""),
            ]
        case Language.ELIXIR:
            return [
                _NodeSpec("call", ""),
            ]
        case Language.HASKELL:
            return [
                _NodeSpec("function", ""),
                _NodeSpec("signature", ""),
                _NodeSpec("data_declaration", ""),
                _NodeSpec("newtype_declaration", ""),
                _NodeSpec("type_synonym_declaration", ""),
                _NodeSpec("class_declaration", ""),
                _NodeSpec("instance_declaration", ""),
            ]
        case Language.BASH:
            return [
                _NodeSpec("function_definition", "name"),
            ]
        case _:
            logger.warning("No node specs defined for language: %s", lang)
            return []


# ─── Result dataclasses ───────────────────────────────────────────────
@dataclass(frozen=True)
class SymbolEntry:
    name: str
    file: str
    line: int
    end_line: int
    node_type: str
    language: str


@dataclass(frozen=True)
class DefinitionResult:
    symbol: str
    file: str
    line: int
    end_line: int
    column: int
    node_type: str
    language: str
    context: str

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class GrepMatch:
    file: str
    line: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class SearchResult:
    """Combined search result with definitions and optional grep fallback."""

    symbol: str
    definitions: list[DefinitionResult]
    grep_matches: list[GrepMatch]
    is_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        if self.definitions:
            return {
                "result": [
                    {
                        "symbol": d.symbol,
                        "file": d.file,
                        "line": d.line,
                        "end_line": d.end_line,
                        "column": d.column,
                        "node_type": d.node_type,
                        "language": d.language,
                        "context": d.context,
                    }
                    for d in self.definitions
                ],
                "kind": "definition",
                "total_items": len(self.definitions),
            }
        if self.grep_matches:
            return {
                "result": [
                    {
                        "file": g.file,
                        "line": g.line,
                        "text": g.text,
                    }
                    for g in self.grep_matches
                ],
                "kind": "grep_fallback",
                "total_items": len(self.grep_matches),
                "note": (
                    "No tree-sitter definition found. "
                    "Showing grep matches where the symbol appears."
                ),
            }
        return {
            "result": [],
            "kind": "none",
            "total_items": 0,
            "note": "Symbol not found in any source file.",
        }


# ─── Caching ──────────────────────────────────────────────────────────
@dataclass
class _ParsedFile:
    content_hash: str
    tree: Tree
    source: bytes
    language: Language


@dataclass
class _CacheEntry:
    content_hash: str
    results: list[DefinitionResult]


# ─── Filesystem helpers ───────────────────────────────────────────────
def _join_path(directory: str, filename: str) -> str:
    """Join directory and filename with forward slashes."""
    return f"{str(directory).rstrip('/')}/{filename}".replace("\\", "/")


def _iter_all_files(
    fs: AbstractFileSystem,
    root: str,
    *,
    max_depth: int = _MAX_WALK_DEPTH,
    max_files: int = _MAX_FILES_PER_WALK,
) -> Iterator[str]:
    """Yield every file path under *root* using ``fs.walk``.

    Protections against hangs:
    - *max_depth* caps how many directory levels we descend.
    - *max_files* caps the total number of file paths yielded.
    - Tracks visited directory path strings to reduce repeated traversal.

    Note:
    This is not the same as resolving real paths, so it is only a best-effort
    guard against cycles on filesystems that expose symlinked directories
    through multiple path aliases.
    """
    try:
        if not fs.exists(root):
            return
        if fs.isfile(root):
            yield root
            return
    except Exception:
        logger.debug("Cannot stat root %s", root, exc_info=True)
        return

    seen_dirs: set[str] = set()
    seen_files: set[str] = set()
    file_count = 0

    root_depth = root.rstrip("/").count("/")

    try:
        for current_path, _dirs, filenames in fs.walk(root, maxdepth=max_depth):
            current_depth = current_path.rstrip("/").count("/") - root_depth
            if current_depth > max_depth:
                continue

            normalized_dir = current_path.rstrip("/")
            if normalized_dir in seen_dirs:
                continue
            seen_dirs.add(normalized_dir)

            for filename in filenames:
                if file_count >= max_files:
                    logger.warning(
                        "File limit (%d) reached during traversal of %s",
                        max_files,
                        root,
                    )
                    return
                full_path = _join_path(current_path, filename)
                if full_path not in seen_files:
                    seen_files.add(full_path)
                    file_count += 1
                    yield full_path
    except Exception:
        logger.debug(
            "Error during walk of %s (yielded %d files so far)",
            root,
            file_count,
            exc_info=True,
        )


def _read_text_safe(fs: AbstractFileSystem, path: str) -> Optional[str]:
    """Read a file as UTF-8 text, returning ``None`` on any error."""
    try:
        return fs.read_text(path, encoding="utf-8", errors="ignore")
    except Exception:
        logger.debug("Failed to read %s", path, exc_info=True)
        return None


# ─── Core helpers ─────────────────────────────────────────────────────
def _extract_text(node: Node, source: bytes) -> str:
    return (
        source[node.start_byte : node.end_byte]
        .decode("utf-8", errors="replace")
        .strip()
    )


def _extract_name_from_field(
    node: Node,
    source: bytes,
    field_name: str,
) -> Optional[str]:
    if not field_name:
        return None

    child = node.child_by_field_name(field_name)
    if child is None:
        return None

    if child.type in {
        "function_declarator",
        "pointer_declarator",
        "parenthesized_declarator",
        "reference_declarator",
        "abstract_declarator",
    }:
        inner = _extract_name_from_node(child, source)
        if inner:
            return inner

    if child.type in {
        "function_definition",
        "class_definition",
        "async_function_definition",
    }:
        return _extract_name_from_node(child, source)

    text = _extract_text(child, source)
    ident = text.split("(")[0].split("<")[0].split("[")[0].strip()
    return ident or None


def _extract_all_definitions(
    node: Node,
    source: bytes,
    specs: list[_NodeSpec],
    file_path: str,
    language: Language,
) -> list[SymbolEntry]:
    results: list[SymbolEntry] = []
    spec_map = {s.node_type: s for s in specs}
    spec_types = set(spec_map)
    stack: list[Node] = [node]

    while stack:
        current = stack.pop()
        if current.type in spec_types:
            spec = spec_map[current.type]
            name = _extract_name_from_node(
                current, source, preferred_field=spec.name_field
            )
            if name:
                results.append(
                    SymbolEntry(
                        name=name,
                        file=file_path,
                        line=current.start_point[0] + 1,
                        end_line=current.end_point[0] + 1,
                        node_type=current.type,
                        language=language.value,
                    )
                )
        for i in range(current.child_count - 1, -1, -1):
            stack.append(current.children[i])
    return results


def _extract_name_from_node(
    node: Node,
    source: bytes,
    preferred_field: str = "",
) -> Optional[str]:
    """Best-effort extraction of the defined symbol name from a node."""
    preferred = _extract_name_from_field(node, source, preferred_field)
    if preferred:
        return preferred

    if node.type == "call" and node.child_count >= 2:
        keyword_node = node.children[0]
        keyword = _extract_text(keyword_node, source)
        if keyword in (
            "def",
            "defp",
            "defmodule",
            "defmacro",
            "defimpl",
            "defprotocol",
        ):
            args_node = node.children[1]
            if args_node.child_count > 0:
                first = args_node.children[0]
                name_text = _extract_text(first, source)
                return name_text.split("(")[0].strip()
        return None

    if node.type in ("function", "signature") and node.child_count > 0:
        return _extract_text(node.children[0], source) or None

    if node.type in (
        "data_declaration",
        "newtype_declaration",
        "type_synonym_declaration",
        "class_declaration",
        "instance_declaration",
    ):
        for child in node.children:
            if child.type == "name" or child.is_named:
                text = _extract_text(child, source)
                if text and text not in (
                    "data",
                    "newtype",
                    "type",
                    "class",
                    "instance",
                    "where",
                ):
                    return text
        return None

    if node.type == "export_statement":
        for child in node.children:
            if child.type in (
                "function_declaration",
                "class_declaration",
                "lexical_declaration",
                "variable_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
                "abstract_class_declaration",
            ):
                return _extract_name_from_node(child, source)
        return None

    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                return _extract_name_from_node(child, source)
        return None

    if node.type == "variable_declarator":
        return _extract_name_from_field(node, source, "name")

    if node.type == "property_declaration":
        for child in node.children:
            if child.type in ("simple_identifier", "identifier"):
                return _extract_text(child, source) or None
        return None

    if node.type == "secondary_constructor":
        return "constructor"

    if node.type == "extension_declaration":
        for child in node.children:
            if child.type in ("user_type", "type_identifier"):
                return _extract_text(child, source) or None
        return None

    if node.type == "init_declaration":
        return "init"

    if node.type == "subscript_declaration":
        return "subscript"

    if node.type == "computed_property":
        return None

    if node.type == "assignment_statement":
        var_list = node.child_by_field_name("variable")
        if var_list is not None:
            return _extract_text(var_list, source) or None
        return None

    if node.type == "local_variable_declaration":
        name_list = node.child_by_field_name("name")
        if name_list is not None:
            return _extract_text(name_list, source) or None
        return None

    if node.type in ("impl_item", "template_declaration"):
        text = _extract_text(node, source)
        first_line = text.splitlines()[0].strip() if text else ""
        return first_line[:80] or None

    for fname in ("name", "declarator", "pattern", "definition"):
        child = node.child_by_field_name(fname)
        if child is None:
            continue
        if child.type in (
            "function_declarator",
            "pointer_declarator",
            "parenthesized_declarator",
            "reference_declarator",
            "abstract_declarator",
        ):
            inner = _extract_name_from_node(child, source)
            if inner:
                return inner
        if child.type in (
            "function_definition",
            "class_definition",
            "async_function_definition",
        ):
            return _extract_name_from_node(child, source)
        text = _extract_text(child, source)
        ident = text.split("(")[0].split("<")[0].split("[")[0].strip()
        if ident:
            return ident

    return None


def _context_snippet(source: bytes, node: Node, max_lines: int = 15) -> str:
    text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"    ... ({len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def _symbol_matches(extracted: str, query: str) -> bool:
    if extracted == query:
        return True
    query_parts = query.rsplit(".", 1)
    if len(query_parts) == 2 and query_parts[1] == extracted:
        return True
    extracted_parts = extracted.rsplit(".", 1)
    if len(extracted_parts) == 2 and extracted_parts[1] == query:
        return True
    if extracted.lower() == query.lower():
        return True
    return False


def _walk_for_definitions(
    node: Node,
    source: bytes,
    specs: list[_NodeSpec],
    symbol: str,
    file_path: str,
    language: Language,
) -> list[DefinitionResult]:
    results: list[DefinitionResult] = []
    spec_map = {s.node_type: s for s in specs}
    spec_types = set(spec_map)
    stack: list[Node] = [node]

    while stack:
        current = stack.pop()
        if current.type in spec_types:
            spec = spec_map[current.type]
            name = _extract_name_from_node(
                current, source, preferred_field=spec.name_field
            )
            if name is not None and _symbol_matches(name, symbol):
                results.append(
                    DefinitionResult(
                        symbol=name,
                        file=file_path,
                        line=current.start_point[0] + 1,
                        end_line=current.end_point[0] + 1,
                        column=current.start_point[1],
                        node_type=current.type,
                        language=language.value,
                        context=_context_snippet(source, current),
                    )
                )
        for i in range(current.child_count - 1, -1, -1):
            stack.append(current.children[i])
    return results


# ─── Grep with line-level results ─────────────────────────────────────
def _grep_file_lines(
    content: str,
    path: str,
    pattern: str,
    context_lines: int = 2,
) -> list[GrepMatch]:
    """Return lines containing *pattern* with surrounding context.

    Accepts already-read *content* so the caller controls I/O.
    Matching is case-insensitive to mirror symbol resolution behavior.
    """
    lines = content.splitlines()
    matches: list[GrepMatch] = []
    needle = pattern.casefold()

    for i, line in enumerate(lines):
        if needle not in line.casefold():
            continue
        start = max(0, i - context_lines)
        end = min(len(lines), i + context_lines + 1)
        snippet = "\n".join(f"{j + 1}: {lines[j]}" for j in range(start, end))
        matches.append(GrepMatch(file=path, line=i + 1, text=snippet))
    return matches


# ─── CodeTools ─────────────────────────────────────────────────────────
@dataclass
class CodeTools:
    """
    Searches for symbol definitions using tree-sitter with grep fallback.

    Uses fsspec to find candidate files containing the symbol, then parses
    them with tree-sitter to locate precise definitions. When no tree-sitter
    definition is found, returns the raw grep matches instead.

    File contents are read **once** per search and reused across the grep
    pre-filter, tree-sitter parsing, and fallback grep phases to minimise
    I/O and avoid repeated calls into the filesystem backend.
    """

    fs: AbstractFileSystem
    root: str = "/"
    max_context_lines: int = 15
    grep_context_lines: int = 2
    max_walk_depth: int = _MAX_WALK_DEPTH
    max_files: int = _MAX_FILES_PER_WALK
    _parse_cache: dict[str, _ParsedFile] = field(
        default_factory=dict, init=False, repr=False
    )
    _resolution_cache: dict[tuple[str, str], _CacheEntry] = field(
        default_factory=dict, init=False, repr=False
    )
    _parser_cache: dict[Language, Parser] = field(
        default_factory=dict, init=False, repr=False
    )

    def _norm(self, path: str) -> str:
        result = norm_unicode(path)
        if result is None:
            raise ValueError(f"Cannot normalize path: {path!r}")
        return result

    def _resolve_root(self, path: str = "") -> str:
        """Turn *path* into an absolute search root.

        Handles three cases:
        - Empty *path*        → self.root
        - Absolute *path*     → path as-is (ignores self.root)
        - Relative *path*     → self.root / path

        Always normalises away double slashes and trailing slashes
        (except for bare "/").
        """
        raw = self.root if not path else path
        raw = normalize_slashes(raw)

        if not raw.startswith("/"):
            raw = f"{self.root.rstrip('/')}/{raw.lstrip('/')}"

        cleaned = "/" + "/".join(part for part in raw.split("/") if part)
        return self._norm(cleaned or "/")

    def _normalize_cache_path(self, path: str) -> str:
        return self._norm(normalize_slashes(path))

    def _get_parser(self, lang: Language) -> Parser:
        parser = self._parser_cache.get(lang)
        if parser is None:
            parser = get_parser(lang.value)
            self._parser_cache[lang] = parser
        return parser

    # ── Public ────────────────────────────────────────────────────────
    def search_definition(
        self,
        symbol: str,
        *,
        path: str = "",
        language_filter: Optional[Language] = None,
        max_results: int = 50,
        max_grep_results: int = 20,
    ) -> SearchResult:
        """Search for the definition of *symbol*.

        Reads each candidate source file at most **once**, then reuses the
        content for tree-sitter parsing and grep fallback.
        """
        search_symbol = symbol.rsplit(".", 1)[-1]
        search_root = self._resolve_root(path)
        needle = search_symbol.casefold()

        candidate_contents: list[tuple[str, Language, str]] = []
        for fpath in _iter_all_files(
            self.fs,
            search_root,
            max_depth=self.max_walk_depth,
            max_files=self.max_files,
        ):
            lang = detect_language(fpath)
            if lang is None:
                continue
            if language_filter is not None and lang != language_filter:
                continue

            text = _read_text_safe(self.fs, fpath)
            if text is not None and needle in text.casefold():
                candidate_contents.append((fpath, lang, text))

        if not candidate_contents:
            return SearchResult(
                symbol=symbol,
                definitions=[],
                grep_matches=[],
                is_fallback=False,
            )

        # ── Phase 1: tree-sitter definitions ──
        all_defs: list[DefinitionResult] = []
        for fpath, lang, text in candidate_contents:
            content_bytes = text.encode("utf-8")
            defs = self._search_file_from_content(fpath, symbol, lang, content_bytes)
            all_defs.extend(defs)
            if len(all_defs) >= max_results:
                all_defs = all_defs[:max_results]
                break

        all_defs.sort(key=lambda r: (r.file, r.line, r.column))
        if all_defs:
            return SearchResult(
                symbol=symbol,
                definitions=all_defs,
                grep_matches=[],
                is_fallback=False,
            )

        # ── Phase 2: fallback to grep matches (no extra I/O) ──
        all_grep: list[GrepMatch] = []
        for fpath, lang, text in candidate_contents:
            if language_filter is not None and lang != language_filter:
                continue
            matches = _grep_file_lines(
                text, fpath, search_symbol, context_lines=self.grep_context_lines
            )
            all_grep.extend(matches)
            if len(all_grep) >= max_grep_results:
                all_grep = all_grep[:max_grep_results]
                break

        return SearchResult(
            symbol=symbol,
            definitions=[],
            grep_matches=all_grep,
            is_fallback=True,
        )

    def list_symbols(
        self,
        path: str = "",
        *,
        language_filter: Optional[Language] = None,
        node_type_filter: Optional[str] = None,
    ) -> list[SymbolEntry]:
        """Return all symbol definitions under *path*."""
        search_root = self._resolve_root(path)

        all_symbols: list[SymbolEntry] = []
        for fpath in _iter_all_files(
            self.fs,
            search_root,
            max_depth=self.max_walk_depth,
            max_files=self.max_files,
        ):
            lang = detect_language(fpath)
            if lang is None:
                continue
            if language_filter is not None and lang != language_filter:
                continue
            content = self._read_file(fpath)
            if content is None:
                continue
            parsed = self._parse_file(fpath, content, lang)
            if parsed is None:
                continue
            specs = _specs_for(lang)
            entries = _extract_all_definitions(
                parsed.tree.root_node, parsed.source, specs, fpath, lang
            )
            if node_type_filter:
                entries = [e for e in entries if e.node_type == node_type_filter]
            all_symbols.extend(entries)

        all_symbols.sort(key=lambda s: (s.file, s.line, s.name))
        return all_symbols

    def clear_cache(self) -> None:
        self._parse_cache.clear()
        self._resolution_cache.clear()
        self._parser_cache.clear()

    def invalidate_file(self, path: str) -> None:
        normalized = self._normalize_cache_path(path)
        self._parse_cache.pop(normalized, None)
        for k in [k for k in self._resolution_cache if k[1] == normalized]:
            self._resolution_cache.pop(k, None)

    # ── Internal ──────────────────────────────────────────────────────
    def _read_file(self, path: str) -> Optional[bytes]:
        """Read file content as bytes."""
        text = _read_text_safe(self.fs, path)
        if text is None:
            return None
        return text.encode("utf-8")

    def _content_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()[:16]

    def _parse_file(
        self, path: str, content: bytes, lang: Language
    ) -> Optional[_ParsedFile]:
        normalized_path = self._normalize_cache_path(path)
        chash = self._content_hash(content)
        cached = self._parse_cache.get(normalized_path)
        if cached is not None and cached.content_hash == chash:
            return cached
        try:
            parser = self._get_parser(lang)
            tree = parser.parse(content)
        except Exception:
            logger.debug(
                "Failed to parse %s as %s", normalized_path, lang.value, exc_info=True
            )
            return None
        parsed = _ParsedFile(
            content_hash=chash,
            tree=tree,
            source=content,
            language=lang,
        )
        self._parse_cache[normalized_path] = parsed
        return parsed

    def _search_file_from_content(
        self,
        path: str,
        symbol: str,
        lang: Language,
        content: bytes,
    ) -> list[DefinitionResult]:
        """Search *content* (already read) for definitions of *symbol*."""
        normalized_path = self._normalize_cache_path(path)
        chash = self._content_hash(content)
        cache_key = (symbol, normalized_path)
        cached_resolution = self._resolution_cache.get(cache_key)
        if cached_resolution is not None and cached_resolution.content_hash == chash:
            return cached_resolution.results

        parsed = self._parse_file(normalized_path, content, lang)
        if parsed is None:
            return []
        specs = _specs_for(lang)
        if not specs:
            return []

        results = _walk_for_definitions(
            node=parsed.tree.root_node,
            source=parsed.source,
            specs=specs,
            symbol=symbol,
            file_path=normalized_path,
            language=lang,
        )
        self._resolution_cache[cache_key] = _CacheEntry(
            content_hash=chash,
            results=results,
        )
        return results


# ─── Agent tool factory ──────────────────────────────────────────────
def _parse_language(language: str) -> Optional[Language] | dict[str, Any]:
    """Parse language string, returning an error dict on failure."""
    if not language:
        return None
    try:
        return Language(language)
    except ValueError:
        return {
            "error": (
                f"Unknown language '{language}'. Supported: "
                + ", ".join(lang.value for lang in Language)
            ),
            "result": [],
            "total_items": 0,
        }


def code_tools(
    fs: AbstractFileSystem,
    root: str = "/",
) -> list:
    """Create code search tools for an LLM agent."""


    tools = CodeTools(fs=fs, root=root)

    def search_def(
        symbol: str,
        path: str = "",
        language: str = "",
    ) -> dict[str, Any]:
        """Search for the definition of a symbol in the codebase."""
        lang_filter = _parse_language(language)
        if isinstance(lang_filter, dict):
            return lang_filter
        return tools.search_definition(
            symbol=symbol,
            path=path,
            language_filter=lang_filter,
        ).to_dict()

    def list_symbols(
        path: str = "",
        language: str = "",
        node_type: str = "",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """List all symbol definitions found under a path."""
        lang_filter = _parse_language(language)
        if isinstance(lang_filter, dict):
            return lang_filter
        symbols = tools.list_symbols(
            path,
            language_filter=lang_filter,
            node_type_filter=node_type or None,
        )
        total = len(symbols)
        resolved_limit = limit if limit is not None else 300
        page = symbols[offset : offset + resolved_limit]
        return {
            "result": [
                {
                    "name": s.name,
                    "file": s.file,
                    "line": s.line,
                    "end_line": s.end_line,
                    "node_type": s.node_type,
                    "language": s.language,
                }
                for s in page
            ],
            "offset": offset,
            "total_items": total,
            "limit": resolved_limit,
        }

    return [search_def, list_symbols]
