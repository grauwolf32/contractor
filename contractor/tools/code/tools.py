"""
Symbol definition search using tree-sitter.

Searches for function/class/method definitions across multiple languages
using grep for candidate file discovery and tree-sitter for precise parsing.
Falls back to grep results when no exact definition is found.

Parallelized version using concurrent.futures.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Optional

from tree_sitter import Node, Tree
from tree_sitter_language_pack import get_parser

from fsspec import AbstractFileSystem

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 8


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
    """Which node types count as definitions and how to extract the name."""

    node_type: str
    name_field: str


def _specs_for(lang: Language) -> list[_NodeSpec]:
    """Return definition node specs for a language."""
    match lang:
        case Language.PYTHON:
            return [
                _NodeSpec("function_definition", "name"),
                _NodeSpec("class_definition", "name"),
                _NodeSpec("decorated_definition", "definition"),
            ]
        case Language.JAVASCRIPT | Language.TYPESCRIPT | Language.TSX:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("method_definition", "name"),
                _NodeSpec("arrow_function", ""),
                _NodeSpec("variable_declarator", "name"),
                _NodeSpec("export_statement", ""),
            ]
        case Language.GO:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("method_declaration", "name"),
                _NodeSpec("type_declaration", ""),
                _NodeSpec("type_spec", "name"),
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
            ]
        case Language.JAVA | Language.KOTLIN:
            return [
                _NodeSpec("method_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("interface_declaration", "name"),
                _NodeSpec("constructor_declaration", "name"),
                _NodeSpec("enum_declaration", "name"),
                _NodeSpec("function_declaration", "name"),
            ]
        case Language.C:
            return [
                _NodeSpec("function_definition", "declarator"),
                _NodeSpec("declaration", "declarator"),
                _NodeSpec("struct_specifier", "name"),
                _NodeSpec("enum_specifier", "name"),
                _NodeSpec("type_definition", "declarator"),
            ]
        case Language.CPP:
            return [
                _NodeSpec("function_definition", "declarator"),
                _NodeSpec("declaration", "declarator"),
                _NodeSpec("class_specifier", "name"),
                _NodeSpec("struct_specifier", "name"),
                _NodeSpec("enum_specifier", "name"),
                _NodeSpec("template_declaration", ""),
                _NodeSpec("namespace_definition", "name"),
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
            ]
        case Language.RUBY:
            return [
                _NodeSpec("method", "name"),
                _NodeSpec("singleton_method", "name"),
                _NodeSpec("class", "name"),
                _NodeSpec("module", "name"),
            ]
        case Language.PHP:
            return [
                _NodeSpec("function_definition", "name"),
                _NodeSpec("method_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("interface_declaration", "name"),
                _NodeSpec("trait_declaration", "name"),
            ]
        case Language.SCALA:
            return [
                _NodeSpec("function_definition", "name"),
                _NodeSpec("class_definition", "name"),
                _NodeSpec("object_definition", "name"),
                _NodeSpec("trait_definition", "name"),
                _NodeSpec("val_definition", "pattern"),
            ]
        case Language.SWIFT:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("class_declaration", "name"),
                _NodeSpec("struct_declaration", "name"),
                _NodeSpec("enum_declaration", "name"),
                _NodeSpec("protocol_declaration", "name"),
            ]
        case Language.LUA:
            return [
                _NodeSpec("function_declaration", "name"),
                _NodeSpec("local_function_declaration", "name"),
                _NodeSpec("function_definition", ""),
            ]
        case Language.ELIXIR:
            return [
                _NodeSpec("call", ""),
            ]
        case Language.HASKELL:
            return [
                _NodeSpec("function", ""),
                _NodeSpec("signature", ""),
            ]
        case Language.BASH:
            return [
                _NodeSpec("function_definition", "name"),
            ]
        case _:
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
    """A precise definition found via tree-sitter."""

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
class ReferenceResult:
    """A symbol usage found in source (not a definition)."""

    symbol: str
    file: str
    line: int
    column: int
    context: str
    ref_kind: str  # "call", "import", "type_annotation", "assignment", etc.

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


# Node types considered as "usage" per language
_REF_NODE_TYPES: dict[Language, set[str]] = {
    Language.PYTHON: {
        "call",
        "attribute",
        "import_from_statement",
        "import_statement",
        "type_annotation",
    },
    Language.JAVASCRIPT: {"call_expression", "import_statement", "member_expression"},
    Language.TYPESCRIPT: {"call_expression", "import_statement", "type_reference"},
    Language.GO: {"call_expression", "selector_expression", "import_spec"},
    Language.RUST: {"call_expression", "macro_invocation", "use_declaration"},
}


@dataclass(frozen=True)
class GrepMatch:
    """A grep-level match: file path, line number, and the matching line text."""

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

    def to_dict(self) -> dict:
        """Serialize for agent tool output."""
        if self.definitions:
            return {
                "symbol": self.symbol,
                "found": True,
                "kind": "definition",
                "count": len(self.definitions),
                "results": [
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
            }
        elif self.grep_matches:
            return {
                "symbol": self.symbol,
                "found": False,
                "kind": "grep_fallback",
                "note": (
                    "No tree-sitter definition found. "
                    "Showing grep matches where the symbol appears."
                ),
                "count": len(self.grep_matches),
                "results": [
                    {
                        "file": g.file,
                        "line": g.line,
                        "text": g.text,
                    }
                    for g in self.grep_matches
                ],
            }
        else:
            return {
                "symbol": self.symbol,
                "found": False,
                "kind": "none",
                "note": "Symbol not found in any source file.",
                "count": 0,
                "results": [],
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


# ─── Thread-safe cache wrapper ────────────────────────────────────────


class _ThreadSafeCache:
    """Simple thread-safe dict wrapper with per-cache locking."""

    def __init__(self) -> None:
        self._parse_cache: dict[str, _ParsedFile] = {}
        self._resolution_cache: dict[tuple[str, str], _CacheEntry] = {}
        self._parse_lock = threading.Lock()
        self._resolution_lock = threading.Lock()

    def get_parsed(self, path: str) -> Optional[_ParsedFile]:
        with self._parse_lock:
            return self._parse_cache.get(path)

    def set_parsed(self, path: str, entry: _ParsedFile) -> None:
        with self._parse_lock:
            self._parse_cache[path] = entry

    def get_resolution(self, key: tuple[str, str]) -> Optional[_CacheEntry]:
        with self._resolution_lock:
            return self._resolution_cache.get(key)

    def set_resolution(self, key: tuple[str, str], entry: _CacheEntry) -> None:
        with self._resolution_lock:
            self._resolution_cache[key] = entry

    def clear(self) -> None:
        with self._parse_lock:
            self._parse_cache.clear()
        with self._resolution_lock:
            self._resolution_cache.clear()

    def invalidate_file(self, path: str) -> None:
        with self._parse_lock:
            self._parse_cache.pop(path, None)
        with self._resolution_lock:
            keys_to_remove = [k for k in self._resolution_cache if k[1] == path]
            for k in keys_to_remove:
                self._resolution_cache.pop(k, None)


# ─── Core helpers ─────────────────────────────────────────────────────


def _extract_all_definitions(
    node: Node,
    source: bytes,
    specs: list[_NodeSpec],
    file_path: str,
    language: Language,
) -> list[SymbolEntry]:
    """Extract all definitions from a file without filtering by name."""
    results: list[SymbolEntry] = []
    spec_types = {s.node_type for s in specs}
    stack: list[Node] = [node]

    while stack:
        current = stack.pop()
        if current.type in spec_types:
            name = _extract_name_from_node(current, source)
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


def _extract_name_from_node(node: Node, source: bytes) -> Optional[str]:
    """Best-effort extraction of the defined symbol name from a node."""

    for field_name in ("name", "declarator", "pattern", "definition"):
        child = node.child_by_field_name(field_name)
        if child is not None:
            if child.type in (
                "function_declarator",
                "pointer_declarator",
                "parenthesized_declarator",
                "reference_declarator",
            ):
                inner = _extract_name_from_node(child, source)
                if inner:
                    return inner
            if child.type in ("function_definition", "class_definition"):
                return _extract_name_from_node(child, source)
            text = source[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            )
            ident = text.split("(")[0].split("<")[0].split("[")[0].strip()
            if ident:
                return ident

    # Elixir: def/defp/defmodule are `call` nodes
    if node.type == "call" and node.child_count >= 2:
        keyword_node = node.children[0]
        keyword = source[keyword_node.start_byte : keyword_node.end_byte].decode(
            "utf-8", errors="replace"
        )
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
                name_text = source[first.start_byte : first.end_byte].decode(
                    "utf-8", errors="replace"
                )
                return name_text.split("(")[0].strip()

    # Haskell
    if node.type in ("function", "signature") and node.child_count > 0:
        first = node.children[0]
        text = source[first.start_byte : first.end_byte].decode(
            "utf-8", errors="replace"
        )
        return text.strip()

    # JS/TS variable_declarator
    if node.type == "variable_declarator":
        name_child = node.child_by_field_name("name")
        if name_child:
            return (
                source[name_child.start_byte : name_child.end_byte]
                .decode("utf-8", errors="replace")
                .strip()
            )

    # export_statement
    if node.type == "export_statement":
        for child in node.children:
            if child.type in (
                "function_declaration",
                "class_declaration",
                "lexical_declaration",
                "variable_declaration",
            ):
                return _extract_name_from_node(child, source)
        return None

    # lexical_declaration / variable_declaration
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                return _extract_name_from_node(child, source)

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
    spec_types = {s.node_type for s in specs}

    stack: list[Node] = [node]
    while stack:
        current = stack.pop()

        if current.type in spec_types:
            name = _extract_name_from_node(current, source)
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


def _walk_for_references(
    node: Node,
    source: bytes,
    symbol: str,
    file_path: str,
    language: Language,
    ref_node_types: set[str],
) -> list[ReferenceResult]:
    results: list[ReferenceResult] = []
    stack: list[Node] = [node]

    while stack:
        current = stack.pop()

        if current.type == "identifier":
            text = source[current.start_byte : current.end_byte].decode(
                "utf-8", errors="replace"
            )
            if text == symbol:
                parent = current.parent
                ref_kind = parent.type if parent else "unknown"
                results.append(
                    ReferenceResult(
                        symbol=symbol,
                        file=file_path,
                        line=current.start_point[0] + 1,
                        column=current.start_point[1],
                        context=_context_snippet(
                            source, parent or current, max_lines=5
                        ),
                        ref_kind=ref_kind,
                    )
                )

        for i in range(current.child_count - 1, -1, -1):
            stack.append(current.children[i])

    return results


# ─── Grep with line-level results ─────────────────────────────────────


def _grep_file_lines(
    fs: AbstractFileSystem,
    path: str,
    pattern: str,
    context_lines: int = 2,
) -> list[GrepMatch]:
    """Read a file and return all lines containing `pattern` with surrounding context."""
    try:
        raw = fs.cat_file(path)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    except Exception:
        return []

    lines = text.splitlines()
    matches: list[GrepMatch] = []
    seen_lines: set[int] = set()

    for i, line in enumerate(lines):
        if pattern in line:
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            snippet_lines = []
            for j in range(start, end):
                prefix = ">>>" if j == i else "   "
                snippet_lines.append(f"{prefix} {j + 1:>5} | {lines[j]}")
            if i not in seen_lines:
                matches.append(
                    GrepMatch(
                        file=path,
                        line=i + 1,
                        text="\n".join(snippet_lines),
                    )
                )
                seen_lines.add(i)

    return matches


# ─── Parallel helpers ─────────────────────────────────────────────────


def _check_file_contains(
    fs: AbstractFileSystem, fpath: str, pattern: str
) -> Optional[str]:
    """
    Check if a single file contains `pattern`.
    Returns the file path if found, else None.
    """
    try:
        raw = fs.cat_file(fpath)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if pattern in text:
            return fpath
    except Exception:
        logger.debug("Could not read %s, skipping", fpath, exc_info=True)
    return None


# ─── Searcher (Parallelized) ─────────────────────────────────────────


@dataclass
class CodeTools:
    """
    Searches for symbol definitions using tree-sitter with grep fallback.

    Uses fsspec to find candidate files containing the symbol, then parses
    them with tree-sitter to locate precise definitions. When no tree-sitter
    definition is found, returns the raw grep matches instead.

    All I/O-bound and CPU-bound file scanning is parallelized using
    ThreadPoolExecutor.
    """

    fs: AbstractFileSystem
    root: str = "/"
    max_context_lines: int = 15
    grep_context_lines: int = 2
    max_workers: int = _DEFAULT_MAX_WORKERS

    _cache: _ThreadSafeCache = field(
        default_factory=_ThreadSafeCache, init=False, repr=False
    )
    _executor: Optional[ThreadPoolExecutor] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="code-tools",
        )

    def shutdown(self) -> None:
        """Shutdown the internal thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    @property
    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="code-tools",
            )
        return self._executor

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
        """
        Search for the definition of `symbol`.

        Returns tree-sitter definitions when found. Falls back to grep
        matches (file, line, surrounding context) when no definition is
        resolved. File discovery and analysis run in parallel.
        """
        search_symbol = symbol.rsplit(".", 1)[-1]

        candidate_files = self._grep_files_parallel(search_symbol, path=path)
        if not candidate_files:
            return SearchResult(
                symbol=symbol,
                definitions=[],
                grep_matches=[],
                is_fallback=False,
            )

        # Phase 1: tree-sitter definitions (parallel)
        all_defs = self._search_definitions_parallel(
            candidate_files, symbol, language_filter, max_results
        )
        all_defs.sort(key=lambda r: (r.file, r.line))

        if all_defs:
            return SearchResult(
                symbol=symbol,
                definitions=all_defs,
                grep_matches=[],
                is_fallback=False,
            )

        # Phase 2: fallback to grep matches (parallel)
        all_grep = self._grep_matches_parallel(
            candidate_files,
            search_symbol,
            language_filter,
            max_grep_results,
        )

        return SearchResult(
            symbol=symbol,
            definitions=[],
            grep_matches=all_grep,
            is_fallback=True,
        )

    def search_references(
        self,
        symbol: str,
        *,
        path: str = "",
        language_filter: Optional[Language] = None,
        max_results: int = 100,
    ) -> list[ReferenceResult]:
        """Find all usages of a symbol (calls, imports, type annotations). Parallelized."""
        search_symbol = symbol.rsplit(".", 1)[-1]
        candidate_files = self._grep_files_parallel(search_symbol, path=path)

        filtered: list[tuple[str, Language]] = []
        for fpath in candidate_files:
            lang = detect_language(fpath)
            if lang is None:
                continue
            if language_filter is not None and lang != language_filter:
                continue
            filtered.append((fpath, lang))

        all_refs: list[ReferenceResult] = []
        lock = threading.Lock()
        enough = threading.Event()

        def _process_refs(fpath: str, lang: Language) -> list[ReferenceResult]:
            if enough.is_set():
                return []

            content = self._read_file(fpath)
            if content is None:
                return []

            parsed = self._parse_file(fpath, content, lang)
            if parsed is None:
                return []

            ref_types = _REF_NODE_TYPES.get(lang, set())
            return _walk_for_references(
                parsed.tree.root_node,
                parsed.source,
                search_symbol,
                fpath,
                lang,
                ref_types,
            )

        futures = {
            self._pool.submit(_process_refs, fpath, lang): fpath
            for fpath, lang in filtered
        }

        for future in as_completed(futures):
            try:
                refs = future.result()
            except Exception:
                logger.debug(
                    "Error processing refs for %s",
                    futures[future],
                    exc_info=True,
                )
                continue

            with lock:
                all_refs.extend(refs)
                if len(all_refs) >= max_results:
                    all_refs = all_refs[:max_results]
                    enough.set()

        if enough.is_set():
            for f in futures:
                f.cancel()

        return all_refs

    def list_symbols(
        self,
        path: str = "",
        *,
        language_filter: Optional[Language] = None,
        node_type_filter: Optional[str] = None,
    ) -> list[SymbolEntry]:
        """
        Return all symbol definitions found under the given path.
        Useful for building a codebase map. Parallelized.
        """
        search_root = (
            f"{self.root.rstrip('/')}/{path}".rstrip("/")
            if path
            else self.root.rstrip("/")
        )
        try:
            all_files = self.fs.find(search_root, detail=False)
        except FileNotFoundError:
            return []

        filtered: list[tuple[str, Language]] = []
        for fpath in all_files:
            lang = detect_language(fpath)
            if lang is None:
                continue
            if language_filter is not None and lang != language_filter:
                continue
            filtered.append((fpath, lang))

        all_symbols: list[SymbolEntry] = []
        lock = threading.Lock()

        def _process_symbols(fpath: str, lang: Language) -> list[SymbolEntry]:
            content = self._read_file(fpath)
            if content is None:
                return []

            parsed = self._parse_file(fpath, content, lang)
            if parsed is None:
                return []

            specs = _specs_for(lang)
            entries = _extract_all_definitions(
                parsed.tree.root_node, parsed.source, specs, fpath, lang
            )

            if node_type_filter:
                entries = [e for e in entries if e.node_type == node_type_filter]

            return entries

        futures = {
            self._pool.submit(_process_symbols, fpath, lang): fpath
            for fpath, lang in filtered
        }

        for future in as_completed(futures):
            try:
                entries = future.result()
            except Exception:
                logger.debug(
                    "Error listing symbols for %s",
                    futures[future],
                    exc_info=True,
                )
                continue
            with lock:
                all_symbols.extend(entries)

        return all_symbols

    def clear_cache(self) -> None:
        self._cache.clear()

    def invalidate_file(self, path: str) -> None:
        self._cache.invalidate_file(path)

    # ── Internal (parallel) ───────────────────────────────────────────

    def _grep_files_parallel(self, pattern: str, path: str = "") -> list[str]:
        """
        Find all source files under `path` whose content contains `pattern`.
        Reads files in parallel using the thread pool.
        """
        search_root = (
            f"{self.root.rstrip('/')}/{path}".rstrip("/")
            if path
            else self.root.rstrip("/")
        )
        try:
            all_files = self.fs.find(search_root, detail=False)
        except FileNotFoundError:
            logger.warning("Search root not found: %s", search_root)
            return []

        source_files = [f for f in all_files if detect_language(f) is not None]

        matching: list[str] = []
        futures = {
            self._pool.submit(_check_file_contains, self.fs, fpath, pattern): fpath
            for fpath in source_files
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    matching.append(result)
            except Exception:
                logger.debug("Error checking %s", futures[future], exc_info=True)

        return matching

    def _search_definitions_parallel(
        self,
        candidate_files: list[str],
        symbol: str,
        language_filter: Optional[Language],
        max_results: int,
    ) -> list[DefinitionResult]:
        """Search for definitions across candidate files in parallel."""
        filtered: list[tuple[str, Language]] = []
        for fpath in candidate_files:
            lang = detect_language(fpath)
            if lang is None:
                continue
            if language_filter is not None and lang != language_filter:
                continue
            filtered.append((fpath, lang))

        all_defs: list[DefinitionResult] = []
        lock = threading.Lock()
        enough = threading.Event()

        def _process_def(fpath: str, lang: Language) -> list[DefinitionResult]:
            if enough.is_set():
                return []
            return self._search_file(fpath, symbol, lang)

        futures = {
            self._pool.submit(_process_def, fpath, lang): fpath
            for fpath, lang in filtered
        }

        for future in as_completed(futures):
            try:
                defs = future.result()
            except Exception:
                logger.debug(
                    "Error searching defs in %s",
                    futures[future],
                    exc_info=True,
                )
                continue

            if defs:
                with lock:
                    all_defs.extend(defs)
                    if len(all_defs) >= max_results:
                        all_defs = all_defs[:max_results]
                        enough.set()

        if enough.is_set():
            for f in futures:
                f.cancel()

        return all_defs

    def _grep_matches_parallel(
        self,
        candidate_files: list[str],
        pattern: str,
        language_filter: Optional[Language],
        max_grep_results: int,
    ) -> list[GrepMatch]:
        """Collect grep matches across files in parallel."""
        filtered: list[str] = []
        for fpath in candidate_files:
            if language_filter is not None:
                lang = detect_language(fpath)
                if lang is None or lang != language_filter:
                    continue
            filtered.append(fpath)

        all_grep: list[GrepMatch] = []
        lock = threading.Lock()
        enough = threading.Event()

        def _process_grep(fpath: str) -> list[GrepMatch]:
            if enough.is_set():
                return []
            return _grep_file_lines(
                self.fs, fpath, pattern, context_lines=self.grep_context_lines
            )

        futures = {self._pool.submit(_process_grep, fpath): fpath for fpath in filtered}

        for future in as_completed(futures):
            try:
                matches = future.result()
            except Exception:
                logger.debug("Error grepping %s", futures[future], exc_info=True)
                continue

            if matches:
                with lock:
                    all_grep.extend(matches)
                    if len(all_grep) >= max_grep_results:
                        all_grep = all_grep[:max_grep_results]
                        enough.set()

        if enough.is_set():
            for f in futures:
                f.cancel()

        return all_grep

    # ── Internal (single-file) ────────────────────────────────────────

    def _read_file(self, path: str) -> Optional[bytes]:
        try:
            content = self.fs.cat_file(path)
            return content if isinstance(content, bytes) else content.encode("utf-8")
        except Exception:
            logger.debug("Failed to read %s", path, exc_info=True)
            return None

    def _content_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()[:16]

    def _parse_file(
        self, path: str, content: bytes, lang: Language
    ) -> Optional[_ParsedFile]:
        chash = self._content_hash(content)
        cached = self._cache.get_parsed(path)
        if cached is not None and cached.content_hash == chash:
            return cached

        try:
            parser = get_parser(lang.value)
            tree = parser.parse(content)
        except Exception:
            logger.debug("Failed to parse %s as %s", path, lang.value, exc_info=True)
            return None

        parsed = _ParsedFile(
            content_hash=chash,
            tree=tree,
            source=content,
            language=lang,
        )
        self._cache.set_parsed(path, parsed)
        return parsed

    def _search_file(
        self,
        path: str,
        symbol: str,
        lang: Language,
    ) -> list[DefinitionResult]:
        content = self._read_file(path)
        if content is None:
            return []

        chash = self._content_hash(content)
        cache_key = (symbol, path)
        cached_resolution = self._cache.get_resolution(cache_key)
        if cached_resolution is not None and cached_resolution.content_hash == chash:
            return cached_resolution.results

        parsed = self._parse_file(path, content, lang)
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
            file_path=path,
            language=lang,
        )

        self._cache.set_resolution(
            cache_key,
            _CacheEntry(content_hash=chash, results=results),
        )

        return results


# ─── Agent tool factory ──────────────────────────────────────────────


def _parse_language(language: str, symbol: str = "") -> Optional[Language] | dict:
    """Parse language string, returning Language enum or error dict."""
    if not language:
        return None
    try:
        return Language(language)
    except ValueError:
        return {
            "symbol": symbol,
            "found": False,
            "kind": "error",
            "note": f"Unknown language '{language}'. Supported: "
            + ", ".join(lang.value for lang in Language),
            "count": 0,
            "results": [],
        }


def code_tools(
    fs: AbstractFileSystem,
    root: str = "/",
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list:
    """
    Create code search tools for an LLM agent.

    Returns a list of tool functions that search for symbol definitions,
    references, and list all symbols using tree-sitter parsing with grep
    fallback. All heavy operations are parallelized.

    Args:
        fs: An fsspec AbstractFileSystem instance pointing to the codebase.
        root: Root path within the filesystem to search from.
        max_workers: Number of parallel workers for file I/O and parsing.

    Returns:
        List of tool functions: [search_def, search_refs, list_symbols]
    """
    tools = CodeTools(fs=fs, root=root, max_workers=max_workers)

    def search_def(
        symbol: str,
        path: str = "",
        language: str = "",
    ) -> dict:
        """Search for the definition of a symbol in the codebase.

        Locates where a function, class, method, struct, trait, or other named
        symbol is defined. Uses tree-sitter parsing to find precise definitions
        across 19 supported languages. If no exact definition is found, returns
        grep-level matches showing every file and line where the symbol appears.

        Args:
            symbol: The symbol name to search for. Can be a simple name like
                "my_function" or a qualified name like "MyClass.my_method".
                For qualified names, the search matches the leaf name and uses
                the qualifier for disambiguation.
            path: Optional subdirectory to restrict the search to, relative to
                the project root. Use "" or omit to search the entire project.
                Example: "src/services" to search only in that subtree.
            language: Optional language filter. When set, only files of this
                language are searched. Use the tree-sitter language name:
                "python", "javascript", "typescript", "tsx", "go", "rust",
                "java", "kotlin", "c", "cpp", "c_sharp", "ruby", "php",
                "scala", "swift", "lua", "elixir", "haskell", "bash".
                Use "" or omit to search all languages.

        Returns:
            A dictionary with the search results:
            - If definitions are found (kind="definition"):
                {
                    "symbol": "my_func",
                    "found": true,
                    "kind": "definition",
                    "count": 1,
                    "results": [
                        {
                            "symbol": "my_func",
                            "file": "src/handler.py",
                            "line": 42,
                            "end_line": 58,
                            "column": 0,
                            "node_type": "function_definition",
                            "language": "python",
                            "context": "def my_func(request):\\n    ..."
                        }
                    ]
                }
            - If no definition found but grep matches exist (kind="grep_fallback"):
                {
                    "symbol": "my_func",
                    "found": false,
                    "kind": "grep_fallback",
                    "note": "No tree-sitter definition found. ...",
                    "count": 3,
                    "results": [
                        {
                            "file": "src/handler.py",
                            "line": 10,
                            "text": ">>>    10 | result = my_func(data)"
                        }
                    ]
                }
            - If not found anywhere (kind="none"):
                {
                    "symbol": "my_func",
                    "found": false,
                    "kind": "none",
                    "note": "Symbol not found in any source file.",
                    "count": 0,
                    "results": []
                }
        """
        lang_filter = _parse_language(language, symbol)
        if isinstance(lang_filter, dict):
            return lang_filter

        result = tools.search_definition(
            symbol=symbol,
            path=path,
            language_filter=lang_filter,
        )

        return result.to_dict()

    def search_refs(
        symbol: str,
        path: str = "",
        language: str = "",
    ) -> dict:
        """Search for all usages (references) of a symbol in the codebase.

        Finds every place where the symbol is used — function calls, imports,
        type annotations, attribute access, macro invocations, etc. Uses
        tree-sitter parsing for precise AST-level identification of references
        rather than simple text matching.

        This is the complement of search_def: while search_def finds where a
        symbol is *defined*, search_refs finds where it is *used*.

        Args:
            symbol: The symbol name to search for. Use the simple identifier
                name (e.g. "process_request", "UserModel"). For qualified
                names like "module.func", the leaf name "func" is matched.
            path: Optional subdirectory to restrict the search to, relative to
                the project root. Use "" or omit to search the entire project.
                Example: "src/handlers" to search only in that subtree.
            language: Optional language filter. When set, only files of this
                language are searched. Use the tree-sitter language name:
                "python", "javascript", "typescript", "tsx", "go", "rust",
                "java", "kotlin", "c", "cpp", "c_sharp", "ruby", "php",
                "scala", "swift", "lua", "elixir", "haskell", "bash".
                Use "" or omit to search all languages.

        Returns:
            A dictionary with all found references:
            {
                "symbol": "process_request",
                "found": true,
                "count": 5,
                "results": [
                    {
                        "file": "src/server.py",
                        "line": 87,
                        "column": 12,
                        "ref_kind": "call",
                        "context": "result = process_request(data)"
                    },
                    {
                        "file": "src/main.py",
                        "line": 3,
                        "column": 25,
                        "ref_kind": "import_from_statement",
                        "context": "from handlers import process_request"
                    }
                ]
            }

            Each result includes:
            - file: Path to the source file containing the reference.
            - line: 1-based line number of the reference.
            - column: 0-based column offset within the line.
            - ref_kind: The AST node type of the parent context, indicating
              how the symbol is used (e.g. "call", "import_from_statement",
              "type_annotation", "attribute", "member_expression").
            - context: A short code snippet (up to 5 lines) showing the
              reference in its surrounding code.
        """
        lang_filter = _parse_language(language, symbol)
        if isinstance(lang_filter, dict):
            return lang_filter

        refs = tools.search_references(symbol, path=path, language_filter=lang_filter)
        return {
            "symbol": symbol,
            "found": bool(refs),
            "count": len(refs),
            "results": [
                {
                    "file": r.file,
                    "line": r.line,
                    "column": r.column,
                    "ref_kind": r.ref_kind,
                    "context": r.context,
                }
                for r in refs
            ],
        }

    def list_symbols(
        path: str = "",
        language: str = "",
        node_type: str = "",
    ) -> dict:
        """List all symbol definitions found under a given path.

        Enumerates every function, class, method, struct, trait, module, and
        other named definitions in the specified directory tree. Useful for
        building a codebase map, understanding project structure, or finding
        all available symbols before searching for a specific one.

        Unlike search_def (which searches for a specific symbol by name),
        this tool returns *all* defined symbols without filtering by name.

        Args:
            path: Directory path to scan, relative to the project root.
                Use "" or omit to scan the entire project.
                Example: "src/models" to list symbols only in that subtree.
            language: Optional language filter. When set, only files of this
                language are scanned. Use the tree-sitter language name:
                "python", "javascript", "typescript", "tsx", "go", "rust",
                "java", "kotlin", "c", "cpp", "c_sharp", "ruby", "php",
                "scala", "swift", "lua", "elixir", "haskell", "bash".
                Use "" or omit to scan all languages.
            node_type: Optional filter by AST node type. When set, only
                symbols of this specific type are returned. Common values:
                - "function_definition" (Python functions)
                - "class_definition" (Python classes)
                - "function_declaration" (JS/TS/Go/Java functions)
                - "class_declaration" (JS/TS/Java classes)
                - "method_definition" (JS/TS methods)
                - "method_declaration" (Java/C# methods)
                - "struct_item" (Rust structs)
                - "trait_item" (Rust traits)
                Use "" or omit to return all symbol types.

        Returns:
            A dictionary listing all discovered symbols:
            {
                "path": "src/models",
                "count": 12,
                "results": [
                    {
                        "name": "UserModel",
                        "file": "src/models/user.py",
                        "line": 15,
                        "end_line": 42,
                        "node_type": "class_definition",
                        "language": "python"
                    },
                    {
                        "name": "create_user",
                        "file": "src/models/user.py",
                        "line": 45,
                        "end_line": 58,
                        "node_type": "function_definition",
                        "language": "python"
                    }
                ]
            }

            Each result includes:
            - name: The symbol's identifier name.
            - file: Path to the source file where the symbol is defined.
            - line: 1-based start line of the definition.
            - end_line: 1-based end line of the definition.
            - node_type: The tree-sitter AST node type of the definition.
            - language: The detected programming language.
        """
        lang_filter = _parse_language(language)
        if isinstance(lang_filter, dict):
            return lang_filter

        symbols = tools.list_symbols(
            path,
            language_filter=lang_filter,
            node_type_filter=node_type or None,
        )
        return {
            "path": path,
            "count": len(symbols),
            "results": [
                {
                    "name": s.name,
                    "file": s.file,
                    "line": s.line,
                    "end_line": s.end_line,
                    "node_type": s.node_type,
                    "language": s.language,
                }
                for s in symbols
            ],
        }

    return [search_def, search_refs, list_symbols]
