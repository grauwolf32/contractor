from __future__ import annotations
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Optional
from tree_sitter import Node, Tree
from tree_sitter_language_pack import get_parser
from fsspec import AbstractFileSystem

logger = logging.getLogger(__name__)

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

    def to_dict(self) -> dict:
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

# ─── Core helpers ─────────────────────────────────────────────────────
def _extract_all_definitions(
    node: Node,
    source: bytes,
    specs: list[_NodeSpec],
    file_path: str,
    language: Language,
) -> list[SymbolEntry]:
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
    if node.type == "call" and node.child_count >= 2:
        keyword_node = node.children[0]
        keyword = source[keyword_node.start_byte:keyword_node.end_byte].decode(
            "utf-8", errors="replace"
        )
        if keyword in ("def", "defp", "defmodule", "defmacro", "defimpl", "defprotocol"):
            args_node = node.children[1]
            if args_node.child_count > 0:
                first = args_node.children[0]
                name_text = source[first.start_byte:first.end_byte].decode(
                    "utf-8", errors="replace"
                )
                return name_text.split("(")[0].strip()
        return None

    if node.type in ("function", "signature") and node.child_count > 0:
        first = node.children[0]
        return source[first.start_byte:first.end_byte].decode(
            "utf-8", errors="replace"
        ).strip()

    if node.type in (
        "data_declaration",
        "newtype_declaration",
        "type_synonym_declaration",
        "class_declaration",
        "instance_declaration",
    ):
        for child in node.children:
            if child.type == "name" or child.is_named:
                text = source[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                ).strip()
                if text and text not in (
                    "data", "newtype", "type", "class", "instance", "where"
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
        name_child = node.child_by_field_name("name")
        if name_child:
            return (
                source[name_child.start_byte:name_child.end_byte]
                .decode("utf-8", errors="replace")
                .strip()
            )
        return None

    if node.type == "property_declaration":
        for child in node.children:
            if child.type in ("simple_identifier", "identifier"):
                return source[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                ).strip()
        return None

    if node.type == "secondary_constructor":
        return "constructor"

    if node.type == "extension_declaration":
        for child in node.children:
            if child.type in ("user_type", "type_identifier"):
                return source[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                ).strip()
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
            return source[var_list.start_byte:var_list.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
        return None

    if node.type == "local_variable_declaration":
        name_list = node.child_by_field_name("name")
        if name_list is not None:
            return source[name_list.start_byte:name_list.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
        return None

    if node.type in ("impl_item", "template_declaration"):
        text = source[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        )
        first_line = text.splitlines()[0].strip()
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
        if child.type in ("function_definition", "class_definition",
                          "async_function_definition"):
            return _extract_name_from_node(child, source)
        text = source[child.start_byte:child.end_byte].decode(
            "utf-8", errors="replace"
        )
        ident = text.split("(")[0].split("<")[0].split("[")[0].strip()
        if ident:
            return ident

    return None

def _context_snippet(source: bytes, node: Node, max_lines: int = 15) -> str:
    text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
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

# ─── Grep with line-level results ─────────────────────────────────────
def _grep_file_lines(
    fs: AbstractFileSystem,
    path: str,
    pattern: str,
    context_lines: int = 2,
) -> list[GrepMatch]:
    """Return lines containing `pattern` with surrounding context as plain text."""
    try:
        raw = fs.cat_file(path)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    except Exception:
        return []
    lines = text.splitlines()
    matches: list[GrepMatch] = []
    seen_lines: set[int] = set()
    for i, line in enumerate(lines):
        if pattern not in line:
            continue
        if i in seen_lines:
            continue
        start = max(0, i - context_lines)
        end = min(len(lines), i + context_lines + 1)
        snippet = "\n".join(f"{j + 1}: {lines[j]}" for j in range(start, end))
        matches.append(GrepMatch(file=path, line=i + 1, text=snippet))
        seen_lines.add(i)
    return matches

def _check_file_for_pattern(
    fs: AbstractFileSystem,
    fpath: str,
    pattern: str,
) -> Optional[str]:
    """
    Read `fpath` and return its path if it contains `pattern`, else None.
    Designed to be called from a thread pool worker.
    """
    try:
        raw = fs.cat_file(fpath)
        text = (
            raw.decode("utf-8", errors="replace")
            if isinstance(raw, bytes)
            else raw
        )
        if pattern in text:
            return fpath
    except Exception:
        logger.debug("Could not read %s, skipping", fpath, exc_info=True)
    return None

# ─── CodeTools ─────────────────────────────────────────────────────────
@dataclass
class CodeTools:
    """
    Searches for symbol definitions using tree-sitter with grep fallback.
    Uses fsspec to find candidate files containing the symbol, then parses
    them with tree-sitter to locate precise definitions. When no tree-sitter
    definition is found, returns the raw grep matches instead.
    """
    fs: AbstractFileSystem
    root: str = "/"
    max_workers: int = 8
    max_context_lines: int = 15
    grep_context_lines: int = 2
    _parse_cache: dict[str, _ParsedFile] = field(
        default_factory=dict, init=False, repr=False
    )
    _resolution_cache: dict[tuple[str, str], _CacheEntry] = field(
        default_factory=dict, init=False, repr=False
    )

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
        resolved.
        """
        search_symbol = symbol.rsplit(".", 1)[-1]
        candidate_files = self._grep_files(search_symbol, path=path)
        if not candidate_files:
            return SearchResult(
                symbol=symbol,
                definitions=[],
                grep_matches=[],
                is_fallback=False,
            )

        # Phase 1: tree-sitter definitions
        all_defs: list[DefinitionResult] = []
        for fpath in candidate_files:
            lang = detect_language(fpath)
            if lang is None:
                continue
            if language_filter is not None and lang != language_filter:
                continue
            defs = self._search_file(fpath, symbol, lang)
            all_defs.extend(defs)
            if len(all_defs) >= max_results:
                all_defs = all_defs[:max_results]
                break

        all_defs.sort(key=lambda r: (r.file, r.line))
        if all_defs:
            return SearchResult(
                symbol=symbol,
                definitions=all_defs,
                grep_matches=[],
                is_fallback=False,
            )

        # Phase 2: fallback to grep matches
        all_grep: list[GrepMatch] = []
        for fpath in candidate_files:
            if language_filter is not None:
                lang = detect_language(fpath)
                if lang is None or lang != language_filter:
                    continue
            matches = _grep_file_lines(
                self.fs, fpath, search_symbol, context_lines=self.grep_context_lines
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
        """Return all symbol definitions under `path`."""
        search_root = (
            f"{self.root.rstrip('/')}/{path}".rstrip("/")
            if path
            else self.root.rstrip("/")
        )
        try:
            all_files = self.fs.find(search_root, detail=False)
        except FileNotFoundError:
            return []

        all_symbols: list[SymbolEntry] = []
        for fpath in all_files:
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
        return all_symbols

    def clear_cache(self) -> None:
        self._parse_cache.clear()
        self._resolution_cache.clear()

    def invalidate_file(self, path: str) -> None:
        self._parse_cache.pop(path, None)
        for k in [k for k in self._resolution_cache if k[1] == path]:
            self._resolution_cache.pop(k, None)

    # ── Internal ──────────────────────────────────────────────────────
    def _grep_files(self, pattern: str, path: str = "") -> list[str]:
        """
        Find all source files under `path` whose content contains `pattern`.
        File reads are dispatched to a thread pool for parallel I/O.
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

        # Only submit source files that tree-sitter can handle.
        candidate_paths = [f for f in all_files if detect_language(f) is not None]

        matching: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_check_file_for_pattern, self.fs, fpath, pattern): fpath
                for fpath in candidate_paths
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    matching.append(result)

        # Sort to give deterministic, stable ordering across runs.
        matching.sort()
        return matching

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
        cached = self._parse_cache.get(path)
        if cached is not None and cached.content_hash == chash:
            return cached
        try:
            parser = get_parser(lang.value)
            tree = parser.parse(content)
        except Exception:
            logger.debug(
                "Failed to parse %s as %s", path, lang.value, exc_info=True
            )
            return None
        parsed = _ParsedFile(
            content_hash=chash,
            tree=tree,
            source=content,
            language=lang,
        )
        self._parse_cache[path] = parsed
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
        cached_resolution = self._resolution_cache.get(cache_key)
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
        self._resolution_cache[cache_key] = _CacheEntry(
            content_hash=chash,
            results=results,
        )
        return results

# ─── Agent tool factory ──────────────────────────────────────────────
def _parse_language(language: str) -> Optional[Language] | dict:
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
    max_workers: int = 8,
) -> list:
    """
    Create code search tools for an LLM agent.

    Args:
        fs: An fsspec AbstractFileSystem instance pointing to the codebase.
        root: Root path within the filesystem to search from.
        max_workers: Number of parallel threads used when scanning files for
            a pattern. Higher values speed up grep on large repos at the cost
            of more concurrent I/O. Defaults to 8.

    Returns:
        List of tool functions: [search_def, list_symbols]
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
            language: Optional language filter. When set, only files of this
                language are searched. Accepted values: "python", "javascript",
                "typescript", "tsx", "go", "rust", "java", "kotlin", "c",
                "cpp", "c_sharp", "ruby", "php", "scala", "swift", "lua",
                "elixir", "haskell", "bash". Use "" to search all languages.

        Returns:
            When definitions are found (kind="definition"):
                {
                    "result": [
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
                    ],
                    "kind": "definition",
                    "total_items": 1
                }
            When no definition found but grep matches exist (kind="grep_fallback"):
                {
                    "result": [
                        {"file": "src/handler.py", "line": 10,
                         "text": "10: result = my_func(data)"}
                    ],
                    "kind": "grep_fallback",
                    "total_items": 3,
                    "note": "No tree-sitter definition found. ..."
                }
            When not found anywhere (kind="none"):
                {
                    "result": [],
                    "kind": "none",
                    "total_items": 0,
                    "note": "Symbol not found in any source file."
                }
            On invalid language:
                {"error": "Unknown language '...'. Supported: ...",
                 "result": [], "total_items": 0}
        """
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
    ) -> dict:
        """List all symbol definitions found under a path.

        Args:
            path: Directory to search, relative to the project root.
                  Use "" to search the entire project.
            language: Optional language filter (same values as search_def).
            node_type: Optional node type filter, e.g. "function_definition"
                       or "class_definition". Use "" to return all types.
            offset: Pagination offset (0-based).
            limit: Maximum number of items to return.

        Returns:
            {
                "result": [
                    {
                        "name": "my_func",
                        "file": "src/handler.py",
                        "line": 42,
                        "end_line": 58,
                        "node_type": "function_definition",
                        "language": "python"
                    },
                    ...
                ],
                "offset": 0,
                "total_items": 10,
                "limit": 300
            }
            On invalid language:
                {"error": "...", "result": [], "total_items": 0}
        """
        lang_filter = _parse_language(language)
        if isinstance(lang_filter, dict):
            return lang_filter
        symbols = tools.list_symbols(
            path, language_filter=lang_filter, node_type_filter=node_type or None
        )
        total = len(symbols)
        resolved_limit = limit if limit is not None else 300
        page = symbols[offset:offset + resolved_limit]
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