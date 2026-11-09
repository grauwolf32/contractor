"""Fixed Tree-sitter language and structural-definition tables for code-analysis@1."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType

from tree_sitter import Node, Parser

from contractor_runtime.toolsets.common.source_languages import EXTENSION_LANGUAGES, Language
from contractor_runtime.toolsets.common.source_names import extract_node_name

MAX_SYMBOL_NAME_CHARS = 256
# The C parser holds the GIL. A callback yields between bounded input chunks
# for large files; parsing a bytes object in a thread alone cannot do that.
COOPERATIVE_PARSE_THRESHOLD = 64 << 10
PARSE_CHUNK_BYTES = 8 << 10


# Reviewed extension table for Trailmark 0.5.0 auto-detection plus parser-walk
# headers. Its parser walk matches these suffixes case-sensitively. Its C and
# Objective-C parsers both walk .h after a source with one of the
# GRAPH_HEADER_PARSER_EXTENSIONS activates them, always with their own
# grammar. An isolated .h does not activate either parser.
GRAPH_EXTENSION_LANGUAGES = MappingProxyType(
    {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".php": "php",
        ".rb": "ruby",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".hxx": "cpp",
        ".cs": "c_sharp",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".sol": "solidity",
        ".cairo": "cairo",
        ".circom": "circom",
        ".hs": "haskell",
        ".erl": "erlang",
        ".masm": "masm",
        ".swift": "swift",
        ".m": "objc",
        ".mm": "objc",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".dart": "dart",
        ".move": "move",
        ".tact": "tact",
        ".fc": "func",
        ".func": "func",
        ".sw": "sway",
        ".rego": "rego",
        ".proto": "proto",
        ".thrift": "thrift",
        ".graphql": "graphql",
        ".gql": "graphql",
        ".sql": "sql",
    }
)

GRAPH_HEADER_PARSER_EXTENSIONS = frozenset({".c", ".m", ".mm"})

# Extensions recognized by the pinned graph engine but not by the shallow v1
# surface. They are reported as unsupported instead of being mistaken for
# ordinary non-source files.
GRAPH_ONLY_SOURCE_EXTENSIONS = frozenset(GRAPH_EXTENSION_LANGUAGES) - frozenset(EXTENSION_LANGUAGES)

# Reviewed copy of the pinned Trailmark 0.5.0 directory-walk exclusions
# (``trailmark.parsers._common.should_skip_dir``): these exact names plus every
# directory whose name starts with a dot. Trailmark never parses a file below
# one of them, so the graph mirror neither copies nor budgets such files. Like
# the extension table, it lives here instead of importing Trailmark internals
# into the Runtime process; a Trailmark upgrade must re-review both.
GRAPH_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".eggs",
        "vendor",
    }
)


@dataclass(frozen=True, slots=True)
class NodeSpec:
    node_type: str
    name_field: str


NODE_SPECS = MappingProxyType(
    {
        Language.PYTHON: (
            NodeSpec("function_definition", "name"),
            NodeSpec("async_function_definition", "name"),
            NodeSpec("class_definition", "name"),
            NodeSpec("decorated_definition", "definition"),
        ),
        Language.JAVASCRIPT: (
            NodeSpec("function_declaration", "name"),
            NodeSpec("function_expression", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("class_expression", "name"),
            NodeSpec("method_definition", "name"),
            NodeSpec("variable_declarator", "name"),
            NodeSpec("export_statement", ""),
            NodeSpec("lexical_declaration", ""),
            NodeSpec("variable_declaration", ""),
        ),
        Language.TYPESCRIPT: (
            NodeSpec("function_declaration", "name"),
            NodeSpec("function_expression", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("class_expression", "name"),
            NodeSpec("method_definition", "name"),
            NodeSpec("variable_declarator", "name"),
            NodeSpec("export_statement", ""),
            NodeSpec("lexical_declaration", ""),
            NodeSpec("variable_declaration", ""),
            NodeSpec("interface_declaration", "name"),
            NodeSpec("type_alias_declaration", "name"),
            NodeSpec("enum_declaration", "name"),
            NodeSpec("abstract_class_declaration", "name"),
        ),
        Language.TSX: (
            NodeSpec("function_declaration", "name"),
            NodeSpec("function_expression", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("class_expression", "name"),
            NodeSpec("method_definition", "name"),
            NodeSpec("variable_declarator", "name"),
            NodeSpec("export_statement", ""),
            NodeSpec("lexical_declaration", ""),
            NodeSpec("variable_declaration", ""),
            NodeSpec("interface_declaration", "name"),
            NodeSpec("type_alias_declaration", "name"),
            NodeSpec("enum_declaration", "name"),
            NodeSpec("abstract_class_declaration", "name"),
        ),
        Language.GO: (
            NodeSpec("function_declaration", "name"),
            NodeSpec("method_declaration", "name"),
            NodeSpec("type_declaration", ""),
            NodeSpec("type_spec", "name"),
            NodeSpec("var_declaration", ""),
            NodeSpec("const_declaration", ""),
            NodeSpec("var_spec", "name"),
            NodeSpec("const_spec", "name"),
        ),
        Language.RUST: (
            NodeSpec("function_item", "name"),
            NodeSpec("struct_item", "name"),
            NodeSpec("enum_item", "name"),
            NodeSpec("trait_item", "name"),
            NodeSpec("impl_item", ""),
            NodeSpec("type_item", "name"),
            NodeSpec("const_item", "name"),
            NodeSpec("static_item", "name"),
            NodeSpec("macro_definition", "name"),
            NodeSpec("mod_item", "name"),
        ),
        Language.JAVA: (
            NodeSpec("method_declaration", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("interface_declaration", "name"),
            NodeSpec("constructor_declaration", "name"),
            NodeSpec("enum_declaration", "name"),
            NodeSpec("annotation_type_declaration", "name"),
            NodeSpec("record_declaration", "name"),
        ),
        Language.KOTLIN: (
            NodeSpec("function_declaration", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("object_declaration", "name"),
            NodeSpec("interface_declaration", "name"),
            NodeSpec("secondary_constructor", ""),
            NodeSpec("companion_object", "name"),
            NodeSpec("property_declaration", ""),
        ),
        Language.C: (
            NodeSpec("function_definition", "declarator"),
            NodeSpec("declaration", "declarator"),
            NodeSpec("struct_specifier", "name"),
            NodeSpec("union_specifier", "name"),
            NodeSpec("enum_specifier", "name"),
            NodeSpec("type_definition", "declarator"),
        ),
        Language.CPP: (
            NodeSpec("function_definition", "declarator"),
            NodeSpec("declaration", "declarator"),
            NodeSpec("class_specifier", "name"),
            NodeSpec("struct_specifier", "name"),
            NodeSpec("union_specifier", "name"),
            NodeSpec("enum_specifier", "name"),
            NodeSpec("template_declaration", ""),
            NodeSpec("namespace_definition", "name"),
            NodeSpec("type_definition", "declarator"),
            NodeSpec("alias_declaration", "name"),
        ),
        Language.C_SHARP: (
            NodeSpec("method_declaration", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("interface_declaration", "name"),
            NodeSpec("struct_declaration", "name"),
            NodeSpec("enum_declaration", "name"),
            NodeSpec("constructor_declaration", "name"),
            NodeSpec("property_declaration", "name"),
            NodeSpec("delegate_declaration", "name"),
            NodeSpec("record_declaration", "name"),
            NodeSpec("namespace_declaration", "name"),
            NodeSpec("local_function_statement", "name"),
        ),
        Language.RUBY: (
            NodeSpec("method", "name"),
            NodeSpec("singleton_method", "name"),
            NodeSpec("class", "name"),
            NodeSpec("module", "name"),
            NodeSpec("do_block", ""),
        ),
        Language.PHP: (
            NodeSpec("function_definition", "name"),
            NodeSpec("method_declaration", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("interface_declaration", "name"),
            NodeSpec("trait_declaration", "name"),
            NodeSpec("enum_declaration", "name"),
        ),
        Language.SCALA: (
            NodeSpec("function_definition", "name"),
            NodeSpec("class_definition", "name"),
            NodeSpec("object_definition", "name"),
            NodeSpec("trait_definition", "name"),
            NodeSpec("val_definition", "pattern"),
            NodeSpec("var_definition", "pattern"),
            NodeSpec("type_definition", "name"),
        ),
        Language.SWIFT: (
            NodeSpec("function_declaration", "name"),
            NodeSpec("class_declaration", "name"),
            NodeSpec("struct_declaration", "name"),
            NodeSpec("enum_declaration", "name"),
            NodeSpec("protocol_declaration", "name"),
            NodeSpec("extension_declaration", ""),
            NodeSpec("typealias_declaration", "name"),
            NodeSpec("init_declaration", ""),
            NodeSpec("subscript_declaration", ""),
            NodeSpec("computed_property", ""),
        ),
        Language.LUA: (
            NodeSpec("function_declaration", "name"),
            NodeSpec("local_function_declaration", "name"),
            NodeSpec("function_definition", ""),
            NodeSpec("assignment_statement", ""),
            NodeSpec("local_variable_declaration", ""),
        ),
        Language.ELIXIR: (NodeSpec("call", ""),),
        Language.HASKELL: (
            NodeSpec("function", ""),
            NodeSpec("signature", ""),
            NodeSpec("data_declaration", ""),
            NodeSpec("newtype_declaration", ""),
            NodeSpec("type_synonym_declaration", ""),
            NodeSpec("class_declaration", ""),
            NodeSpec("instance_declaration", ""),
        ),
        Language.BASH: (NodeSpec("function_definition", "name"),),
    }
)


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    name: str
    path: str
    line: int
    end_line: int
    column: int
    node_type: str
    language: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class ParseResult:
    symbols: tuple[SymbolRecord, ...]
    parse_error: bool
    symbol_limit_reached: bool
    long_names_skipped: bool = False


def graph_only_source(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in GRAPH_ONLY_SOURCE_EXTENSIONS


def graph_walk_excluded(path: str) -> bool:
    """Return whether the pinned graph engine's directory walk skips ``path``."""

    return any(
        part in GRAPH_EXCLUDED_DIRECTORIES or part.startswith(".")
        for part in PurePosixPath(path).parts[:-1]
    )


def parse_symbols(
    parser: Parser,
    source: bytes,
    path: str,
    language: Language,
    max_symbols: int,
) -> ParseResult:
    if len(source) < COOPERATIVE_PARSE_THRESHOLD:
        tree = parser.parse(source)
    else:

        def read_chunk(byte_offset: int, _position: object) -> bytes:
            time.sleep(0)  # Release the GIL for heartbeats on the event loop.
            return source[byte_offset : byte_offset + PARSE_CHUNK_BYTES]

        tree = parser.parse(read_chunk)
    root = tree.root_node
    spec_map = {spec.node_type: spec for spec in NODE_SPECS[language]}
    stack: list[Node] = [root]
    symbols: list[SymbolRecord] = []
    symbol_limit_reached = False
    long_names_skipped = False
    while stack:
        current = stack.pop()
        spec = spec_map.get(current.type)
        if spec is not None:
            name = extract_node_name(current, source, spec.name_field)
            if name:
                if len(name) > MAX_SYMBOL_NAME_CHARS:
                    long_names_skipped = True
                elif len(symbols) >= max_symbols:
                    symbol_limit_reached = True
                    break
                else:
                    symbols.append(
                        SymbolRecord(
                            name=name,
                            path=path,
                            line=current.start_point[0] + 1,
                            end_line=current.end_point[0] + 1,
                            column=current.start_point[1],
                            node_type=spec.node_type,
                            language=language.value,
                            start_byte=current.start_byte,
                            end_byte=current.end_byte,
                        )
                    )
        for index in range(current.child_count - 1, -1, -1):
            stack.append(current.children[index])
    return ParseResult(tuple(symbols), root.has_error, symbol_limit_reached, long_names_skipped)
