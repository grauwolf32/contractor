"""Fixed Tree-sitter language and structural-definition tables for code-analysis@1."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

MAX_SYMBOL_NAME_CHARS = 256


class Language(StrEnum):
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
    C_SHARP = "c_sharp"
    RUBY = "ruby"
    PHP = "php"
    SCALA = "scala"
    SWIFT = "swift"
    LUA = "lua"
    ELIXIR = "elixir"
    HASKELL = "haskell"
    BASH = "bash"


PARSER_NAMES = MappingProxyType(
    {
        language: "csharp" if language is Language.C_SHARP else language.value
        for language in Language
    }
)

EXTENSION_LANGUAGES = MappingProxyType(
    {
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
        ".cs": Language.C_SHARP,
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
)

CPP_HEADER_CONTEXT_EXTENSIONS = frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"})

# Reviewed extension table for Trailmark 0.5.0 auto-detection plus parser-walk
# headers. Its C and Objective-C parsers both walk .h after another source
# activates them. An isolated .h does not activate either parser.
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
            NodeSpec("arrow_function", ""),
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


# C and C++ nest a declared name below pointer, reference, function, array,
# parenthesized, attributed and initializer declarators. Reference,
# parenthesized and attributed declarators hold their inner declarator in a
# child without a field name.
_C_DECLARATOR_WRAPPERS = frozenset(
    {
        "array_declarator",
        "attributed_declarator",
        "function_declarator",
        "init_declarator",
        "parenthesized_declarator",
        "pointer_declarator",
        "reference_declarator",
    }
)
# C++ name nodes whose source text is not one plain identifier.
_CPP_NAME_NODES = frozenset(
    {"destructor_name", "operator_name", "qualified_identifier", "template_function"}
)
_C_DECLARATOR_CHILDREN = (
    _C_DECLARATOR_WRAPPERS
    | _CPP_NAME_NODES
    | frozenset({"field_identifier", "identifier", "type_identifier"})
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


def detect_language(path: str, *, cpp_headers: bool = False) -> Language | None:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".h" and cpp_headers:
        return Language.CPP
    return EXTENSION_LANGUAGES.get(suffix)


def cpp_header_context(paths: Iterable[str]) -> bool:
    return any(
        PurePosixPath(path).suffix.lower() in CPP_HEADER_CONTEXT_EXTENSIONS for path in paths
    )


def graph_only_source(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in GRAPH_ONLY_SOURCE_EXTENSIONS


def graph_walk_excluded(path: str) -> bool:
    """Return whether the pinned graph engine's directory walk skips ``path``."""

    return any(
        part in GRAPH_EXCLUDED_DIRECTORIES or part.startswith(".")
        for part in PurePosixPath(path).parts[:-1]
    )


def load_parser(language: Language) -> Parser:
    return get_parser(PARSER_NAMES[language])


def probe_all_parsers() -> bool:
    try:
        for language in Language:
            tree = load_parser(language).parse(b"")
            if tree.root_node is None:
                return False
        return True
    except Exception:
        return False


def parse_symbols(
    parser: Parser,
    source: bytes,
    path: str,
    language: Language,
    max_symbols: int,
) -> ParseResult:
    tree = parser.parse(source)
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
            name = _extract_name(current, source, spec.name_field)
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


def extract_node_name(node: Node, source: bytes, preferred_field: str = "") -> str | None:
    """Extract one reviewed structural name for another Tree-sitter Toolset."""

    return _extract_name(node, source, preferred_field)


def _extract_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="strict").strip()


def _extract_field(node: Node, source: bytes, field_name: str) -> str | None:
    if not field_name:
        return None
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    if child.type in _C_DECLARATOR_WRAPPERS or child.type in _CPP_NAME_NODES:
        return _c_declarator_name(child, source)
    if child.type in {
        "async_function_definition",
        "class_definition",
        "function_definition",
    }:
        return _extract_name(child, source)
    return _clean_identifier(_extract_text(child, source))


def _extract_name(node: Node, source: bytes, preferred_field: str = "") -> str | None:
    preferred = _extract_field(node, source, preferred_field)
    if preferred:
        return preferred

    # The binding's variable_declarator supplies an arrow's name. An unnamed
    # function expression has no definition name of its own either.
    if node.type in {"arrow_function", "function_expression"}:
        return None

    if node.type == "call" and node.child_count >= 2:
        keyword = _extract_text(node.children[0], source)
        if keyword not in {"def", "defimpl", "defmacro", "defmodule", "defp", "defprotocol"}:
            return None
        arguments = node.children[1]
        if arguments.child_count:
            return _clean_identifier(_extract_text(arguments.children[0], source))
        return None

    if node.type in {"function", "signature"} and node.child_count:
        return _extract_text(node.children[0], source) or None

    if node.type in {
        "class_declaration",
        "data_declaration",
        "instance_declaration",
        "newtype_declaration",
        "type_synonym_declaration",
    }:
        for child in node.children:
            if child.type == "name" or child.is_named:
                text = _extract_text(child, source)
                if text and text not in {
                    "class",
                    "data",
                    "instance",
                    "newtype",
                    "type",
                    "where",
                }:
                    return text
        return None

    if node.type == "export_statement":
        for child in node.children:
            if child.type in {
                "abstract_class_declaration",
                "class_declaration",
                "enum_declaration",
                "function_declaration",
                "interface_declaration",
                "lexical_declaration",
                "type_alias_declaration",
                "variable_declaration",
            }:
                return _extract_name(child, source)
        return None

    if node.type in {"lexical_declaration", "variable_declaration"}:
        for child in node.children:
            if child.type == "variable_declarator":
                return _extract_name(child, source)
        return None

    if node.type == "variable_declarator":
        return _extract_field(node, source, "name")

    if node.type == "property_declaration":
        for child in node.children:
            if child.type in {"identifier", "simple_identifier"}:
                return _extract_text(child, source) or None
        return None

    if node.type == "secondary_constructor":
        return "constructor"
    if node.type == "extension_declaration":
        for child in node.children:
            if child.type in {"type_identifier", "user_type"}:
                return _extract_text(child, source) or None
        return None
    if node.type == "init_declaration":
        return "init"
    if node.type == "subscript_declaration":
        return "subscript"
    if node.type == "computed_property":
        return None

    if node.type == "assignment_statement":
        variable = node.child_by_field_name("variable")
        return _extract_text(variable, source) if variable is not None else None
    if node.type == "local_variable_declaration":
        name = node.child_by_field_name("name")
        return _extract_text(name, source) if name is not None else None

    if node.type in {"impl_item", "template_declaration"}:
        first_line = _extract_text(node, source).splitlines()[0].strip()
        return first_line[:80] or None

    for field_name in ("name", "declarator", "pattern", "definition"):
        extracted = _extract_field(node, source, field_name)
        if extracted:
            return extracted
    for child in node.children:
        if child.type in {"identifier", "simple_identifier", "type_identifier"}:
            return _extract_text(child, source) or None
    return None


def _c_declarator_name(node: Node, source: bytes) -> str | None:
    """Return the name a C or C++ declarator declares below its wrappers."""

    current: Node | None = node
    while current is not None and current.type in _C_DECLARATOR_WRAPPERS:
        inner = current.child_by_field_name("declarator")
        if inner is None:
            inner = next(
                (child for child in current.named_children if child.type in _C_DECLARATOR_CHILDREN),
                None,
            )
        current = inner
    if current is None:
        return None
    if current.type == "qualified_identifier":
        return _cpp_qualified_name(current, source)
    if current.type in {"destructor_name", "operator_name"}:
        # Keep the written text: operator()/operator[] would lose their
        # operator in _clean_identifier, and search_def's text prefilter needs
        # every bare definition name to occur in its source.
        return _extract_text(current, source) or None
    if current.type == "template_function":
        name = current.child_by_field_name("name")
        return _c_declarator_name(name, source) if name is not None else None
    return _clean_identifier(_extract_text(current, source))


def _cpp_qualified_name(node: Node, source: bytes) -> str | None:
    """Join scopes without template arguments: ``Box<T>::get`` is ``Box::get``."""

    parts: list[str] = []
    current = node
    while current.type == "qualified_identifier":
        scope = current.child_by_field_name("scope")
        name = current.child_by_field_name("name")
        if name is None:
            return None
        if scope is None:
            # A global qualifier keeps its leading "::".
            parts.append("")
        else:
            scope_name = _clean_identifier(_extract_text(scope, source))
            if scope_name is None:
                return None
            parts.append(scope_name)
        current = name
    terminal = _c_declarator_name(current, source)
    if terminal is None:
        return None
    parts.append(terminal)
    return "::".join(parts)


def _clean_identifier(value: str) -> str | None:
    identifier = value.split("(", 1)[0].split("<", 1)[0].split("[", 1)[0].strip()
    return identifier or None
