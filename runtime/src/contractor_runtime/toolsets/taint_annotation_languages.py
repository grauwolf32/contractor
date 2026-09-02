"""Tree-sitter resolution for function-like taint annotation targets."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from tree_sitter import Node, Parser

from contractor_runtime.toolsets.code_analysis_languages import Language, extract_node_name


@dataclass(frozen=True, slots=True)
class AnnotationTarget:
    name: str
    definition_line: int
    insertion_line: int
    selector_lines: tuple[int, ...]
    node_type: str


@dataclass(frozen=True, slots=True)
class AnnotationParseResult:
    targets: tuple[AnnotationTarget, ...]
    parse_error: bool


_DIRECT_NODE_FIELDS = MappingProxyType(
    {
        Language.PYTHON: {
            "function_definition": "name",
            "async_function_definition": "name",
        },
        Language.JAVASCRIPT: {
            "function_declaration": "name",
            "generator_function_declaration": "name",
            "function_expression": "name",
            "generator_function": "name",
            "method_definition": "name",
        },
        Language.TYPESCRIPT: {
            "function_declaration": "name",
            "generator_function_declaration": "name",
            "function_expression": "name",
            "generator_function": "name",
            "method_definition": "name",
        },
        Language.TSX: {
            "function_declaration": "name",
            "generator_function_declaration": "name",
            "function_expression": "name",
            "generator_function": "name",
            "method_definition": "name",
        },
        Language.GO: {
            "function_declaration": "name",
            "method_declaration": "name",
        },
        Language.RUST: {"function_item": "name"},
        Language.JAVA: {
            "method_declaration": "name",
            "constructor_declaration": "name",
        },
        Language.KOTLIN: {
            "function_declaration": "name",
            "secondary_constructor": "",
        },
        Language.C: {"function_definition": "declarator"},
        Language.CPP: {"function_definition": "declarator"},
        Language.C_SHARP: {
            "method_declaration": "name",
            "constructor_declaration": "name",
            "local_function_statement": "name",
        },
        Language.RUBY: {"method": "name", "singleton_method": "name"},
        Language.PHP: {
            "function_definition": "name",
            "method_declaration": "name",
        },
        Language.SCALA: {"function_definition": "name"},
        Language.SWIFT: {
            "function_declaration": "name",
            "init_declaration": "",
            "subscript_declaration": "",
        },
        Language.LUA: {
            "function_declaration": "name",
            "local_function_declaration": "name",
        },
        Language.ELIXIR: {"call": ""},
        Language.HASKELL: {"function": ""},
        Language.BASH: {"function_definition": "name"},
    }
)

_JAVASCRIPT_LANGUAGES = frozenset({Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX})
_CALLABLE_VALUE_TYPES = frozenset(
    {
        "arrow_function",
        "function_expression",
        "generator_function",
        "anonymous_function",
    }
)
_WRAPPER_PARENT_TYPES = MappingProxyType(
    {
        Language.PYTHON: frozenset({"decorated_definition"}),
        Language.JAVASCRIPT: frozenset(
            {"export_statement", "lexical_declaration", "variable_declaration"}
        ),
        Language.TYPESCRIPT: frozenset(
            {"export_statement", "lexical_declaration", "variable_declaration"}
        ),
        Language.TSX: frozenset(
            {"export_statement", "lexical_declaration", "variable_declaration"}
        ),
        Language.CPP: frozenset({"template_declaration"}),
        Language.PHP: frozenset({"expression_statement"}),
        Language.LUA: frozenset({"variable_declaration"}),
    }
)
_PREFIX_SIBLING_TYPES = MappingProxyType(
    {
        Language.RUST: frozenset({"attribute_item"}),
        Language.TYPESCRIPT: frozenset({"decorator"}),
        Language.TSX: frozenset({"decorator"}),
        Language.C_SHARP: frozenset({"attribute_list"}),
        Language.KOTLIN: frozenset({"annotation"}),
    }
)


def parse_annotation_targets(
    parser: Parser,
    source: bytes,
    language: Language,
) -> AnnotationParseResult:
    tree = parser.parse(source)
    root = tree.root_node
    targets: list[AnnotationTarget] = []
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        candidate = _candidate(node, source, language)
        if candidate is not None:
            name, name_node = candidate
            outer = _outer_declaration(node, language)
            outer = _syntax_prefix(outer, source, language)
            definition_line = name_node.start_point[0] + 1
            insertion_line = outer.start_point[0] + 1
            selector_lines = tuple(
                sorted(
                    {
                        definition_line,
                        node.start_point[0] + 1,
                        insertion_line,
                    }
                )
            )
            targets.append(
                AnnotationTarget(
                    name=name,
                    definition_line=definition_line,
                    insertion_line=insertion_line,
                    selector_lines=selector_lines,
                    node_type=node.type,
                )
            )
        for index in range(node.child_count - 1, -1, -1):
            stack.append(node.children[index])
    targets.sort(key=lambda item: (item.insertion_line, item.definition_line, item.name))
    return AnnotationParseResult(tuple(targets), root.has_error)


def _candidate(
    node: Node,
    source: bytes,
    language: Language,
) -> tuple[str, Node] | None:
    if language in _JAVASCRIPT_LANGUAGES and node.type == "variable_declarator":
        value = node.child_by_field_name("value")
        name_node = node.child_by_field_name("name")
        if value is None or value.type not in _CALLABLE_VALUE_TYPES or name_node is None:
            return None
        return _node_text(name_node, source), name_node

    if language is Language.PHP and node.type == "assignment_expression":
        value = node.child_by_field_name("right")
        name_node = node.child_by_field_name("left")
        if value is None or value.type not in _CALLABLE_VALUE_TYPES or name_node is None:
            return None
        return _node_text(name_node, source).removeprefix("$"), name_node

    if language is Language.LUA and node.type == "assignment_statement":
        value = node.child_by_field_name("value")
        if value is None or not _has_callable_child(value):
            return None
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        return _node_text(name_node, source), name_node

    field = _DIRECT_NODE_FIELDS[language].get(node.type)
    if field is None:
        return None
    if _is_assignment_value(node, language):
        return None
    if language is Language.ELIXIR and not _elixir_function(node, source):
        return None
    name = extract_node_name(node, source, field)
    if not name:
        return None
    name_node = node.child_by_field_name(field) if field else None
    return name, name_node or node


def _outer_declaration(node: Node, language: Language) -> Node:
    current = node
    wrappers = _WRAPPER_PARENT_TYPES.get(language, frozenset())
    while current.parent is not None and current.parent.type in wrappers:
        current = current.parent
    return current


def _syntax_prefix(node: Node, source: bytes, language: Language) -> Node:
    prefixes = _PREFIX_SIBLING_TYPES.get(language, frozenset())
    current = node
    while current.prev_named_sibling is not None:
        previous = current.prev_named_sibling
        if previous.type not in prefixes:
            break
        between = source[previous.end_byte : current.start_byte]
        if between.strip():
            break
        current = previous
    return current


def _is_assignment_value(node: Node, language: Language) -> bool:
    parent = node.parent
    if parent is None:
        return False
    if language in _JAVASCRIPT_LANGUAGES and parent.type == "variable_declarator":
        return parent.child_by_field_name("value") == node
    if language is Language.PHP and parent.type == "assignment_expression":
        return parent.child_by_field_name("right") == node
    if language is Language.LUA:
        ancestor = parent
        while ancestor is not None and ancestor.type in {
            "expression_list",
            "parenthesized_expression",
        }:
            ancestor = ancestor.parent
        return ancestor is not None and ancestor.type == "assignment_statement"
    return False


def _has_callable_child(node: Node) -> bool:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in _CALLABLE_VALUE_TYPES | {"function_definition"}:
            return True
        stack.extend(current.named_children)
    return False


def _elixir_function(node: Node, source: bytes) -> bool:
    if not node.children:
        return False
    keyword = _node_text(node.children[0], source)
    return keyword in {"def", "defp", "defmacro"}


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="strict").strip()
