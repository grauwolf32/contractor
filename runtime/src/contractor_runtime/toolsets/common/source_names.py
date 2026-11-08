"""Shared structural name extraction from Tree-sitter source nodes."""

from tree_sitter import Node

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
        return extract_node_name(child, source)
    return _clean_identifier(_extract_text(child, source))


def extract_node_name(node: Node, source: bytes, preferred_field: str = "") -> str | None:
    preferred = _extract_field(node, source, preferred_field)
    if preferred:
        return preferred

    # An unnamed function expression has no definition name of its own; a
    # binding's variable_declarator supplies it.
    if node.type == "function_expression":
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
                return extract_node_name(child, source)
        return None

    if node.type in {"lexical_declaration", "variable_declaration"}:
        for child in node.children:
            if child.type == "variable_declarator":
                return extract_node_name(child, source)
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
