"""Closed, bounded schema declarations shared by discovery and ADK tools."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import jcs
from jsonschema import Draft202012Validator

from contractor_runtime.toolsets.mcp.errors import MCPError

MAX_SCHEMA_BYTES = 64 * 1024
MAX_DECLARATION_BYTES = 256 * 1024
MAX_SCHEMA_DEPTH = 16
MAX_SCHEMA_NODES = 2048
SCHEMA_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "oneOf",
        "allOf",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "title",
        "description",
        "default",
    }
)


def bounded_json(value: Any, limit: int, *, code: str) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    except (TypeError, ValueError, RecursionError, UnicodeError):
        raise MCPError(code) from None
    if len(encoded) > limit:
        raise MCPError(code)
    return encoded


def validate_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise MCPError("unsupported_schema")
    bounded_json(schema, MAX_SCHEMA_BYTES, code="schema_too_large")
    pending = [(schema, 0)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if nodes > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
            raise MCPError("schema_too_complex")
        if isinstance(node, dict):
            pending.extend((value, depth + 1) for value in node.values())
        elif isinstance(node, list):
            pending.extend((value, depth + 1) for value in node)
    schemas = [schema]
    while schemas:
        node = schemas.pop()
        if isinstance(node, bool):
            continue
        if not isinstance(node, dict) or set(node) - SCHEMA_KEYWORDS:
            raise MCPError("unsupported_schema")
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            raise MCPError("unsupported_schema")
        schemas.extend(properties.values())
        for keyword in ("items", "additionalProperties"):
            if keyword in node:
                schemas.append(node[keyword])
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in node:
                if not isinstance(node[keyword], list):
                    raise MCPError("unsupported_schema")
                schemas.extend(node[keyword])
    try:
        Draft202012Validator.check_schema(schema)
    except Exception:
        raise MCPError("unsupported_schema") from None
    return copy.deepcopy(schema)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str = field(repr=False)
    input_schema: dict[str, Any] = field(repr=False)
    output_schema: dict[str, Any] | None = field(default=None, repr=False)

    def declaration(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.output_schema is not None:
            result["outputSchema"] = self.output_schema
        return copy.deepcopy(result)


def declarations_digest(definitions: dict[str, ToolDefinition]) -> str:
    values = [definitions[name].declaration() for name in sorted(definitions)]
    bounded_json(values, MAX_DECLARATION_BYTES, code="declarations_too_large")
    try:
        return "sha256:" + hashlib.sha256(jcs.canonicalize(values)).hexdigest()
    except Exception:
        raise MCPError("unsupported_schema") from None
