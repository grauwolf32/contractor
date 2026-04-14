"""
OpenAPI Local $ref Resolver

Resolves all local JSON references ($ref) in a component dict
using the full OpenAPI schema as the lookup root.
"""

from __future__ import annotations

import copy
from typing import Any


def resolve_refs(
    component: dict[str, Any],
    schema: dict[str, Any],
    *,
    max_depth: int = 100,
    _initial_seen: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """
    Resolve all local $ref pointers in `component` using `schema` as the root.

    Args:
        component: The dict containing $ref pointers to resolve.
        schema: The full OpenAPI schema used to look up ref targets.
        max_depth: Max recursion depth to guard against circular refs.
        _initial_seen: Set of refs already being resolved (for cycle detection).

    Returns:
        A deep copy of `component` with all $refs resolved inline.
    """
    component = copy.deepcopy(component)

    def _follow_pointer(ref: str) -> Any:
        if not ref.startswith("#/"):
            raise ValueError(f"Only local refs supported. Got: {ref!r}")
        parts = ref.lstrip("#/").split("/")
        parts = [p.replace("~1", "/").replace("~0", "~") for p in parts]
        node = schema
        for part in parts:
            if isinstance(node, dict):
                if part not in node:
                    raise KeyError(
                        f"Cannot resolve ref {ref!r}: "
                        f"key {part!r} not found in {list(node.keys())}"
                    )
                node = node[part]
            elif isinstance(node, list):
                try:
                    node = node[int(part)]
                except (ValueError, IndexError) as exc:
                    raise KeyError(
                        f"Cannot resolve ref {ref!r}: invalid index {part!r}"
                    ) from exc
            else:
                raise KeyError(
                    f"Cannot resolve ref {ref!r}: "
                    f"unexpected type {type(node).__name__} at {part!r}"
                )
        return copy.deepcopy(node)

    def _resolve(
        node: Any,
        depth: int = 0,
        seen: frozenset[str] = frozenset(),
    ) -> Any:
        if depth > max_depth:
            raise RecursionError(f"Max resolution depth ({max_depth}) exceeded")

        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                if not isinstance(ref, str) or not ref.startswith("#"):
                    return node
                if ref in seen:
                    return {"$circular_ref": ref}

                resolved = _follow_pointer(ref)

                siblings = {k: v for k, v in node.items() if k != "$ref"}
                if siblings and isinstance(resolved, dict):
                    resolved.update(siblings)

                # Add ref to seen BEFORE recursing into the resolved value
                return _resolve(resolved, depth + 1, seen | {ref})

            return {k: _resolve(v, depth + 1, seen) for k, v in node.items()}

        if isinstance(node, list):
            return [_resolve(item, depth + 1, seen) for item in node]

        return node

    return _resolve(component, seen=_initial_seen)


def resolve_local_refs(
    schema: dict[str, Any],
    *,
    max_depth: int = 100,
) -> dict[str, Any]:
    """Resolve all local $ref pointers in the entire schema."""
    schema = copy.deepcopy(schema)

    # Pre-resolve each named schema with its own self-ref already in `seen`,
    # so that self-referential schemas are detected as circular immediately.
    components_schemas = schema.get("components", {}).get("schemas", {})
    for name in list(components_schemas.keys()):
        self_ref = f"#/components/schemas/{name}"
        components_schemas[name] = resolve_refs(
            components_schemas[name],
            schema,
            max_depth=max_depth,
            _initial_seen=frozenset({self_ref}),
        )

    # Now resolve the rest of the schema (paths, etc.)
    # Schemas are already resolved, so remaining $refs to them will be inlined.
    result = resolve_refs(schema, schema, max_depth=max_depth)
    return result
