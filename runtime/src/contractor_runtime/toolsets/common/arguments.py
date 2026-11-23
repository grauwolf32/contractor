"""Bounded JSON representations, explicitly selected by each owning tool."""

import json
import math
from collections.abc import Callable
from typing import Any

from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.argument_diagnostics import json_type

MAX_CONTAINER_DEPTH = 32
MAX_CONTAINER_NODES = 65536


def validate_json(
    value: Any,
    *,
    field: str,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
    invalid: Callable,
) -> None:
    """Measure compact UTF-8 JSON incrementally; stop before copying oversized input."""
    size = nodes = 0

    def reject(message):
        raise invalid(field, message)

    def add(count):
        nonlocal size
        size += count
        if size > max_bytes:
            reject(f"Use at most {max_bytes} bytes of compact UTF-8 JSON.")

    def string(text):
        if len(text) > max_bytes:
            reject(f"Use at most {max_bytes} bytes of compact UTF-8 JSON.")
        add(2)
        for offset in range(0, len(text), 4096):
            add(len(json.dumps(text[offset : offset + 4096], ensure_ascii=False).encode()) - 2)

    def walk(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            reject(f"Use at most {max_nodes} JSON values.")
        if type(item) in (dict, list):
            if depth >= max_depth:
                reject(f"Use at most {max_depth} container levels.")
            add(2)
            for index, entry in enumerate(item.items() if type(item) is dict else item):
                if index:
                    add(1)
                if type(item) is dict:
                    key, entry = entry
                    if type(key) is not str:
                        reject("Use only string object keys.")
                    string(key)
                    add(1)
                walk(entry, depth + 1)
        elif type(item) is str:
            string(item)
        elif item is None:
            add(4)
        elif type(item) is bool:
            add(4 if item else 5)
        elif type(item) is int:
            if item.bit_length() > max_bytes * 4:
                reject("Use a bounded JSON number.")
            try:
                rendered = str(item)
            except ValueError:
                reject("Use a bounded JSON number.")
            add(len(rendered))
        elif type(item) is float and math.isfinite(item):
            add(len(json.dumps(item)))
        else:
            reject("Use JSON values with finite numbers and valid Unicode.")

    try:
        walk(value, 0)
    except (UnicodeError, RecursionError):
        reject("Use valid Unicode and bounded JSON nesting.")


def container_argument(
    value: Any,
    *,
    field: str,
    kinds: tuple[type, ...],
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
    invalid: Callable,
) -> tuple[Any, bool]:
    """Decode at most one layer, preserving every nested value and its type."""
    normalized = type(value) is str
    expected = " or ".join("object" if kind is dict else "array" for kind in kinds)
    error_factory = invalid
    actual = json_type(value)

    def invalid(field, message):
        return error_factory(field, message, expected_type=expected, actual_type=actual)

    if normalized:
        if len(value) > max_bytes:
            raise invalid(field, f"Use at most {max_bytes} UTF-8 representation bytes.")
        try:
            size = len(value.encode())
        except UnicodeError:
            raise invalid(field, "Use valid UTF-8 text.") from None
        if size > max_bytes:
            raise invalid(field, f"Use at most {max_bytes} UTF-8 representation bytes.")
        try:
            value = strict_json_loads(value)
        except (ValueError, UnicodeError, RecursionError):
            # Do not expose parser text, fragments or duplicate object keys.
            raise invalid(field, f"Provide one strict JSON {expected} document.") from None
    if type(value) not in kinds:
        raise invalid(field, f"Provide a JSON {expected}, natively or serialized once.")
    validate_json(
        value,
        field=field,
        max_bytes=max_bytes,
        max_depth=max_depth,
        max_nodes=max_nodes,
        invalid=invalid,
    )
    return value, normalized
