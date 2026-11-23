"""JSON decoding that rejects duplicate object keys and non-standard constants."""

from __future__ import annotations

import json
from typing import Any, NoReturn


class DuplicateJSONKey(ValueError):
    def __init__(self) -> None:
        super().__init__("duplicate JSON object key")


class NonStandardJSONConstant(ValueError):
    def __init__(self) -> None:
        super().__init__("non-standard JSON constant")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKey
        result[key] = value
    return result


def reject_json_constant(_value: str) -> NoReturn:
    raise NonStandardJSONConstant


def strict_json_loads(data: str | bytes) -> Any:
    """Decode JSON, raising a ValueError for duplicate keys, NaN or Infinity.

    Callers map ValueError, UnicodeDecodeError and RecursionError to their own
    error types; the two subclasses above distinguish the strictness checks.
    """

    return json.loads(
        data, object_pairs_hook=unique_json_object, parse_constant=reject_json_constant
    )
