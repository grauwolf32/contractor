"""Expands api/testdata/v1alpha1/credential-field-value-cases.json into test cases.

The Go twin is internal/contracts/contractstest.ReadCredentialFieldValueCases.
Only test_shared_* tests call it: it opens a shared fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[2] / "api" / "testdata" / "v1alpha1"
_KEYS = {"description", "base", "positions", "characters", "values", "fields"}
_FIELD_KEYS = {"name", "maxBytes", "rule", "credential", "llmCredential", "runtimeSettings"}


@dataclass(frozen=True)
class FieldValueCase:
    name: str
    value: str
    valid: bool


@dataclass(frozen=True)
class CredentialFieldValueCases:
    grammar: tuple[FieldValueCase, ...]
    fields: tuple[dict[str, Any], ...]
    base: dict[str, Any]


def read_credential_field_value_cases() -> CredentialFieldValueCases:
    fixture = json.loads(
        (FIXTURES / "credential-field-value-cases.json").read_text(encoding="utf-8")
    )
    if set(fixture) != _KEYS or not fixture["values"] or not fixture["fields"]:
        raise ValueError("credential-field-value-cases.json has an unexpected shape")
    positions: dict[str, str] = fixture["positions"]
    if any(template.count("{c}") != 1 for template in positions.values()):
        raise ValueError("every position template holds {c} once")
    grammar: list[FieldValueCase] = []
    following = 0
    for characters in fixture["characters"]:
        if characters["from"] != following or characters["to"] < characters["from"]:
            raise ValueError(f"character range {characters} does not continue at {following}")
        if not set(characters["valid"]) <= set(positions):
            raise ValueError(f"character range {characters} names an unknown position")
        for code in range(characters["from"], characters["to"] + 1):
            for position in sorted(positions):
                grammar.append(
                    FieldValueCase(
                        name=f"U+{code:04X} {position}",
                        value=positions[position].replace("{c}", chr(code)),
                        valid=position in characters["valid"],
                    )
                )
        following = characters["to"] + 1
    if following != 128:
        raise ValueError("the character ranges must cover every ASCII character")
    for value in fixture["values"]:
        if set(value) != {"name", "value", "valid"} or not isinstance(value["valid"], bool):
            raise ValueError(f"{value}: a value needs a name and a verdict")
        grammar.append(FieldValueCase(value["name"], value["value"], value["valid"]))
    for field in fixture["fields"]:
        if not {"name"} < set(field) <= _FIELD_KEYS or ("maxBytes" in field) != ("rule" in field):
            raise ValueError(f"{field}: incomplete credential field-value field")
    if not fixture["base"].startswith("valid/"):
        raise ValueError("the RuntimeSettings base must be a valid/ fixture")
    base = json.loads((FIXTURES / fixture["base"]).read_text(encoding="utf-8"))
    return CredentialFieldValueCases(tuple(grammar), tuple(fixture["fields"]), base)


def field_cases(field: dict[str, Any], grammar: tuple[FieldValueCase, ...]) -> list[FieldValueCase]:
    """The grammar cases plus, for a field with a byte bound, its three bound cases."""

    cases = list(grammar)
    maximum = field.get("maxBytes")
    if maximum is not None:
        cases += [
            FieldValueCase(f"{maximum} visible bytes", "a" * maximum, True),
            FieldValueCase(f"{maximum + 1} visible bytes", "a" * (maximum + 1), False),
            FieldValueCase(
                f"{maximum} bytes with an inner space run", "a" + " " * (maximum - 2) + "b", True
            ),
        ]
    return cases


def set_pointer(document: dict[str, Any], pointer: str, value: Any) -> None:
    """Set the member pointer names, creating absent parent objects."""

    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}")
    *path, key = pointer[1:].split("/")
    current = document
    for segment in path:
        current = current.setdefault(segment, {})
        if not isinstance(current, dict):
            raise ValueError(f"JSON pointer {pointer!r} does not name an object")
    current[key] = value
