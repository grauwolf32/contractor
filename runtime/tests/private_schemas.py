"""The private wire schemas under api/v1alpha1 as the Runtime tests validate them.

JSON Schema's maxLength counts characters. Where the Server and the Runtime
bound a string in UTF-8 bytes the schema also states x-maxBytes, which this
validator enforces like the Go schema tests do
(internal/contracts/fixture_index_test.go), so the schema, Go and Python reach
one verdict.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError, validators
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "api" / "v1alpha1"


def _max_bytes(
    validator: Any, bound: int, instance: Any, schema: dict[str, Any]
) -> Iterator[ValidationError]:
    del validator, schema
    if isinstance(instance, str) and len(instance.encode("utf-8")) > bound:
        yield ValidationError(f"string exceeds {bound} UTF-8 bytes")


PrivateSchemaValidator = validators.extend(Draft202012Validator, {"x-maxBytes": _max_bytes})


@cache
def private_schemas() -> dict[str, dict[str, Any]]:
    """Every private schema by file name."""

    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in SCHEMA_ROOT.glob("*.schema.json")
    }


@cache
def _registry() -> Registry:
    # jsonschema picks the validator class of a referenced resource from its
    # $schema, which would drop x-maxBytes there; every private schema is
    # draft 2020-12, so the resources state that dialect instead.
    return Registry().with_resources(
        (
            schema["$id"],
            Resource.from_contents(
                {key: value for key, value in schema.items() if key != "$schema"},
                default_specification=DRAFT202012,
            ),
        )
        for schema in private_schemas().values()
    )


def private_validator(location: str) -> Any:
    """A validator for an index.json schema location such as file.json#/$defs/Name."""

    filename, _, fragment = location.partition("#")
    reference = private_schemas()[filename]["$id"] + (f"#{fragment}" if fragment else "")
    return PrivateSchemaValidator({"$ref": reference}, registry=_registry())
