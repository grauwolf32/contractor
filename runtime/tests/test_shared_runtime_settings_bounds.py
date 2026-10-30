"""RuntimeSettings bounds the Runtime shares with the Go Server and the schema.

internal/contracts/runtimesettings reads the same
api/testdata/v1alpha1/runtime-settings-bounds-cases.json, and internal/contracts
the same api/testdata/v1alpha1/credential-field-value-cases.json; make
test-wire-cross-language runs both sides.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

import pytest
from credential_field_values import field_cases, read_credential_field_value_cases, set_pointer
from fakes.telemetry import SERVER_EXPORT
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from referencing import Registry, Resource

from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.contracts.fields import is_credential_field_value
from contractor_runtime.contracts.settings import default_source_archive_limits

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "api" / "testdata" / "v1alpha1"


def _fixture() -> dict[str, Any]:
    fixture = json.loads(
        (FIXTURES / "runtime-settings-bounds-cases.json").read_text(encoding="utf-8")
    )
    if set(fixture) != {"description", "base", "serverDefaults", "cases"} or not fixture["cases"]:
        raise ValueError("runtime-settings-bounds-cases.json has an unexpected shape")
    for case in fixture["cases"]:
        if not {"name", "valid"} <= set(case) <= {"name", "set", "remove", "valid", "crossField"}:
            raise ValueError(f"{case}: a case needs a name and a verdict")
        if case.get("crossField") and case["valid"]:
            raise ValueError(f"{case['name']}: a crossField case is invalid")
    return fixture


FIXTURE = _fixture()
FIELD_VALUES = read_credential_field_value_cases()


@functools.cache
def _schema_validator() -> Draft202012Validator:
    schemas = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (ROOT / "api" / "v1alpha1").glob("*.schema.json")
    ]
    registry = Registry().with_resources(
        [(schema["$id"], Resource.from_contents(schema)) for schema in schemas]
    )
    allocation = next(
        schema for schema in schemas if schema["$id"].rsplit("/", 1)[-1] == "allocation.schema.json"
    )
    return Draft202012Validator(
        {"$ref": allocation["$id"] + "#/$defs/RuntimeSettings"}, registry=registry
    )


def _parent(document: dict[str, Any], pointer: str) -> tuple[dict[str, Any], str]:
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}")
    *path, key = pointer[1:].split("/")
    current = document
    for segment in path:
        current = current[segment]
        if not isinstance(current, dict):
            raise ValueError(f"JSON pointer {pointer!r} does not name an object")
    return current, key


def _case_document(case: dict[str, Any]) -> dict[str, Any]:
    document = copy.deepcopy(json.loads((FIXTURES / FIXTURE["base"]).read_text(encoding="utf-8")))
    for pointer, value in case.get("set", {}).items():
        parent, key = _parent(document, pointer)
        parent[key] = copy.deepcopy(value)
    for pointer in case.get("remove", []):
        parent, key = _parent(document, pointer)
        del parent[key]
    return document


def test_shared_runtime_settings_server_defaults() -> None:
    # The Go twin compares serverDefaults with the Server's defaults; the
    # Runtime has none and only accepts what the Server sends.
    assert FIXTURE["base"].startswith("valid/")
    assert SERVER_EXPORT.model_dump(by_alias=True) == FIXTURE["serverDefaults"]["export"]
    assert (
        default_source_archive_limits().model_dump(by_alias=True)
        == (FIXTURE["serverDefaults"]["sourceArchiveLimits"])
    )


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_shared_runtime_settings_bounds_cases(case: dict[str, Any]) -> None:
    document = _case_document(case)
    if case["valid"]:
        RuntimeSettings.model_validate_json(json.dumps(document))
    else:
        with pytest.raises(ValidationError):
            RuntimeSettings.model_validate_json(json.dumps(document))
    schema_accepts = not list(_schema_validator().iter_errors(document))
    assert schema_accepts is (case["valid"] or case.get("crossField", False)), case["name"]


def test_shared_credential_field_value_grammar() -> None:
    disagreements = [
        case.name
        for case in FIELD_VALUES.grammar
        if is_credential_field_value(case.value) is not case.valid
    ]
    assert disagreements == []


@pytest.mark.parametrize(
    "field",
    [field for field in FIELD_VALUES.fields if "runtimeSettings" in field],
    ids=lambda field: field["name"],
)
def test_shared_credential_field_values_in_runtime_settings(field: dict[str, Any]) -> None:
    # The RuntimeSettings model and the private schema give every case of the
    # shared credential field-value grammar its verdict where the value sits.
    location = field["runtimeSettings"]
    disagreements = []
    for case in field_cases(field, FIELD_VALUES.grammar):
        document = copy.deepcopy(FIELD_VALUES.base)
        for pointer, value in location.get("set", {}).items():
            set_pointer(document, pointer, copy.deepcopy(value))
        set_pointer(document, location["pointer"], case.value)
        try:
            RuntimeSettings.model_validate_json(json.dumps(document))
            model_accepts = True
        except ValidationError:
            model_accepts = False
        schema_accepts = _schema_validator().is_valid(document)
        if (model_accepts, schema_accepts) != (case.valid, case.valid):
            disagreements.append((case.name, model_accepts, schema_accepts))
    assert disagreements == []
