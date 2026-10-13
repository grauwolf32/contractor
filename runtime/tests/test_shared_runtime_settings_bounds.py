"""RuntimeSettings bounds the Runtime shares with the Go Server and the schema.

internal/contracts/runtimesettings reads the same
api/testdata/v1alpha1/runtime-settings-bounds-cases.json; make
test-wire-cross-language runs both sides.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from fakes.telemetry import SERVER_EXPORT
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from referencing import Registry, Resource

from contractor_runtime.contracts import RuntimeSettings
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


def _schema_validator() -> Draft202012Validator:
    schemas = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (ROOT / "api" / "v1alpha1").glob("*.schema.json")
    ]
    registry = Registry().with_resources(
        [(schema["$id"], Resource.from_contents(schema)) for schema in schemas]
    )
    allocation = next(
        schema for schema in schemas if schema["$id"].endswith("allocation.schema.json")
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
