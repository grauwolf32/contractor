"""One portable Artifact name contract across wire validation and JSON Schema."""

import json
from pathlib import Path

import pytest
from fakes.spec import allocation_spec
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from contractor_runtime.contracts import AllocationSpecV2, ArtifactRef

ROOT = Path(__file__).parents[2]
CASES = json.loads((ROOT / "api/testdata/v1alpha1/artifact-name-cases.json").read_text())
SCHEMA = json.loads((ROOT / "api/v1alpha1/common.schema.json").read_text())["$defs"]["artifactRef"]


@pytest.mark.parametrize("field", ["namespace", "name"])
@pytest.mark.parametrize(
    "valid,name",
    [(True, name) for name in CASES["valid"]] + [(False, name) for name in CASES["invalid"]],
)
def test_artifact_names_match_schema(field: str, valid: bool, name: str) -> None:
    value = {"namespace": "worker", "name": "report", field: name}
    assert Draft202012Validator(SCHEMA).is_valid(value) == valid
    if valid:
        assert ArtifactRef.model_validate(value).model_dump(exclude_none=True) == value
    else:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate(value)


@pytest.mark.parametrize("name", CASES["invalid"])
def test_allocation_namespace_rejects_nonportable_names(name: str) -> None:
    value = allocation_spec().model_dump(mode="json", by_alias=True, exclude_none=True)
    value["namespace"] = name
    with pytest.raises(ValidationError):
        AllocationSpecV2.model_validate(value)
