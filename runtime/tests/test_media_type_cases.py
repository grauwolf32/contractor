"""One media-type grammar across Runtime validation and the private JSON Schema."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from contractor_runtime.artifacts import _validate_media_type
from contractor_runtime.contracts import MEDIA_TYPE_PATTERN, ArtifactReadResult

ROOT = Path(__file__).parents[2]
CASES = json.loads((ROOT / "api/testdata/v1alpha1/media-type-cases.json").read_text())
SCHEMA = json.loads((ROOT / "api/v1alpha1/common.schema.json").read_text())["$defs"]["mediaType"]


@pytest.mark.parametrize(
    "valid,value",
    [(True, value) for value in CASES["valid"]] + [(False, value) for value in CASES["invalid"]],
)
def test_media_type_cases_match_runtime_and_schema(valid: bool, value: str) -> None:
    assert (MEDIA_TYPE_PATTERN.fullmatch(value) is not None) == valid
    assert Draft202012Validator(SCHEMA).is_valid(value) == valid
    result = {
        "apiVersion": "contractor/v1alpha1",
        "artifact": {"namespace": "worker", "name": "report", "revision": "1"},
        "mediaType": value,
        "size": 0,
    }
    if valid:
        _validate_media_type(value)
        ArtifactReadResult.model_validate(result)
    else:
        with pytest.raises(ValueError, match="media type is invalid"):
            _validate_media_type(value)
        with pytest.raises(ValidationError):
            ArtifactReadResult.model_validate(result)
