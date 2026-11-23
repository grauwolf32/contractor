import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contractor_runtime.contracts import ResolvedModelPolicy

CASES = json.loads(
    (Path(__file__).parents[2] / "api/testdata/v1alpha1/model-image-support-cases.json").read_text()
)


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["name"])
def test_shared_model_image_support(case):
    value = {
        **CASES["base"],
        "ref": {"policyId": "image-policy", "version": "1", "digest": "sha256:" + "a" * 64},
    }
    if case["present"]:
        value["supportsImages"] = case["value"]
    if not case["valid"]:
        with pytest.raises(ValidationError):
            ResolvedModelPolicy.model_validate(value)
    else:
        policy = ResolvedModelPolicy.model_validate(value)
        assert policy.supports_images is case["value"]
