"""The ordinary Artifact tools receive refs matching the common descriptor schema."""

import copy
import json
from pathlib import Path

import jcs
import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from contractor_runtime.strict_json import strict_json_loads

ROOT = Path(__file__).parents[2]


def test_shared_material_set_descriptors_and_cumulative_limits():
    fixture = json.loads((ROOT / "api/testdata/material-set-cases.json").read_text())
    schema = json.loads((ROOT / "api/v1alpha1/material-set.schema.json").read_text())
    common = json.loads((ROOT / "api/v1alpha1/common.schema.json").read_text())
    registry = Registry().with_resource(common["$id"], Resource.from_contents(common))
    validator = Draft202012Validator(schema, registry=registry)
    assert fixture["limits"]["materials"] == schema["x-maxSelectedMaterials"]
    assert fixture["limits"]["descriptorBytes"] == schema["x-maxDescriptorBytes"]
    for case in fixture["cases"]:
        members = []
        for i in range(case["count"]):
            member = copy.deepcopy(fixture["base"])
            member["ref"]["name"] = "same" if case["name"] == "duplicates" else f"report-{i}"
            if case["name"] == "missing-revision":
                del member["ref"]["revision"]
            member.update(case.get("change", {}))
            members.append(member)
        value = {"schema": "contractor.material-set.v1", "materials": members}
        if case["name"] == "null-members":
            value["materials"] = None
        raw = jcs.canonicalize(value)
        if case["name"] == "raw-byte-overflow":
            raw += b" " * schema["x-maxDescriptorBytes"]
        try:
            if len(raw) > schema["x-maxDescriptorBytes"]:
                raise ValueError("descriptor bytes")
            parsed = strict_json_loads(raw.decode())
            validator.validate(parsed)
            distinct = set(jcs.canonicalize(m) for m in parsed["materials"])
            if (
                len(distinct) > schema["x-maxSelectedMaterials"]
                or sum(len(m) for m in distinct) > schema["x-maxDescriptorBytes"]
            ):
                raise ValueError("cumulative selection")
        except (ValueError, TypeError, ValidationError) as error:
            if case["valid"]:
                pytest.fail(f"{case['name']}: {error}")
        else:
            assert case["valid"], case["name"]
