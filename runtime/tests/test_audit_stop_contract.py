"""Shared server stop/read semantics, independent of the Worker success gate."""

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_audit_common_contract import REGISTRY, ROOT, SCHEMA

from contractor_runtime.strict_json import strict_json_loads

CASES = json.loads((ROOT / "api/testdata/audit-stops.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_system_stop_read_contract(case):
    valid = True
    try:
        value = strict_json_loads(case["document"])
        Draft202012Validator(
            {"$ref": SCHEMA["$id"] + "#/$defs/task_stop"}, registry=REGISTRY
        ).validate(value)
        if (
            case["checking"] != ("verdict" in value)
            or "\x00" in value["reason"]["message"]
            or len(value["reason"]["message"].encode()) > 4096
            or any(
                source["scope_kind"] != "audit" or source["scope_id"] != case["audit_id"]
                for source in value["evidence"]
            )
        ):
            valid = False
    except (ValueError, TypeError, KeyError, ValidationError):
        valid = False
    assert valid == case["valid"]
