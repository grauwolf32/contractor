"""Shared accepted plan verdicts and real trusted task/execution reader parity."""

import copy
import hashlib
import json
from pathlib import Path

import jcs
import jsonschema
import pytest
from test_audit_results_toolset import digest, fixture_inputs, package

from contractor_runtime.toolsets.audit_results.accepted_plans import (
    MAX_ACCEPTED_PLAN_BYTES,
    validate_accepted_plan,
)
from contractor_runtime.toolsets.audit_results.packages import (
    _decode_execution_manifest,
    _decode_task_package,
    _match_trusted_inputs,
)
from contractor_runtime.toolsets.common.check_plans import (
    MAX_CHECK_CAPABILITIES,
    MAX_CHECK_CAPABILITY_TOOLS,
    METHOD_PHASES,
    ProposedCheck,
)

FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "api/testdata/audit-check-plans/accepted-plan-cases.json"
    ).read_text()
)


def sha(value):
    return "sha256:" + hashlib.sha256(jcs.canonicalize(value)).hexdigest()


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_shared_accepted_plan(case):
    value = copy.deepcopy(FIXTURE["base"])
    for patch in case["patches"]:
        parent = value
        for part in patch["path"][:-1]:
            parent = parent[part]
        if patch.get("delete"):
            del parent[patch["path"][-1]]
        else:
            parent[patch["path"][-1]] = patch["value"]
    if case.get("refreshDependenciesDigest"):
        value["plan"]["dependencies_digest"] = sha(value["dependencies"])
    if case["valid"]:
        validate_accepted_plan(value["plan"], value["dependencies"], value["plan"]["item_key"])
    else:
        with pytest.raises(ValueError):
            validate_accepted_plan(value["plan"], value["dependencies"], value["plan"]["item_key"])


def test_shared_accepted_plan_registry_and_jcs_bytes():
    assert FIXTURE["maximumPlanBytes"] == MAX_ACCEPTED_PLAN_BYTES
    assert FIXTURE["maximumCapabilities"] == MAX_CHECK_CAPABILITIES
    assert FIXTURE["maximumCapabilityTools"] == MAX_CHECK_CAPABILITY_TOOLS
    assert {method["method"]: method["phase"] for method in FIXTURE["methods"]} == METHOD_PHASES
    assert sha(FIXTURE["base"]["plan"]) == FIXTURE["canonicalAcceptedPlanDigest"]
    assert sha(FIXTURE["base"]["plan"]["plan"]) == FIXTURE["canonicalPlanDigest"]
    assert sha(FIXTURE["base"]["dependencies"]) == FIXTURE["canonicalDependenciesDigest"]
    # The argument model preserves the publisher's bounded repair path and ordinary methods.
    assert ProposedCheck(objective="ordinary", method="operator-method").method == "operator-method"


def test_shared_accepted_plan_actual_task_execution_readers():
    task_payload, execution_payload = fixture_inputs()
    task, package_id = _decode_task_package(task_payload, "application/zip")
    execution = json.loads(execution_payload)
    plan = copy.deepcopy(FIXTURE["base"]["plan"])
    task.update(
        {
            "item_key": plan["item_key"],
            "subject_key": "receipt-1",
            "kind": "finding-verification",
            "workflow_role": plan["route"]["workflow_role"],
            "accepted_plan": plan,
            "dependencies": copy.deepcopy(FIXTURE["base"]["dependencies"]),
            "finding": {
                "receipt_id": plan["source"]["receipt_id"],
                "proposed_check_ordinal": 0,
                "proposal_ref": plan["source"]["retained_proposal"]["ref"],
                "proposal_digest": plan["source"]["retained_proposal"]["digest"],
                "objective": plan["plan"]["objective"],
                "method": plan["plan"]["method"],
                "limitations": [],
            },
        }
    )
    task.pop("checklist", None)
    item = execution["items"][0]
    item.update(
        {
            "item_key": task["item_key"],
            "subject_key": task["subject_key"],
            "accepted_plan": copy.deepcopy(plan),
            "dependencies": copy.deepcopy(task["dependencies"]),
        }
    )
    payload = package(
        package_id,
        "item-task",
        [("task-document", "task.json", "application/json", jcs.canonicalize(task))],
    )
    item["task_package_digest"] = digest(payload)
    decoded_task, decoded_id = _decode_task_package(payload, "application/zip")
    decoded_execution = _decode_execution_manifest(jcs.canonicalize(execution), "application/json")
    _match_trusted_inputs([(decoded_task, decoded_id, digest(payload))], decoded_execution)
    decoded_execution["items"][0]["accepted_plan"]["route"]["workflow_role"] = "another"
    with pytest.raises(ValueError, match="identities differ"):
        _match_trusted_inputs([(decoded_task, decoded_id, digest(payload))], decoded_execution)
    task["accepted_plan"] = None
    invalid = package(
        package_id,
        "item-task",
        [("task-document", "task.json", "application/json", jcs.canonicalize(task))],
    )
    with pytest.raises(ValueError):
        _decode_task_package(invalid, "application/zip")


def test_shared_accepted_plan_published_schema():
    schema = json.loads(
        (
            Path(__file__).parents[2] / "api/audit-check-plans/v1/accepted-plan.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(FIXTURE["base"]["plan"])
