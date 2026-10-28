"""Shared accepted plan verdicts and real trusted task/execution reader parity."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jcs
import jsonschema
import pytest
from test_audit_results_toolset import digest, fixture_inputs, package

from contractor_runtime.toolsets.audit_results.accepted_plans import (
    MAX_ACCEPTED_PLAN_BYTES,
    validate_accepted_plan,
)
from contractor_runtime.toolsets.audit_results.contracts import (
    AuditInvocationOwner,
    AuditTrustedInputs,
)
from contractor_runtime.toolsets.audit_results.encoding import (
    AuditResultError,
    CanonicalAuditPackageEncoder,
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
            "profile_mode": "custom-checklist",
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
    # A different, internally valid task is rejected by its actual package
    # digest without transporting a second accepted-plan/dependency copy.
    changed = copy.deepcopy(task)
    changed["accepted_plan"]["plan"]["objective"] += " Changed check."
    changed["accepted_plan"]["plan_digest"] = sha(changed["accepted_plan"]["plan"])
    changed["finding"]["objective"] = changed["accepted_plan"]["plan"]["objective"]
    different = package(
        package_id,
        "item-task",
        [("task-document", "task.json", "application/json", jcs.canonicalize(changed))],
    )
    changed_task, changed_id = _decode_task_package(different, "application/zip")
    with pytest.raises(ValueError, match="identities differ"):
        _match_trusted_inputs([(changed_task, changed_id, digest(different))], decoded_execution)
    encoder = CanonicalAuditPackageEncoder()
    inputs = AuditTrustedInputs(
        AuditInvocationOwner("allocation-1", "invocation-1", digest(payload), (task["item_key"],)),
        payload,
        jcs.canonicalize(execution),
    )
    cached = encoder.trusted_inputs(inputs)
    assert cached.tasks[task["item_key"]]["accepted_plan"] == plan
    assert encoder.trusted_inputs(inputs) is cached
    changed_inputs = replace(
        inputs,
        owner=replace(inputs.owner, task_set_sha256=digest(different)),
        task_package=different,
    )
    with pytest.raises(AuditResultError, match="audit_result_invalid"):
        encoder.trusted_inputs(changed_inputs)
    assert encoder.trusted_inputs(inputs) is cached
    for removed in ("accepted_plan", "dependencies"):
        expanded = copy.deepcopy(execution)
        expanded["items"][0][removed] = copy.deepcopy(task[removed])
        with pytest.raises(ValueError, match="manifest item is invalid"):
            _decode_execution_manifest(jcs.canonicalize(expanded), "application/json")
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


@pytest.mark.parametrize(
    "case", FIXTURE["typedTaskProfileModeCases"], ids=lambda case: case["name"]
)
def test_shared_typed_task_profile_mode(case):
    task_payload, _ = fixture_inputs()
    task, package_id = _decode_task_package(task_payload, "application/zip")
    plan = copy.deepcopy(FIXTURE["base"]["plan"])
    task.update(
        {
            "kind": "finding-verification",
            "item_key": plan["item_key"],
            "workflow_role": plan["route"]["workflow_role"],
            "accepted_plan": plan,
            "dependencies": copy.deepcopy(FIXTURE["base"]["dependencies"]),
            "finding": {
                "receipt_id": plan["source"]["receipt_id"],
                "proposed_check_ordinal": plan["source"]["proposed_check_ordinal"],
                "proposal_ref": plan["source"]["retained_proposal"]["ref"],
                "proposal_digest": plan["source"]["retained_proposal"]["digest"],
                "objective": plan["plan"]["objective"],
                "method": plan["plan"]["method"],
                "limitations": [],
            },
        }
    )
    task.pop("checklist", None)
    if case.get("delete"):
        task.pop("profile_mode", None)
    else:
        task["profile_mode"] = case["value"]
    data = package(
        package_id,
        "item-task",
        [("task-document", "task.json", "application/json", jcs.canonicalize(task))],
    )
    if case["valid"]:
        _decode_task_package(data, "application/zip")
    else:
        with pytest.raises(ValueError):
            _decode_task_package(data, "application/zip")
