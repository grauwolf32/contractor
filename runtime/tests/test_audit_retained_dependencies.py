"""Shared transport verdicts and actual trusted task/execution reader gates."""

import copy
import hashlib
import json
from pathlib import Path

import jcs
import jsonschema
import pytest
from test_audit_results_toolset import digest, fixture_inputs, package

from contractor_runtime.toolsets.audit_results.dependencies import (
    MAX_DEPENDENCIES,
    MAX_DEPENDENCY_BYTES,
    validate_retained_dependencies,
)
from contractor_runtime.toolsets.audit_results.packages import (
    _decode_execution_manifest,
    _decode_task_package,
    _match_trusted_inputs,
)

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "api/testdata/audit-dependencies/manifest-cases.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_shared_retained_dependency_manifest(case):
    assert FIXTURE["maximumDescriptors"] == MAX_DEPENDENCIES
    assert FIXTURE["maximumBytes"] == MAX_DEPENDENCY_BYTES
    value = copy.deepcopy(FIXTURE["base"])
    if case["path"]:
        target = value
        for part in case["path"][:-1]:
            target = target[part]
        if case.get("delete"):
            del target[case["path"][-1]]
        else:
            target[case["path"][-1]] = case["value"]
    if case["valid"]:
        validate_retained_dependencies(value, value["item_key"])
    else:
        with pytest.raises(ValueError):
            validate_retained_dependencies(value, value.get("item_key"))


def test_shared_retained_dependency_bounds_and_complete_graph():
    value = copy.deepcopy(FIXTURE["base"])
    for count in (MAX_DEPENDENCIES, MAX_DEPENDENCIES + 1):
        value["dependencies"] = []
        for index in range(count):
            descriptor = copy.deepcopy(FIXTURE["base"]["dependencies"][0])
            descriptor["name"] = f"dep-{index:02d}"
            descriptor["requires"] = [f"dep-{previous:02d}" for previous in range(index)]
            value["dependencies"].append(descriptor)
        if count == MAX_DEPENDENCIES:
            validate_retained_dependencies(value, value["item_key"])
        else:
            with pytest.raises(ValueError):
                validate_retained_dependencies(value, value["item_key"])
    value["dependencies"] = copy.deepcopy(FIXTURE["base"]["dependencies"][:2])
    for descriptor in value["dependencies"]:
        descriptor["source"]["ref"]["revision"] = "r" * (64 * 1024)
        descriptor["retained"]["ref"]["revision"] = "r" * (64 * 1024)
    with pytest.raises(ValueError, match="bytes exceed"):
        validate_retained_dependencies(value, value["item_key"])


def test_shared_retained_dependency_real_task_and_execution_readers():
    task_payload, execution_payload = fixture_inputs()
    task, package_id = _decode_task_package(task_payload, "application/zip")
    execution = json.loads(execution_payload)
    task["dependencies"] = copy.deepcopy(FIXTURE["base"])
    task["dependencies"]["item_key"] = task["item_key"]
    task_payload = package(
        package_id,
        "item-task",
        [("task-document", "task.json", "application/json", jcs.canonicalize(task))],
    )
    execution["items"][0]["task_package_digest"] = digest(task_payload)
    decoded_task, decoded_id = _decode_task_package(task_payload, "application/zip")
    decoded_execution = _decode_execution_manifest(jcs.canonicalize(execution), "application/json")
    _match_trusted_inputs([(decoded_task, decoded_id, digest(task_payload))], decoded_execution)
    changed = copy.deepcopy(task)
    changed["dependencies"]["workflow_digest"] = "sha256:" + "f" * 64
    different = package(
        package_id,
        "item-task",
        [("task-document", "task.json", "application/json", jcs.canonicalize(changed))],
    )
    changed_task, changed_id = _decode_task_package(different, "application/zip")
    with pytest.raises(ValueError, match="identities differ"):
        _match_trusted_inputs([(changed_task, changed_id, digest(different))], decoded_execution)
    execution["items"][0]["dependencies"] = copy.deepcopy(task["dependencies"])
    with pytest.raises(ValueError, match="manifest item is invalid"):
        _decode_execution_manifest(jcs.canonicalize(execution), "application/json")
    task["dependencies"]["dependencies"][1]["origin"]["receipt_id"] = "bad / id"
    invalid = package(
        package_id,
        "item-task",
        [("task-document", "task.json", "application/json", jcs.canonicalize(task))],
    )
    with pytest.raises(ValueError):
        _decode_task_package(invalid, "application/zip")


def test_shared_retained_dependency_published_schema_and_canonical_digest():
    schema = json.loads(
        (Path(__file__).parents[2] / "api/audit-dependencies/v1/manifest.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(FIXTURE["base"])
    assert (
        "sha256:" + hashlib.sha256(jcs.canonicalize(FIXTURE["base"])).hexdigest()
        == FIXTURE["canonicalDigest"]
    )
