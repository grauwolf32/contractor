from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from contractor_runtime.audit_completion_contracts import (
    MAX_COLLECTED_BYTES,
    MAX_PACKAGE_BYTES,
    AuditEncodedPackage,
    AuditEvidence,
    AuditInvocationOwner,
    AuditPublicationReceipt,
    AuditRecordReceipt,
    AuditSnapshot,
    AuditTrustedInputs,
    ContinueCompletion,
    NormalizedAuditItem,
    RecordedAuditItem,
    SealedAuditSnapshot,
)
from contractor_runtime.contracts import (
    AgentRegistrationV2,
    AllocationSpec,
    AllocationSpecV2,
    RuntimeCompletionCapabilities,
    WorkerCompletionContract,
    decode_private_v2,
)

ROOT = Path(__file__).parents[2]
CASES = json.loads(
    (ROOT / "testdata/contracts/private-v2/audit-completion-cases.json").read_bytes()
)
MODELS = {
    "contract": WorkerCompletionContract,
    "capabilities": RuntimeCompletionCapabilities,
    "allocation": AllocationSpecV2,
    "legacy-allocation": AllocationSpec,
    "registration": AgentRegistrationV2,
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_go_completion_wire_fixtures(case):
    raw = json.dumps(case["value"]).encode()
    if case["valid"]:
        value = decode_private_v2(MODELS[case["model"]], raw)
        assert value.model_dump(mode="json", by_alias=True, exclude_none=True) == case["value"]
    else:
        with pytest.raises(ValueError):
            decode_private_v2(MODELS[case["model"]], raw)


def test_generated_private_and_legacy_schemas_accept_complete_contracts():
    schemas = {
        path.name: json.loads(path.read_bytes())
        for path in (ROOT / "api/v1alpha1").glob("*.schema.json")
    }
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    private_alloc = json.loads((ROOT / "api/private-v2/allocation.schema.json").read_bytes())
    private_registration = json.loads(
        (ROOT / "api/private-v2/agent-registration.schema.json").read_bytes()
    )
    for case in CASES:
        if case["valid"] and case["model"] in {
            "allocation",
            "registration",
            "legacy-allocation",
            "contract",
        }:
            schema = {
                "allocation": private_alloc,
                "registration": private_registration,
                "legacy-allocation": schemas["allocation.schema.json"],
                "contract": schemas["worker-completion-contract.schema.json"],
            }[case["model"]]
            Draft202012Validator(schema, registry=registry).validate(case["value"])
    for case in CASES:
        if not case["valid"] and case["model"] == "contract" and case["name"] != "same-input":
            assert list(
                Draft202012Validator(
                    schemas["worker-completion-contract.schema.json"], registry=registry
                ).iter_errors(case["value"])
            )


def owner(keys=("first", "second")):
    return AuditInvocationOwner("allocation", "worker-invocation", "sha256:" + "a" * 64, keys)


def item(key="first", revision=1, **kwargs):
    return RecordedAuditItem(
        NormalizedAuditItem(key, "inconclusive", "bounded observation", **kwargs), revision
    )


def test_normalized_snapshot_is_immutable_owned_and_uses_trusted_order():
    snapshot = AuditSnapshot(owner(), (item(),))
    assert snapshot.items[0].revision == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.items[0].value.summary = "model rewrite"
    with pytest.raises(ValueError):
        NormalizedAuditItem("first", "blocked", "reason", completed=["coverage"])
    with pytest.raises(ValueError):
        AuditSnapshot(owner(), (item("second"), item("first")))
    with pytest.raises(ValueError):
        AuditSnapshot(owner(), (item("foreign"),))
    with pytest.raises(ValueError):
        AuditSnapshot(owner(), (item(), item()))
    with pytest.raises(ValueError):
        SealedAuditSnapshot(owner(), (item(),))
    sealed = SealedAuditSnapshot(owner(), (item(), item("second", 2)))
    assert sealed.owner == snapshot.owner
    assert sealed.owner != dataclasses.replace(sealed.owner, invocation_id="worker-another")


@pytest.mark.parametrize("revision", [0, -1, True, 2**53])
def test_recorded_revisions_are_positive_exact_bounded_integers(revision):
    with pytest.raises(ValueError):
        item(revision=revision)


def test_limits_and_separate_recorded_publication_receipts():
    evidence = AuditEvidence("source", "retained support")
    with pytest.raises(ValueError):
        AuditInvocationOwner("a", "b", "sha256:" + "a" * 64, tuple(f"item-{i}" for i in range(65)))
    with pytest.raises(ValueError):
        NormalizedAuditItem("first", "blocked", "я" * 8193)
    with pytest.raises(ValueError):
        NormalizedAuditItem("first", "pass", "not a canonical assessment")
    with pytest.raises(ValueError):
        NormalizedAuditItem(
            "first", "blocked", "reason", completed=tuple(f"c-{i}" for i in range(513))
        )
    with pytest.raises(ValueError):
        AuditSnapshot(
            owner(), (item(evidence=(evidence,) * 129), item("second", evidence=(evidence,) * 128))
        )
    receipt = AuditRecordReceipt(owner(), (("first", 1),), ("second",))
    assert receipt.accepted_count == 1 and receipt.total_count == 2 and not receipt.complete
    assert not hasattr(receipt, "revision") and not hasattr(receipt, "artifact")
    with pytest.raises(ValueError):
        dataclasses.replace(receipt, missing_item_keys=())
    with pytest.raises(ValueError):
        AuditEncodedPackage(owner(), b"zip", MAX_COLLECTED_BYTES + 1, (3,))
    with pytest.raises(ValueError):
        AuditEncodedPackage(owner(), b"x" * (MAX_PACKAGE_BYTES + 1), 1, (1,))
    with pytest.raises(ValueError):
        AuditEncodedPackage(owner(), b"zip", 1, (MAX_PACKAGE_BYTES + 1,))
    valid = AuditPublicationReceipt(owner(), "audit-check", "result", "r1", "sha256:" + "b" * 64, 3)
    assert valid.owner == receipt.owner and valid.revision == "r1"
    with pytest.raises(ValueError):
        dataclasses.replace(valid, revision="")
    with pytest.raises(ValueError):
        ContinueCompletion("x" * (16 * 1024 + 1))


def test_types_do_not_enable_runtime_support(tmp_path):
    from contractor_runtime.factories import built_in_factories

    assert "audit-results@2" not in built_in_factories(tmp_path).toolsets
    raw = json.loads(
        (ROOT / "testdata/contracts/private-v2/valid/agent-registration.json").read_bytes()
    )
    registration = AgentRegistrationV2.model_validate_json(json.dumps(raw))
    assert registration.capabilities is None
    assert "capabilities" not in registration.model_dump(by_alias=True, exclude_none=True)
    from fakes.spec import allocation_spec

    spec = allocation_spec()
    assert spec.completion_contract is None
    assert "completionContract" not in spec.model_dump(by_alias=True, exclude_none=True)


def test_encoder_inputs_pin_both_artifacts_and_keep_sensitive_bytes_immutable():
    package = b"task-package-secret"
    identity = dataclasses.replace(
        owner(), task_set_sha256="sha256:" + hashlib.sha256(package).hexdigest()
    )
    inputs = AuditTrustedInputs(identity, package, b"manifest-secret")
    assert (
        inputs.execution_manifest_sha256
        == "sha256:" + hashlib.sha256(b"manifest-secret").hexdigest()
    )
    assert "secret" not in repr(inputs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.task_package = b"replacement"
    with pytest.raises(ValueError, match="digest"):
        AuditTrustedInputs(identity, b"changed", b"manifest")
    with pytest.raises(ValueError, match="bounds"):
        AuditTrustedInputs(identity, bytearray(package), b"manifest")
    protected = NormalizedAuditItem(
        "first", "blocked", "summary-secret", evidence=(AuditEvidence("source", "evidence-secret"),)
    )
    assert "secret" not in repr(protected)
