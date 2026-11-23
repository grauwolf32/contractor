"""Exact target credential references the Runtime shares with the Server and the schema.

api/testdata/v1alpha1/target-credential-cases.json holds the RuntimeCredentialRef
texts that name one immutable target credential revision (RCF-R184). The Go
Server reads the same cases (internal/contracts/target_credential_ref_test.go);
make test-wire-cross-language runs both sides. The same material cases validate
supplied cookies and API key fields at the allocation-private Runtime boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from private_schemas import private_validator

from contractor_runtime.contracts import (
    PrivateProtocolDecodeError,
    RuntimeCredentialRef,
    decode_private,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "api"
    / "testdata"
    / "v1alpha1"
    / "target-credential-cases.json"
)
TARGET_KINDS = {"target-basic@1", "target-bearer@1", "target-cookies@1", "target-api-keys@1"}


def test_shared_target_cookie_and_api_key_material_cases():
    from contractor_runtime.contracts.settings import HTTPIdentityMaterial
    from contractor_runtime.strict_json import strict_json_loads

    cases = json.loads(FIXTURE.read_text())["materials"]
    selected = [case for case in cases if case["kind"] in {"target-cookies@1", "target-api-keys@1"}]
    assert selected
    for case in selected:
        try:
            material = (
                strict_json_loads(case["text"].encode()) if "text" in case else case["material"]
            )
            key = "fields" if case["kind"] == "target-api-keys@1" else "cookies"
            if not isinstance(material, dict) or set(material) != {key}:
                raise ValueError("cookie material shape differs")
            decode_private(
                HTTPIdentityMaterial,
                json.dumps(
                    {
                        "credential": {"credentialId": "fixture", "kind": case["kind"]},
                        "fields": material[key],
                    }
                ).encode(),
            )
            accepted = True
        except (ValueError, TypeError, PrivateProtocolDecodeError):
            accepted = False
        assert accepted == case["valid"], case["name"]


def _references() -> list[dict[str, Any]]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if set(fixture) != {"description", "references", "materials"} or not fixture["references"]:
        raise ValueError("target-credential-cases.json has an unexpected shape")
    names: set[str] = set()
    for case in fixture["references"]:
        if set(case) - {"name", "text", "valid", "schema"} or not {"name", "text", "valid"} <= set(
            case
        ):
            raise ValueError(f"{case}: a reference case needs a name, text and a verdict")
        if case["name"] in names or case.get("schema", "accepts") != "accepts":
            raise ValueError(f"{case['name']}: duplicate name or unknown schema verdict")
        if case.get("schema") and case["valid"]:
            raise ValueError(f"{case['name']}: only an invalid case records a schema verdict")
        names.add(case["name"])
    return fixture["references"]


def _schema_accepts(text: str) -> bool:
    try:
        document = json.loads(text)
    except ValueError:
        return False
    validator = private_validator("allocation.schema.json#/$defs/RuntimeCredentialRef")
    return not any(validator.iter_errors(document))


def test_shared_target_credential_references_match_the_server_and_schema() -> None:
    problems: list[str] = []
    target_kinds = 0
    for case in _references():
        try:
            ref = decode_private(RuntimeCredentialRef, case["text"])
        except PrivateProtocolDecodeError:
            ref = None
        if (ref is not None) is not case["valid"]:
            problems.append(f"{case['name']}: decode_private accepted = {ref is not None}")
        if ref is not None and ref.kind in TARGET_KINDS:
            target_kinds += 1
        expected = case["valid"] or case.get("schema") == "accepts"
        if _schema_accepts(case["text"]) is not expected:
            problems.append(f"{case['name']}: the private schema disagrees")
    assert target_kinds > 0
    assert problems == []
