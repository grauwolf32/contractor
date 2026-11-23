"""The Server, Runtime and standalone schema read the same engagement vectors."""

from __future__ import annotations

import json
from pathlib import Path

import jcs
from private_schemas import private_validator

from contractor_runtime.contracts import PrivateProtocolDecodeError, decode_private
from contractor_runtime.contracts.engagement import Engagement, canonical_url
from contractor_runtime.digests import sha256_digest

FIXTURE = Path(__file__).resolve().parents[2] / "api/testdata/v1alpha1/engagement-cases.json"


def test_shared_engagement() -> None:
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["documents"] and fixture["urls"]
    validator = private_validator("engagement.schema.json")
    problems: list[str] = []
    for case in fixture["documents"]:
        try:
            doc = decode_private(Engagement, case["text"])
        except (PrivateProtocolDecodeError, UnicodeError):
            doc = None
        if (doc is not None) != case["valid"]:
            problems.append(f"{case['name']}: Runtime accepted={doc is not None}")
        if doc is not None:
            canonical = jcs.canonicalize(doc.model_dump(by_alias=True, exclude_none=True))
            assert canonical.decode() == case["canonical"], case["name"]
            assert sha256_digest(canonical) == case["digest"], case["name"]
        try:
            accepted = validator.is_valid(json.loads(case["text"]))
        except (ValueError, UnicodeError):
            accepted = False
        if accepted != (case["valid"] or case.get("schema") == "accepts"):
            problems.append(f"{case['name']}: schema disagrees")
    for case in fixture["urls"]:
        try:
            canonical = canonical_url(case["url"])
        except ValueError:
            canonical = None
        if canonical != case.get("canonical"):
            problems.append(f"{case['url']}: canonical URL disagrees")
    assert problems == []
