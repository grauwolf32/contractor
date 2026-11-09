"""Common bounded candidate envelope shared with the Server domain consumer."""

import json
from pathlib import Path

import jsonschema

from contractor_runtime.toolsets.scan.findings import (
    MAX_BATCH_BYTES,
    MAX_CANDIDATES,
    MAX_EVIDENCE_BYTES,
)


def test_shared_finding_candidate_envelope():
    fixture = json.loads(
        (Path(__file__).parents[2] / "api/testdata/finding-candidates-cases.json").read_text()
    )
    assert fixture["candidateLimit"] == MAX_CANDIDATES
    assert fixture["batchByteLimit"] == MAX_BATCH_BYTES
    assert fixture["evidenceByteLimit"] == MAX_EVIDENCE_BYTES
    schema = json.loads(
        (Path(__file__).parents[2] / "api/v1alpha1/finding-candidates.schema.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    for case in fixture["cases"]:
        assert validator.is_valid(case["batch"]) == case["valid"], case["name"]
