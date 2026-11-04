"""Exercise the producer's real Runtime assembler with the shared wire cases."""

import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from contractor_runtime.contracts.audit_documents import decode_audit_document
from contractor_runtime.toolsets.audit_results.proposals import assemble_proposals


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parents[2] / "api/testdata/audit-common/proposals.json").read_text()
    )["cases"],
    ids=lambda case: case["name"],
)
def test_shared_producer_assembler_cases(case):
    def assemble():
        kind = json.loads(case["document"])["schema"]
        value = decode_audit_document(case["document"].encode(), schema=kind)
        if kind == "contractor.audit.proposal-context.v1":
            return case["document"].encode()
        return assemble_proposals(
            json.dumps(value["hypotheses"]), json.dumps(value["work"]), case["context"].encode()
        )

    if case["valid"]:
        result = json.loads(assemble())
        assert result["schema"] == json.loads(case["document"])["schema"]
    else:
        with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
            assemble()
