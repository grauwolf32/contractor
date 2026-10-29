"""Shared optional analysis authoring contract; exact evidence remains Server-owned."""

import json
from pathlib import Path

import pytest

from contractor_runtime.toolsets.common.finding_analysis import canonical_analysis


def test_shared_finding_analysis_cases() -> None:
    path = Path(__file__).parents[2] / "api/testdata/finding-analysis-cases.json"
    cases = json.loads(path.read_text())["cases"]
    assert cases
    for case in cases:
        if case["valid"]:
            assert canonical_analysis(case["document"]) == case["document"].encode()
        else:
            with pytest.raises(ValueError, match="finding analysis is invalid"):
                canonical_analysis(case["document"])


def test_analysis_authoring_canonicalizes_and_bounds_text() -> None:
    value = {
        "schema": "contractor.finding.analysis.v1",
        "client_key": "candidate",
        "sources": [],
        "locations": [],
        "flows": [],
        "http_exchanges": [],
    }
    assert canonical_analysis(json.dumps(value, indent=2)) == canonical_analysis(json.dumps(value))
    with pytest.raises(ValueError, match="finding analysis is invalid"):
        canonical_analysis(" " * (1024 * 1024 + 1))
