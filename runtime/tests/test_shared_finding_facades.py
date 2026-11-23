"""The finding facades' shared HTTP attempt and location rules (Go auditdomain)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contractor_runtime.toolsets.common.finding_locations import normalize_locations
from contractor_runtime.toolsets.common.http_evidence import HTTPAttempt

FIXTURES = Path(__file__).parents[2] / "api" / "testdata" / "findings"
LOCATION_CASES = json.loads((FIXTURES / "finding-locations.json").read_text())
HTTP_ATTEMPT_CASES = json.loads((FIXTURES / "finding-http-attempts.json").read_text())


@pytest.mark.parametrize("case", HTTP_ATTEMPT_CASES, ids=lambda case: case["name"])
def test_shared_runtime_and_server_share_http_attempt_rules(case: dict[str, Any]) -> None:
    attempt = {
        "method": "GET",
        "url": case["url"] + case.get("repeat", "") * case.get("count", 0),
        "headers": [],
        "body_base64": "",
        "status": case["status"],
        "response_headers": [],
    }
    if case["valid"]:
        HTTPAttempt.model_validate(attempt)
    else:
        with pytest.raises(ValueError):
            HTTPAttempt.model_validate(attempt)


@pytest.mark.parametrize("case", LOCATION_CASES, ids=lambda case: case["name"])
def test_shared_locations_match_go_contract(case: dict[str, Any]) -> None:
    if case["valid"]:
        assert normalize_locations([case["location"]]) == [case["location"]]
    else:
        with pytest.raises(ValueError):
            normalize_locations([case["location"]])
