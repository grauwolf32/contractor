"""The model recovery participant protocol the Runtime shares with the Server.

internal/gatewayrecovery reads the same
api/testdata/v1alpha1/gateway-recovery-participant-cases.json; make
test-wire-cross-language runs both sides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from contractor_runtime.llm.client import GatewayRequestError, _attempt_update, _redelivery
from contractor_runtime.llm.errors import GatewayFailure
from contractor_runtime.llm.recovery import (
    RECOVERY_ACTIONS,
    RECOVERY_FAILURE_CODES,
    TERMINAL_RECOVERY_ACTIONS,
    RecoveryUpdate,
)

FIXTURES = Path(__file__).resolve().parents[2] / "api" / "testdata" / "v1alpha1"


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CASES = _fixture("gateway-recovery-participant-cases.json")
if set(CASES) != {
    "description",
    "actions",
    "failureCodes",
    "requests",
    "outcomes",
    "redelivery",
}:
    raise ValueError("gateway-recovery-participant-cases.json has an unexpected shape")


def _arguments(update: dict[str, Any]) -> tuple[str, str | None, float]:
    if not {"action"} <= set(update) <= {"action", "code", "retryAfterSeconds"}:
        raise ValueError(f"invalid participant update {update}")
    return (update["action"], update.get("code"), update.get("retryAfterSeconds", 0))


def test_shared_gateway_recovery_actions_and_failure_codes() -> None:
    assert {item["action"] for item in CASES["actions"]} == RECOVERY_ACTIONS
    assert {
        item["action"] for item in CASES["actions"] if item["terminal"]
    } == TERMINAL_RECOVERY_ACTIONS
    assert set(CASES["failureCodes"]) == RECOVERY_FAILURE_CODES
    # Every code the shared classifier marks retryable may be reported as
    # failed, and no permanent one may (SCH-R112).
    for case in _fixture("gateway-failure-classification-cases.json")["cases"]:
        assert (case["code"] in RECOVERY_FAILURE_CODES) is case["retryable"], case["name"]


@pytest.mark.parametrize("case", CASES["requests"], ids=lambda case: case["name"])
def test_shared_gateway_recovery_requests(case: dict[str, Any]) -> None:
    if case["valid"]:
        RecoveryUpdate.model_validate_json(json.dumps(case["request"]))
    else:
        with pytest.raises(ValidationError):
            RecoveryUpdate.model_validate_json(json.dumps(case["request"]))


def _error(failure: dict[str, Any], retry_after_seconds: float) -> GatewayRequestError:
    if (
        not {"code", "retryable"}
        <= set(failure)
        <= {
            "code",
            "retryable",
            "abandoned",
            "shouldRetry",
        }
    ):
        raise ValueError(f"invalid participant failure {failure}")
    error = GatewayRequestError(
        "ProviderError",
        retryable=failure["retryable"],
        failure=GatewayFailure(failure["code"], failure["retryable"]),
        retry_after_seconds=retry_after_seconds,
    )
    # The client marks an x-should-retry: false response and an abandoned
    # request this way before the update is chosen.
    if failure.get("shouldRetry") is False or failure.get("abandoned"):
        error.transport_retry_allowed = False
    error.abandoned = failure.get("abandoned", False)
    return error


@pytest.mark.parametrize("case", CASES["outcomes"], ids=lambda case: case["name"])
def test_shared_gateway_recovery_outcomes(case: dict[str, Any]) -> None:
    if case.get("cancelled"):
        update = _attempt_update(None, cancelled=True)
    elif case["failure"] is None:
        update = _attempt_update(None)
    else:
        update = _attempt_update(_error(case["failure"], case["retryAfterSeconds"]))
    assert update == _arguments(case["update"])
    assert (update[0] == "failed") is case["resend"]
    action, code, retry_after_seconds = update
    RecoveryUpdate(
        model="worker",
        requestId="request-1",
        action=action,
        code=code,
        retryAfterSeconds=retry_after_seconds,
    )


@pytest.mark.parametrize("case", CASES["redelivery"], ids=lambda case: case["update"]["action"])
def test_shared_gateway_recovery_redelivery(case: dict[str, Any]) -> None:
    assert _redelivery(_arguments(case["update"])) == _arguments(case["redelivered"])
