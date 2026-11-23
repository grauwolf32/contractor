"""Registered ordinary check semantics shared by finding tools and result readers."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from contractor_runtime.contracts.base import _require_text

FINDING_CHECK_PLAN_SCHEMA = "contractor.audit.finding-check-plan.v1"
MAX_CHECK_CAPABILITIES = 32
MAX_CHECK_CAPABILITY_TOOLS = 128
METHOD_PHASES = {
    "verify": "verify",
    "replay": "replay",
    "applicability": "applicability",
    "alternative": "alternative",
    "static-trace": "verify",
    "http-probe": "verify",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}")


class ProposedCheck(BaseModel):
    """One follow-up question; its method grants no execution authority."""

    model_config = ConfigDict(extra="forbid", strict=True)
    objective: str
    method: str


def validate_proposed_check(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"objective", "method"}:
        raise ValueError("proposed check shape is invalid")
    if type(value["objective"]) is not str or "\x00" in value["objective"]:
        raise ValueError("proposed check objective is invalid")
    _require_text("proposed check objective", value["objective"], max_bytes=64 * 1024)
    if type(value["method"]) is not str or _IDENTIFIER.fullmatch(value["method"]) is None:
        raise ValueError("proposed check method is invalid")


def validate_check_plan(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"schema", "method", "phase", "objective"}:
        raise ValueError("check plan shape is invalid")
    validate_proposed_check({"method": value["method"], "objective": value["objective"]})
    if (
        value["schema"] != FINDING_CHECK_PLAN_SCHEMA
        or value["method"] not in METHOD_PHASES
        or value["phase"] != METHOD_PHASES[value["method"]]
    ):
        raise ValueError("check plan method or phase is unregistered")
