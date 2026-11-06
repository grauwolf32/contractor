"""Validate model arguments before constructing canonical Audit results."""

from typing import Any, NotRequired, TypedDict

from contractor_runtime.core.input_errors import ToolInputError


class BatchResultArgument(TypedDict):
    rationale: str
    tested_conditions: list[str]
    limitations: list[str]
    evidence: NotRequired[list[str]]
    verdict: NotRequired[str]
    http_request_ids: NotRequired[list[int]]


class AuditArgumentError(ToolInputError):
    code = "audit_result_invalid"

    def __init__(self, field: str, reason: str, **details: Any):
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "field": self.field, "message": self.reason, **self.details}


def array_argument(field: str, value: Any, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise AuditArgumentError(
            field,
            "Provide a JSON array (including []), not a string containing JSON.",
            expectedType="array",
        )
    if len(value) > maximum:
        raise AuditArgumentError(
            field,
            f"Provide at most {maximum} array entries.",
            limit=maximum,
            actualCount=len(value),
        )
    return value
