"""Validate model arguments before constructing canonical Audit results."""

from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict

from contractor_runtime.toolsets.audit_results.contracts import IDENTIFIER, MAX_VALUES
from contractor_runtime.toolsets.common.input_errors import ToolInputError


class EvidenceArgument(TypedDict):
    kind: str
    summary: str


class BatchResultArgument(TypedDict):
    assessment: str
    summary: str
    completed: list[str]
    gaps: list[str]
    evidence: NotRequired[list[EvidenceArgument]]
    proposal_keys: NotRequired[list[str]]


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


def identifier_list(
    field: str,
    value: Any,
    *,
    maximum: int = MAX_VALUES,
    allowed: Sequence[str] | None = None,
) -> list[str]:
    values = array_argument(field, value, maximum)
    for index, entry in enumerate(values):
        if not isinstance(entry, str) or IDENTIFIER.fullmatch(entry) is None:
            raise AuditArgumentError(
                field,
                "Use identifiers of at most 160 characters: letters, digits, '.', '_', ':', "
                "'-'; start with a letter or digit.",
                index=index,
            )
        if allowed is not None and entry not in allowed:
            raise AuditArgumentError(
                field,
                "Value is not requested by the assigned task. Use only allowedValues; "
                "do not mark unverified coverage completed.",
                index=index,
                invalidValue=entry,
                allowedValues=list(allowed),
            )
    return sorted(set(values))
