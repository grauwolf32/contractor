"""Validate model arguments before constructing canonical Audit results."""

from typing import Any, NotRequired, TypedDict

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.audit_results.contracts import MAX_COLLECTED_BYTES, AuditVerdict
from contractor_runtime.toolsets.common.arguments import (
    MAX_CONTAINER_DEPTH,
    MAX_CONTAINER_NODES,
    container_argument,
    validate_json,
)


class BatchResultArgument(TypedDict):
    item_key: str
    rationale: str
    tested_conditions: list[str] | str
    limitations: list[str] | str
    evidence: NotRequired[list[str] | str]
    verdict: NotRequired[AuditVerdict]
    http_request_ids: NotRequired[list[int] | str]


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
    value, _ = container_argument(
        value,
        field=field,
        kinds=(list,),
        max_bytes=MAX_COLLECTED_BYTES,
        max_depth=MAX_CONTAINER_DEPTH,
        max_nodes=MAX_CONTAINER_NODES,
        invalid=AuditArgumentError,
    )
    if len(value) > maximum:
        raise AuditArgumentError(
            field,
            f"Provide at most {maximum} array entries.",
            limit=maximum,
            actualCount=len(value),
        )
    return value


def admit_arguments(value: Any) -> None:
    # Single-call envelope and item record add at most two container levels.
    # The aggregate admission limits bytes before any per-field parsing. Its
    # node bound follows from that byte bound, rather than multiplying it by
    # the number of independently valid fields.
    validate_json(
        value,
        field="arguments",
        max_bytes=MAX_COLLECTED_BYTES,
        max_depth=MAX_CONTAINER_DEPTH + 2,
        max_nodes=MAX_COLLECTED_BYTES,
        invalid=AuditArgumentError,
    )
