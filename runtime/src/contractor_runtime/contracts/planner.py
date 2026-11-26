"""Transport-neutral Planner invocation and execution-host messages."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator

from contractor_runtime.contracts.base import (
    MAX_EXACT_INTEGER,
    NAME_PATTERN,
    TerminationError,
    VersionedWireModel,
    WireModel,
    _require_text,
    json_byte_size,
)
from contractor_runtime.contracts.settings import (
    ResolvedModelPolicy,
    _require_worker_compactor_policy,
)

MAX_PLANNER_OPERATION_BYTES = 2 << 20
MAX_PLANNER_INVOCATION_BYTES = 512 * 1024
MAX_PLANNER_TOOLS = 64
MAX_PLANNER_OPERATIONS = 5001


class PlannerTool(WireModel):
    name: str = Field(pattern="^" + NAME_PATTERN.pattern + "$")
    description: str
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def validate_description(self) -> Self:
        _require_text("description", self.description)
        return self


class PlannerCompaction(WireModel):
    instructions: str = Field(min_length=1, max_length=8000)
    threshold: int = Field(ge=1, le=MAX_EXACT_INTEGER)
    keep_recent_groups: int = Field(ge=1, le=16)
    max_compactions: int = Field(ge=1, le=16)
    policy: ResolvedModelPolicy

    @model_validator(mode="after")
    def validate_compaction(self) -> Self:
        _require_text("instructions", self.instructions)
        _require_worker_compactor_policy(self.policy)
        return self


class PlannerInvocation(WireModel):
    input: str
    tools: list[PlannerTool] = Field(min_length=1, max_length=MAX_PLANNER_TOOLS)
    compaction: PlannerCompaction | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_invocation(self) -> Self:
        _require_text("input", self.input)
        if len({tool.name for tool in self.tools}) != len(self.tools):
            raise ValueError("duplicate Planner tool")
        return self


class PlannerOperation(VersionedWireModel):
    sequence: int = Field(ge=1, le=MAX_PLANNER_OPERATIONS)
    name: str = Field(pattern="^" + NAME_PATTERN.pattern + "$")
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        if json_byte_size(self.model_dump(by_alias=True)) > MAX_PLANNER_OPERATION_BYTES:
            raise ValueError("Planner operation exceeds its bound")
        return self


class PlannerOperationResult(VersionedWireModel):
    result: dict[str, Any]
    stop: bool
    escalate: bool
    failure: TerminationError | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.escalate and not self.stop:
            raise ValueError("Planner escalation requires stop")
        if json_byte_size(self.model_dump(by_alias=True)) > MAX_PLANNER_OPERATION_BYTES:
            raise ValueError("Planner operation result exceeds its bound")
        return self
