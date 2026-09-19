"""Pinned argument bindings for a deterministic, model-free Worker."""

from __future__ import annotations

import math
import re
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from contractor_runtime.contracts.base import WireModel

ARGUMENT_NAME = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"


def binding_name(value: str) -> bool:
    return (
        0 < len(value.encode("utf-8")) <= 128
        and value == value.strip()
        and not any(char in value for char in "\r\n\t\x00")
    )


class ToolArgumentBinding(WireModel):
    source: Literal["parameter", "artifact", "literal"]
    name: str | None = Field(default=None, exclude_if=lambda value: value is None)
    value: Any = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.source != "literal":
            if self.name is None or not binding_name(self.name) or "value" in self.model_fields_set:
                raise ValueError("invalid tool argument source binding")
        else:
            if "name" in self.model_fields_set:
                raise ValueError("literal tool argument must not have name")
            value = self.value
            if isinstance(value, str):
                if len(value.encode("utf-8")) > 8192:
                    raise ValueError("literal tool string exceeds its bound")
            elif type(value) is bool:
                pass
            elif type(value) in {int, float}:
                if not math.isfinite(value) or abs(value) > 9007199254740991:
                    raise ValueError("literal tool number exceeds its bound")
            else:
                raise ValueError("literal tool arguments must be non-null scalars")
        return self


class ToolExecutionConfig(WireModel):
    tool: str = Field(pattern=ARGUMENT_NAME)
    arguments: dict[str, ToolArgumentBinding] = Field(max_length=32)
    result_artifact: str
    timeout_seconds: int = Field(ge=1, le=3600, strict=True)

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        if not binding_name(self.result_artifact) or any(
            re.fullmatch(ARGUMENT_NAME, name) is None for name in self.arguments
        ):
            raise ValueError("invalid tool argument or result binding name")
        return self
