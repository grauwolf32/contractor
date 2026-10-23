"""Pinned argument bindings for a deterministic, model-free Worker."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from contractor_runtime.contracts.base import WireModel, trim_whitespace

ARGUMENT_NAME = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"


def binding_name(value: str) -> bool:
    return (
        0 < len(value.encode("utf-8")) <= 128
        and value == trim_whitespace(value)
        and not any(char in value for char in "\r\n\t\x00")
    )


class ToolArgumentBinding(WireModel):
    source: Literal["parameter", "artifact", "literal"]
    name: str | None = Field(default=None, exclude_if=lambda value: value is None)
    value: Any = Field(default=None, exclude_if=lambda value: value is None)

    parse: Literal["integer"] | None = Field(default=None, exclude_if=lambda value: value is None)
    default: Any = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        options = self.model_fields_set & {"parse", "default"}
        if self.source != "parameter" and options:
            raise ValueError("parse and default require a parameter binding")
        if "parse" in options and self.parse is None:
            raise ValueError("tool parameter parser cannot be null")
        if "default" in options:
            if self.parse == "integer":
                self._integer_default()
            elif not isinstance(self.default, str) or len(self.default.encode("utf-8")) > 8192:
                raise ValueError("unparsed parameter default must be a bounded string")
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
                if abs(value) > 9007199254740991 or not math.isfinite(value):
                    raise ValueError("literal tool number exceeds its bound")
            else:
                raise ValueError("literal tool arguments must be non-null scalars")
        return self

    def _integer_default(self) -> int:
        value = self.default
        if (
            type(value) not in {int, float}
            or not 0 <= value <= 9007199254740991
            or int(value) != value
        ):
            raise ValueError("integer tool parameter default must be an exact nonnegative integer")
        return int(value)

    def resolve_parameter(self, parameters: Mapping[str, str]) -> Any:
        """Project one parameter binding; explicit empty text never selects a default."""
        if self.source != "parameter" or self.name is None:
            raise ValueError("invalid parameter tool argument")
        if self.name not in parameters:
            if "default" not in self.model_fields_set:
                raise ValueError("required bound tool parameter is missing")
            return self._integer_default() if self.parse == "integer" else self.default
        value = parameters[self.name]
        if self.parse is None:
            return value
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 16
            or re.fullmatch(r"0|[1-9][0-9]*", value) is None
        ):
            raise ValueError("tool parameter must be a canonical unsigned integer")
        number = int(value)
        if number > 9007199254740991:
            raise ValueError("tool parameter exceeds the exact integer bound")
        return number


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
