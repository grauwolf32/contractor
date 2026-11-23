"""Bounded content-free USD observations shared with Go execution reports."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER, WireModel

USD_AMOUNT_PATTERN = r"^(0|[1-9][0-9]{0,17})(\.[0-9]{0,17}[1-9])?$"
MAX_LLM_COST_GROUPS = 64
CostPhase = Literal["main", "compaction", "finalization", "summarization"]
CostSource = Literal["provider", "gateway"]


class LLMCostGroup(WireModel):
    model: str
    phase: CostPhase
    source: CostSource
    calls: Annotated[int, Field(strict=True, ge=1, le=MAX_EXACT_INTEGER)]
    amount: str

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if (
            not value
            or value.strip() != value
            or len(value.encode("utf-8")) > 256
            or any(unicodedata.category(c) == "Cc" for c in value)
        ):
            raise ValueError("LLM cost model is invalid")
        return value

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: str) -> str:
        if re.fullmatch(USD_AMOUNT_PATTERN, value) is None:
            raise ValueError("LLM cost amount is not canonical")
        return value


class LLMCost(WireModel):
    currency: Literal["USD"]
    calls: Annotated[int, Field(strict=True, ge=0, le=MAX_EXACT_INTEGER)]
    groups: Annotated[list[LLMCostGroup], Field(max_length=MAX_LLM_COST_GROUPS)]

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        keys = [(g.model, g.phase, g.source) for g in self.groups]
        if len(set(keys)) != len(keys) or sum(g.calls for g in self.groups) > self.calls:
            raise ValueError("LLM cost groups do not partition priced calls")
        return self
