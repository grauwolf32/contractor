"""Normalize Gateway cost at the physical request boundary, before ADK conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation, localcontext
from threading import Lock
from typing import Any

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER
from contractor_runtime.contracts.llm_cost import (
    MAX_LLM_COST_GROUPS,
    CostPhase,
    LLMCost,
    LLMCostGroup,
)

_PHASE: ContextVar[CostPhase] = ContextVar("llm_cost_phase", default="main")
_NUMBER = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]{1,2})?")


@contextmanager
def model_cost_phase(phase: CostPhase) -> Iterator[None]:
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)


def normalize_usd(raw: str) -> str | None:
    if len(raw) > 80 or _NUMBER.fullmatch(raw) is None:
        return None
    try:
        with localcontext() as context:
            context.prec = 80
            value = Decimal(raw).normalize()
            if value < 0 or value >= Decimal("1e18") or value.as_tuple().exponent < -18:
                return None
            result = format(value, "f")
            return result.rstrip("0").rstrip(".") if "." in result else result
    except (InvalidOperation, ValueError):
        return None


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate Gateway metadata member")
        result[key] = value
    return result


def response_cost(model: str, headers: Mapping[str, str], body: bytes) -> tuple[str, str] | None:
    provider = headers.get("llm_provider-x-litellm-response-cost", "")
    if model.startswith("openrouter/") and body:
        try:
            document = json.loads(body, parse_float=Decimal, object_pairs_hook=_unique_pairs)
            usage = document.get("usage") if isinstance(document, dict) else None
        except (ValueError, RecursionError, UnicodeError):
            usage = None
        if isinstance(usage, dict):
            if usage.get("is_byok") is True:
                return None
            if usage.get("cost") is not None:
                raw = usage["cost"]
                if not isinstance(raw, (Decimal, int)) or isinstance(raw, bool):
                    return None
                amount = normalize_usd(str(raw))
                if amount is None or (provider and normalize_usd(provider) != amount):
                    return None
                return "provider", amount
    if provider:
        amount = normalize_usd(provider)
        return ("provider", amount) if amount is not None else None
    amount = normalize_usd(headers.get("x-litellm-response-cost", ""))
    return ("gateway", amount) if amount is not None else None


class CostCollector:
    """One allocation's bounded counters, shared by all of its model phases."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._calls = 0
        self._groups: dict[tuple[str, str, str], LLMCostGroup] = {}

    def observe(self, model: str, headers: Mapping[str, str], body: bytes) -> None:
        # Called from the transport thread; copy_context preserves the phase.
        priced = response_cost(model, headers, body)
        with self._lock:
            if self._calls == MAX_EXACT_INTEGER:
                return
            self._calls += 1
            if priced is None:
                return
            source, amount = priced
            phase = _PHASE.get()
            key = (model, phase, source)
            prior = self._groups.get(key)
            if prior is not None:
                with localcontext() as context:
                    context.prec = 80
                    total = normalize_usd(format(Decimal(prior.amount) + Decimal(amount), "f"))
                if total is not None:
                    prior.amount = total
                    prior.calls += 1
                return
            if len(self._groups) >= MAX_LLM_COST_GROUPS:
                return
            try:
                group = LLMCostGroup(
                    model=model, phase=phase, source=source, calls=1, amount=amount
                )
            except ValueError:
                return
            self._groups[key] = group

    def snapshot(self, minimum_calls: int) -> LLMCost:
        with self._lock:
            return LLMCost(
                currency="USD",
                calls=min(MAX_EXACT_INTEGER, max(self._calls, minimum_calls)),
                groups=[self._groups[k].model_copy(deep=True) for k in sorted(self._groups)],
            )
