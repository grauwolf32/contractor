"""Private Runtime protocol codec models and validation."""

from __future__ import annotations

import json
from typing import Any, Literal

import jcs
from pydantic import (
    ValidationError,
)

from contractor_runtime.contracts.base import PRIVATE_PROTOCOL_VERSION_V2, WireModel
from contractor_runtime.contracts.registration import (
    AgentRegistrationResponseV2,
    AgentRegistrationV2,
)
from contractor_runtime.contracts.reports import AllocationFinalResponse


class PrivateProtocolDecodeError(ValueError):
    """Bounded private-wire failure whose rendering never includes input."""

    def __init__(self, reason: Literal["version", "duplicate_key", "schema", "invariant"]):
        self.reason = reason
        super().__init__(f"private protocol v2 {reason} error")


class _DuplicateJSONKey(ValueError):
    pass


def _invalid_json_constant(_value: str) -> Any:
    raise ValueError("non-standard JSON constant")


def encode_private_v2(value: WireModel) -> bytes:
    """Return RFC 8785 canonical private JSON; callers must not log it."""

    dumped = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return jcs.canonicalize(dumped)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def decode_private_v2[PrivateModelT: WireModel](
    model: type[PrivateModelT], raw: str | bytes
) -> PrivateModelT:
    """Decode one secret-bearing v2 document without reflecting input in errors."""

    if (
        issubclass(model, AllocationFinalResponse)
        and len(raw.encode("utf-8") if isinstance(raw, str) else raw) > 1024 * 1024
    ):
        raise PrivateProtocolDecodeError("schema")
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_invalid_json_constant
        )
    except _DuplicateJSONKey:
        raise PrivateProtocolDecodeError("duplicate_key") from None
    except (ValueError, UnicodeDecodeError, TypeError):
        raise PrivateProtocolDecodeError("schema") from None
    if not isinstance(value, dict):
        raise PrivateProtocolDecodeError("schema")
    if (
        issubclass(model, (AgentRegistrationV2, AgentRegistrationResponseV2))
        and value.get("privateProtocolVersion") != PRIVATE_PROTOCOL_VERSION_V2
    ):
        raise PrivateProtocolDecodeError("version")
    try:
        # Keep Pydantic's strict JSON conversions (notably RFC 3339 strings to
        # aware datetimes) after the duplicate-key pre-scan above.
        return model.model_validate_json(raw)
    except ValidationError as error:
        reason: Literal["schema", "invariant"] = "invariant"
        if any(item["type"] != "value_error" for item in error.errors(include_input=False)):
            reason = "schema"
        raise PrivateProtocolDecodeError(reason) from None
