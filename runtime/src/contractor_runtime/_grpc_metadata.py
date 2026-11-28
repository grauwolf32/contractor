"""Bounded correlation metadata for private native RPC responses."""

from __future__ import annotations

import re
import uuid

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def error_detail(value):
    """Validate only the named native fields; peer diagnostics never enter them."""
    if (
        not isinstance(value, dict)
        or set(value) != {"code", "retryable", "requestId"}
        or not isinstance(value["code"], str)
        or ERROR_CODE_PATTERN.fullmatch(value["code"]) is None
        or type(value["retryable"]) is not bool
        or not isinstance(value["requestId"], str)
        or REQUEST_ID_PATTERN.fullmatch(value["requestId"]) is None
    ):
        raise ValueError("invalid native error detail")
    return value["code"], value["retryable"], value["requestId"]


async def response_request_id(context) -> str:
    identifiers = [value for key, value in context.invocation_metadata() if key == "x-request-id"]
    request_id = (
        identifiers[0]
        if len(identifiers) == 1 and REQUEST_ID_PATTERN.fullmatch(identifiers[0])
        else "request_" + uuid.uuid4().hex
    )
    await context.send_initial_metadata((("x-request-id", request_id),))
    return request_id
