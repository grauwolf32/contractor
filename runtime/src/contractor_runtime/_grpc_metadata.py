"""Bounded correlation metadata for private native RPC responses."""

from __future__ import annotations

import uuid

from contractor_runtime._https import REQUEST_ID_PATTERN


async def response_request_id(context) -> str:
    identifiers = [value for key, value in context.invocation_metadata() if key == "x-request-id"]
    request_id = (
        identifiers[0]
        if len(identifiers) == 1 and REQUEST_ID_PATTERN.fullmatch(identifiers[0])
        else "request_" + uuid.uuid4().hex
    )
    await context.send_initial_metadata((("x-request-id", request_id),))
    return request_id
