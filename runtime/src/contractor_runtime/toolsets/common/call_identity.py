"""Stable provider-event identity used by allocation-bound keyed mutations."""

import hashlib

from contractor_runtime.core.input_errors import ToolInputError


def call_client_key(context) -> str:
    call_id = getattr(context, "function_call_id", None)
    if not isinstance(call_id, str) or not call_id:
        raise ToolInputError("Runtime tool-call identity is unavailable; nothing was submitted")
    return "call-" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()
