"""Shared result envelope for agent-facing (frontend) tools.

Every frontend tool returns one of two shapes so the LLM has a single,
uniform way to read an outcome across the whole tool surface:

- success: ``{"result": <value>, **meta}``
- failure: ``{"error": "<message>", **meta}``

These two never mix in one dict — the presence of ``result`` vs ``error`` is
the success signal, so success metadata must never ride along on an error.

Why a *called helper* and not a decorator: ADK introspects each tool with
``inspect.signature`` + ``typing.get_type_hints`` to build the function-call
schema sent to the model. A decorator that forgets ``functools.wraps`` silently
strips every parameter from that schema (the model then sees an argument-less
tool). Wrapping the body via ``guard(lambda: ...)`` keeps the frontend function
a plain ``def`` whose real signature is exactly what ADK reads — immune to that
failure mode regardless of ADK version or feature flags. Keep backends Pythonic
(raise exceptions / return raw values); ``guard`` is the bridge that adapts them
to the envelope.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["ok", "err", "guard", "is_envelope"]


def ok(result: Any = None, **meta: Any) -> dict[str, Any]:
    """Build a success envelope: ``{"result": result, **meta}``.

    ``meta`` carries optional companion fields such as ``total_items``,
    ``kind``, ``offset``, ``truncated`` — never an ``error`` key.
    """
    return {"result": result, **meta}


def err(message: str, **meta: Any) -> dict[str, Any]:
    """Build an error envelope: ``{"error": message, **meta}``.

    Must not carry success keys (``result``/``total_items``/…): the caller
    distinguishes outcomes purely by which key is present.
    """
    return {"error": message, **meta}


def is_envelope(value: Any) -> bool:
    """True if ``value`` is already a ``result``/``error`` envelope dict."""
    return isinstance(value, dict) and ("result" in value or "error" in value)


def guard(thunk: Callable[[], Any]) -> dict[str, Any]:
    """Run a tool body and normalize its outcome to the result/error envelope.

    - returns an envelope dict unchanged (the body built its own, e.g. with
      ``ok(...)`` carrying ``total_items``/``kind``);
    - wraps any other return value as ``{"result": value}``;
    - turns a raised exception into ``{"error": str(exc)}`` so a backend fault
      surfaces to the model as a clean error instead of an opaque tool crash.

    Call it from inside a plain frontend ``def`` — ``return guard(lambda:
    backend(...))`` — so the function keeps the real signature ADK introspects.
    """
    try:
        out = thunk()
    except Exception as exc:  # noqa: BLE001 - deliberate: any fault -> error envelope
        return err(str(exc))
    if is_envelope(out):
        return out
    return ok(out)
