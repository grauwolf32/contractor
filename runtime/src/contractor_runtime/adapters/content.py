"""Bounded content capture for configured telemetry exporters."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

MAX_CONTENT_BYTES = 256 * 1024


def _json_value(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True, by_alias=True)
    raise TypeError("unsupported telemetry content")


def encode_content(value: Any) -> str:
    # Limit the retained serialization, preserving valid JSON even on truncation.
    parts: list[bytes] = []
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, default=_json_value)
    for chunk in encoder.iterencode(value):
        raw = chunk.encode("utf-8")
        if size + len(raw) > MAX_CONTENT_BYTES:
            parts.append(raw[: MAX_CONTENT_BYTES - size])
            preview = b"".join(parts).decode("utf-8", errors="ignore")
            while True:
                result = json.dumps({"truncated": True, "preview": preview}, ensure_ascii=False)
                if len(result.encode("utf-8")) <= MAX_CONTENT_BYTES:
                    return result
                preview = preview[: len(preview) // 2]
        parts.append(raw)
        size += len(raw)
    return b"".join(parts).decode("utf-8")


def capture_span_content(
    span: Any,
    *,
    input: Callable[[], Any] | None = None,
    output: Callable[[], Any] | None = None,
) -> None:
    # Lazy access is important: disabled capture must not inspect/serialize data.
    if not callable(getattr(span, "set_content", None)):
        return
    try:
        encode = getattr(span, "encode_content", encode_content)
        if input is not None:
            span.set_content("input", encode(input()))
        if output is not None:
            span.set_content("output", encode(output()))
    except Exception:
        # Optional telemetry cannot change model/tool execution outcomes.
        return


def model_request_content(request: Any) -> dict[str, Any]:
    config = getattr(request, "config", None)
    return {
        "systemInstruction": getattr(config, "system_instruction", None),
        "contents": getattr(request, "contents", None),
        "tools": getattr(config, "tools", None),
    }
