"""Request header grammar, case-insensitive merging and aggregate admission."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from contractor_runtime.contracts.fields import is_field_value
from contractor_runtime.core.http_limits import (
    MAX_HEADER_VALUE_BYTES,
    MAX_HEADERS,
    header_block_bytes,
    request_header_max_bytes,
)
from contractor_runtime.toolsets.http.errors import HTTPToolError

_FORBIDDEN_REQUEST_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-request-id",
    }
)
_HEADER_NAME = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def validate_headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping) or len(headers) > MAX_HEADERS:
        raise HTTPToolError("http_request_invalid")
    result: dict[str, str] = {}
    seen: set[str] = set()
    total = 0
    for name, raw_value in headers.items():
        try:
            encoded_name = name.encode("ascii") if isinstance(name, str) else b""
        except UnicodeEncodeError:
            raise HTTPToolError("http_request_invalid") from None
        if not isinstance(name, str) or _HEADER_NAME.fullmatch(encoded_name) is None:
            raise HTTPToolError("http_request_invalid")
        normalized = name.lower()
        if normalized in seen or normalized in _FORBIDDEN_REQUEST_HEADERS:
            raise HTTPToolError("http_request_invalid")
        if not isinstance(raw_value, str) or not is_field_value(raw_value):
            raise HTTPToolError("http_request_invalid")
        size = request_utf8_size(raw_value)
        if size > MAX_HEADER_VALUE_BYTES:
            raise HTTPToolError("http_request_invalid")
        total += len(name) + size
        if total > request_header_max_bytes():
            raise HTTPToolError("http_request_invalid")
        seen.add(normalized)
        result[name] = raw_value
    return result


def merge_headers(defaults: Mapping[str, str], request: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, tuple[str, str]] = {
        name.lower(): (name, value) for name, value in defaults.items()
    }
    for name, value in request.items():
        result[name.lower()] = (name, value)
    if len(result) > MAX_HEADERS:
        raise HTTPToolError("http_request_invalid")
    if header_block_bytes(result.values()) > request_header_max_bytes():
        raise HTTPToolError("http_request_invalid")
    return {name: value for name, value in result.values()}


def request_utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeError:
        raise HTTPToolError("http_request_invalid") from None


def body_headers(
    defaults: Mapping[str, str], request: Mapping[str, str], content_type: str | None
) -> dict[str, str]:
    selected = dict(request)
    if content_type is not None and "content-type" not in {name.lower() for name in selected}:
        selected["Content-Type"] = content_type
    return merge_headers(defaults, selected)
