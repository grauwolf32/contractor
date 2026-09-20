"""Closed, allocation-only toolset settings. No endpoint IO during decoding."""

from __future__ import annotations

import ssl
from typing import Literal, Self

from pydantic import Field, SecretStr, field_serializer, model_validator

from contractor_runtime.contracts.base import (
    _FORBIDDEN_RUNTIME_HEADERS,
    _HEADER_NAME_PATTERN,
    WireModel,
    _require_runtime_endpoint,
)

MCP_TOOLSET_REF = "mcp@1"
MAXIMUM_CONFIGURED_TOOLSETS = 32
MCP_RESERVED_HEADERS = _FORBIDDEN_RUNTIME_HEADERS | {
    "accept",
    "content-type",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
}


class MCPToolsetSettings(WireModel):
    endpoint: str
    headers: dict[str, SecretStr] = Field(default_factory=dict, exclude_if=lambda value: not value)
    ca_bundle_pem: str | None = None
    connect_timeout_seconds: int = Field(ge=1, le=60)
    request_timeout_seconds: int = Field(ge=1, le=600)

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        _require_runtime_endpoint("MCP endpoint", self.endpoint)
        if "?" in self.endpoint or "#" in self.endpoint:
            raise ValueError("invalid MCP endpoint")
        if self.ca_bundle_pem is not None:
            # Local import avoids a cycle with RuntimeSettings.
            from contractor_runtime.contracts.settings import _validate_ca_bundle

            _validate_ca_bundle("MCP", self.ca_bundle_pem)
            try:
                ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(
                    cadata=self.ca_bundle_pem
                )
            except (ValueError, ssl.SSLError):
                raise ValueError("MCP CA bundle is invalid") from None
        if len(self.headers) > 32:
            raise ValueError("too many MCP headers")
        seen: set[str] = set()
        total = 0
        for name, secret in self.headers.items():
            lower = name.lower()
            if (
                not 1 <= len(name.encode("utf-8")) <= 64
                or not _HEADER_NAME_PATTERN.fullmatch(name)
                or lower in seen
                or lower in MCP_RESERVED_HEADERS
            ):
                raise ValueError("invalid MCP header name")
            seen.add(lower)
            value = secret.get_secret_value()
            if not 1 <= len(value.encode("utf-8")) <= 4096 or "\r" in value or "\n" in value:
                raise ValueError("invalid MCP header value")
            total += len(name.encode("utf-8")) + len(value.encode("utf-8"))
        if total > 16 * 1024:
            raise ValueError("MCP headers exceed size bound")
        return self

    @field_serializer("headers", when_used="json")
    def serialize_headers(self, value: dict[str, SecretStr]) -> dict[str, str]:
        return {name: secret.get_secret_value() for name, secret in value.items()}


# This union grows only with code-owned schema/factory registrations. Ref keys
# select the schema; configuration documents cannot register new decoders.
ToolsetRuntimeSettings = dict[Literal["mcp@1"], MCPToolsetSettings]
