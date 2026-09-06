"""Allocation-local OpenAI-compatible LLM Gateway client construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI

from contractor_runtime.adapters.http_proxy import ProxyHTTPClient

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext

OPENAI_GATEWAY_MAX_RETRIES = 3
OPENAI_GATEWAY_RETRY_GRACE_SECONDS = 60.0


class GatewayClientClosedError(RuntimeError):
    """Raised without connection details when an allocation client is closed."""


@dataclass(slots=True)
class GatewayClientHandle:
    """Own one OpenAI client while respecting adapter HTTP-client ownership."""

    _client: AsyncOpenAI | None
    _owns_http_client: bool
    operation_timeout_seconds: float

    @property
    def client(self) -> AsyncOpenAI:
        client = self._client
        if client is None:
            raise GatewayClientClosedError("LLM Gateway client is closed")
        return client

    @property
    def closed(self) -> bool:
        return self._client is None

    def __repr__(self) -> str:
        return (
            "GatewayClientHandle("
            f"closed={self.closed!r}, owns_http_client={self._owns_http_client!r}, "
            f"operation_timeout_seconds={self.operation_timeout_seconds!r})"
        )

    async def close(self) -> None:
        """Erase the credential and close only a directly owned HTTP client."""

        client = self._client
        owns_http_client = self._owns_http_client
        self._client = None
        self._owns_http_client = False
        if client is None:
            return
        client.api_key = ""
        if owns_http_client:
            await client.close()


def new_gateway_client(
    *,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None = None,
) -> GatewayClientHandle:
    """Construct a bounded-retry client for one immutable allocation route."""

    owns_http_client = http_client is None
    if http_client is None:
        # Process-global HTTP(S)_PROXY and trust-store variables are not an
        # allocation configuration channel. Explicit Runtime adapters supply
        # their own client when proxy routing is selected.
        http_client = httpx.AsyncClient(timeout=timeout_seconds, trust_env=False)
    client = AsyncOpenAI(
        api_key=api_key or "contractor-no-token",
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=OPENAI_GATEWAY_MAX_RETRIES,
        http_client=http_client,
    )
    return GatewayClientHandle(
        client,
        owns_http_client,
        timeout_seconds + OPENAI_GATEWAY_RETRY_GRACE_SECONDS,
    )


def build_gateway_client(context: WorkerBuildContext) -> GatewayClientHandle:
    """Resolve the exact allocation settings into an isolated client handle."""

    settings = context.runtime_settings
    token = settings.llm_gateway_token
    token_value = token.get_secret_value() if token is not None else None
    timeout = float(settings.request_timeout_seconds)
    adapter_handle = context.adapter_handles.model_http
    if adapter_handle is None:
        return new_gateway_client(
            base_url=settings.llm_gateway_url,
            api_key=token_value,
            timeout_seconds=timeout,
        )
    if not isinstance(adapter_handle, ProxyHTTPClient):
        raise TypeError("llm-gateway proxy handle has an invalid type")
    return new_gateway_client(
        base_url=settings.llm_gateway_url,
        api_key=token_value,
        timeout_seconds=timeout,
        http_client=adapter_handle.async_client,
    )
