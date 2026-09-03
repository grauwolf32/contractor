"""Allocation-local LLM Gateway client construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from contractor_runtime.adapters.http_proxy import ProxyHTTPClient

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext


def gateway_client_options(context: WorkerBuildContext) -> dict[str, Any]:
    """Build LiteLLM options without applying any process-global proxy state."""

    settings = context.runtime_settings
    token = settings.llm_gateway_token
    token_value = token.get_secret_value() if token is not None else ""
    timeout = float(settings.request_timeout_seconds)
    handle = context.adapter_handles.model_http
    if handle is None:
        result: dict[str, Any] = {
            "api_base": settings.llm_gateway_url,
            "timeout": timeout,
            # A Worker/summarizer model call is the retry boundary. Hidden
            # provider retries would duplicate an accepted request without
            # consuming another explicit ModelPolicy call.
            "num_retries": 0,
        }
        if token_value:
            result["api_key"] = token_value
        return result
    if not isinstance(handle, ProxyHTTPClient):
        raise TypeError("llm-gateway proxy handle has an invalid type")

    client = AsyncOpenAI(
        api_key=token_value or "contractor-no-token",
        base_url=settings.llm_gateway_url,
        timeout=timeout,
        max_retries=0,
        http_client=handle.async_client,
    )
    return {
        "api_base": settings.llm_gateway_url,
        "timeout": timeout,
        "num_retries": 0,
        "client": client,
    }


def clear_gateway_client_options(options: dict[str, Any]) -> None:
    """Erase token-bearing LiteLLM/OpenAI references without closing adapter HTTP."""

    client = options.pop("client", None)
    if isinstance(client, AsyncOpenAI):
        client.api_key = ""
    options.clear()
