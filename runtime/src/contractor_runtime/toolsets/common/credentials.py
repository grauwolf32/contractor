"""Credentials Runtime injects into project target, proxy and Gateway requests.

The model never sees these header values. A toolset that captures or relays a
request carrying one retains a marker in its place.
"""

from __future__ import annotations

import base64

from contractor_runtime.contracts import RuntimeSettings

# Retained in place of the project target credential that Runtime injects.
RUNTIME_TARGET_CREDENTIAL = "[runtime-target-credential]"
# Retained in place of a Runtime infrastructure (proxy or Gateway) credential.
RUNTIME_CREDENTIAL = "[runtime-credential]"


def target_authorization(settings: RuntimeSettings) -> str | None:
    """The Authorization value http_request injects for the project HTTP target."""

    target = settings.http_origin_target
    if target is None:
        return None
    if target.bearer_token is not None:
        return f"Bearer {target.bearer_token.get_secret_value()}"
    if target.basic_auth is not None:
        return _basic_authorization(
            target.basic_auth.username.get_secret_value(),
            target.basic_auth.password.get_secret_value(),
        )
    return None


def proxy_authorization(settings: RuntimeSettings) -> str | None:
    """The Proxy-Authorization value every proxied route sends to the forward proxy."""

    proxy = settings.http_proxy
    if proxy is None:
        return None
    if proxy.bearer_token is not None:
        return f"Bearer {proxy.bearer_token.get_secret_value()}"
    if proxy.basic_auth is not None:
        return _basic_authorization(
            proxy.basic_auth.username.get_secret_value(),
            proxy.basic_auth.password.get_secret_value(),
        )
    return None


def gateway_authorization(settings: RuntimeSettings) -> str | None:
    """The Authorization value the LLM Gateway client sends, also when proxied."""

    token = settings.llm_gateway_token
    if token is None or not token.get_secret_value():
        return None
    return f"Bearer {token.get_secret_value()}"


def _basic_authorization(username: str, password: str) -> str:
    # httpx, httpcore and the scanner proxy clients encode user:password as UTF-8.
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {token}"
