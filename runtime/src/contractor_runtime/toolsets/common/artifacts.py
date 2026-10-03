"""Artifact client construction and credential redaction helpers for toolsets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import RuntimeSettings

ArtifactClientFactory = Callable[[str, RuntimeSettings], ArtifactClient]


def runtime_secrets(settings: RuntimeSettings) -> tuple[str, ...]:
    """Only the RuntimeSettings credentials, without endpoints or CA bundles.

    Worker results may legitimately name an endpoint (a same-host deployment
    audits services next to its own), but never a credential.
    """

    values: list[str] = []
    token = settings.llm_gateway_token
    if token is not None:
        values.append(token.get_secret_value())
    if settings.telemetry is not None:
        values.extend(secret.get_secret_value() for secret in settings.telemetry.headers.values())
    if settings.http_proxy is not None:
        proxy = settings.http_proxy
        if proxy.basic_auth is not None:
            values.extend(
                (
                    proxy.basic_auth.username.get_secret_value(),
                    proxy.basic_auth.password.get_secret_value(),
                )
            )
        if proxy.bearer_token is not None:
            values.append(proxy.bearer_token.get_secret_value())
    if settings.caido is not None and settings.caido.bearer_token is not None:
        values.append(settings.caido.bearer_token.get_secret_value())
    if settings.http_origin_target is not None:
        # The target URL is the audited application and legitimately appears in
        # results; only the credentials Runtime injects for it are private.
        target = settings.http_origin_target
        if target.basic_auth is not None:
            values.extend(
                (
                    target.basic_auth.username.get_secret_value(),
                    target.basic_auth.password.get_secret_value(),
                )
            )
        if target.bearer_token is not None:
            values.append(target.bearer_token.get_secret_value())
    return tuple(value for value in values if value)


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    """Default factory whose client fails on its first request."""

    return ArtifactClient(allocation_id, _UnavailableTransport())


def _reject_unconfigured_client(
    allocation_id: str, runtime_settings: RuntimeSettings
) -> ArtifactClient:
    """Default factory that fails when the toolset creates its tools."""

    del allocation_id, runtime_settings
    raise RuntimeError(_UNCONFIGURED_MESSAGE)


_UNCONFIGURED_MESSAGE = "Artifact transport is not configured"


class _UnavailableTransport:
    async def request(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError(_UNCONFIGURED_MESSAGE)
