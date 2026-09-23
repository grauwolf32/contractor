"""Artifact client construction and credential redaction helpers for toolsets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import RuntimeSettings

ArtifactClientFactory = Callable[[str, RuntimeSettings], ArtifactClient]


def gateway_secrets(settings: RuntimeSettings) -> tuple[str, ...]:
    token = settings.llm_gateway_token
    return () if token is None else (token.get_secret_value(),)


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
