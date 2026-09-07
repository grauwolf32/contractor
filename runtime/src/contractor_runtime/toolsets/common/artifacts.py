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
    return ArtifactClient(allocation_id, _UnavailableTransport())


class _UnavailableTransport:
    async def request(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("Artifact transport is not configured")
