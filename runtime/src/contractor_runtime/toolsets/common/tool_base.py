"""Base classes for model tools backed by one allocation-bound ArtifactClient."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.toolsets.common.artifact_visibility import ArtifactObservingTool
from contractor_runtime.toolsets.common.metrics import (
    RecordedToolCall,
    ToolMetrics,
    instrumented_call,
)


class _ClosableSession(Protocol):
    async def close(self) -> None: ...


class ArtifactTool(ArtifactObservingTool):
    """Record calls with the runtime secrets redacted until the tool closes."""

    name: str
    description: str

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        self._secrets = ()

    def _recorded(self, arguments: Mapping[str, Any]) -> RecordedToolCall:
        return RecordedToolCall(self._metrics, self.name, arguments, secrets=self._secrets)

    async def _call[T](
        self,
        arguments: Mapping[str, Any],
        operation: Awaitable[T],
        metric_result: Callable[[T], Mapping[str, Any]],
    ) -> T:
        return await instrumented_call(self._recorded(arguments), operation, metric_result)


class SessionArtifactTool[S: _ClosableSession](ArtifactTool):
    """ArtifactTool that also closes the session shared by its toolset."""

    def __init__(
        self,
        session: S,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._session = session
        super().__init__(client, metrics, secrets)

    async def close(self) -> None:
        await self._session.close()
        await super().close()
