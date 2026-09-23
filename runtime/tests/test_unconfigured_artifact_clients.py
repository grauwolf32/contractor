from __future__ import annotations

import asyncio

import pytest

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.artifacts import (
    _reject_unconfigured_client,
    _unconfigured_client,
)
from contractor_runtime.toolsets.likec4.tools import LikeC4ToolsetFactory
from contractor_runtime.toolsets.memory.tools import MemoryToolsetFactory
from contractor_runtime.toolsets.openapi.tools import OpenAPIToolsetFactory
from contractor_runtime.toolsets.source_analysis.tools import SourceAnalysisToolsetFactory
from contractor_runtime.toolsets.text_artifacts.tools import TextArtifactsToolsetFactory


def test_unconfigured_client_fails_on_first_request() -> None:
    client = _unconfigured_client("alloc-1", None)  # type: ignore[arg-type]
    assert client.allocation_id == "alloc-1"
    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(client.read_artifact(ArtifactRef(namespace="analysis", name="report")))


def test_rejecting_factory_fails_at_construction() -> None:
    with pytest.raises(RuntimeError, match="not configured"):
        _reject_unconfigured_client("alloc-1", None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (LikeC4ToolsetFactory, _unconfigured_client),
        (MemoryToolsetFactory, _unconfigured_client),
        (OpenAPIToolsetFactory, _unconfigured_client),
        (SourceAnalysisToolsetFactory, _reject_unconfigured_client),
        (TextArtifactsToolsetFactory, _reject_unconfigured_client),
    ],
)
def test_toolsets_default_to_shared_unconfigured_factories(factory: type, expected: object) -> None:
    assert factory()._client_factory is expected
