from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
import trailmark

from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.factories import built_in_factories
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.toolsets import code_analysis
from contractor_runtime.toolsets.code_analysis import (
    CODE_ANALYSIS_REF,
    SHALLOW_TOOLS,
    CodeAnalysisToolsetFactory,
)
from contractor_runtime.toolsets.code_analysis_languages import GRAPH_EXTENSION_LANGUAGES
from contractor_runtime.toolsets.trailmark_host import probe_trailmark_child


def test_offline_tiny_fixture_probe_succeeds_and_leaves_no_residue(tmp_path: Path) -> None:
    async def scenario() -> None:
        assert await probe_trailmark_child(tmp_path)
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_reviewed_graph_language_table_matches_pinned_public_api() -> None:
    assert set(GRAPH_EXTENSION_LANGUAGES.values()) == set(trailmark.supported_languages())
    assert ".func" in GRAPH_EXTENSION_LANGUAGES
    assert ".sw" in GRAPH_EXTENSION_LANGUAGES
    assert ".sway" not in GRAPH_EXTENSION_LANGUAGES
    assert ".hrl" not in GRAPH_EXTENSION_LANGUAGES


def test_local_probe_is_positive_but_graph_tools_are_not_advertised_yet(tmp_path: Path) -> None:
    async def scenario() -> None:
        factories = built_in_factories(
            tmp_path / "scratch",
            workspace_settings=_workspace_settings("local", tmp_path / "provider"),
        )
        snapshot = await discover_capabilities(factories)
        factory = factories.toolsets[CODE_ANALYSIS_REF]
        assert isinstance(factory, CodeAnalysisToolsetFactory)
        assert factory.graph_probe_succeeded is True
        assert _tools(snapshot) == SHALLOW_TOOLS
        assert not tuple((tmp_path / "scratch").glob("code-analysis-mirror-*"))

    asyncio.run(scenario())


def test_memory_runtime_skips_graph_probe_and_remains_shallow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def unexpected(_root: Path, *, timeout_seconds: float = 5.0) -> bool:
        del timeout_seconds
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(code_analysis, "probe_trailmark_child", unexpected)

    async def scenario() -> None:
        factories = built_in_factories(
            tmp_path / "scratch",
            workspace_settings=_workspace_settings("memory"),
        )
        snapshot = await discover_capabilities(factories)
        factory = factories.toolsets[CODE_ANALYSIS_REF]
        assert isinstance(factory, CodeAnalysisToolsetFactory)
        assert factory.graph_probe_succeeded is False
        assert _tools(snapshot) == SHALLOW_TOOLS
        assert calls == 0

    asyncio.run(scenario())


def test_missing_or_broken_trailmark_affects_only_future_graph_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = code_analysis.dependency_versions_match

    def versions(
        required: Mapping[str, str] = code_analysis.PINNED_DEPENDENCIES,
    ) -> bool:
        if "trailmark" in required:
            return False
        return original(required)

    monkeypatch.setattr(code_analysis, "dependency_versions_match", versions)

    async def scenario() -> None:
        factories = built_in_factories(
            tmp_path / "scratch",
            workspace_settings=_workspace_settings("local", tmp_path / "provider"),
        )
        snapshot = await discover_capabilities(factories)
        factory = factories.toolsets[CODE_ANALYSIS_REF]
        assert isinstance(factory, CodeAnalysisToolsetFactory)
        assert factory.graph_probe_succeeded is False
        assert _tools(snapshot) == SHALLOW_TOOLS

    asyncio.run(scenario())


def test_failed_graph_probe_does_not_remove_shallow_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unavailable(_root: Path, *, timeout_seconds: float = 5.0) -> bool:
        del timeout_seconds
        return False

    monkeypatch.setattr(code_analysis, "probe_trailmark_child", unavailable)

    async def scenario() -> None:
        factories = built_in_factories(
            tmp_path / "scratch",
            workspace_settings=_workspace_settings("local", tmp_path / "provider"),
        )
        snapshot = await discover_capabilities(factories)
        assert _tools(snapshot) == SHALLOW_TOOLS

    asyncio.run(scenario())


def _workspace_settings(storage: str, root: Path | None = None) -> WorkspaceSettings:
    return WorkspaceSettings(
        storage=storage,  # type: ignore[arg-type]
        work_root=root,
        limits=WorkspaceLimits(
            max_files=100,
            max_expanded_bytes=8 * 1024 * 1024,
            max_managed_text_bytes=4 * 1024 * 1024,
            max_file_bytes=4 * 1024 * 1024,
        ),
    )


def _tools(snapshot: object) -> frozenset[str]:
    toolsets = {item.ref: frozenset(item.tools) for item in snapshot.toolsets}  # type: ignore[attr-defined]
    return toolsets[CODE_ANALYSIS_REF]
