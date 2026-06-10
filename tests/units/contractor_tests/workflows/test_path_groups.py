"""Unit tests for router-prefix path grouping (coverage budgeting)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from google.adk.artifacts import BaseArtifactService
from google.genai import types

from cli.fs import RootedLocalFileSystem
from contractor.workflows import WorkflowContext
from contractor.workflows.path_groups import (
    PathGroup,
    group_key_for_path,
    group_paths_by_prefix,
)
from contractor.workflows.trace_annotation import OpenApiPath


def _paths(*raw: str) -> list[OpenApiPath]:
    return [OpenApiPath(path=p, operations=[]) for p in raw]


class TestGroupKey:
    def test_depth_one_uses_first_segment(self):
        assert group_key_for_path("/users/{user-id}", 1) == "users"
        assert group_key_for_path("/users/export", 1) == "users"
        assert group_key_for_path("/admin/stats", 1) == "admin"

    def test_depth_two(self):
        assert group_key_for_path("/api/v1/users", 2) == "api_v1"

    def test_param_braces_stripped(self):
        assert group_key_for_path("/{tenant}/users", 1) == "tenant"

    def test_depth_beyond_segments_uses_all(self):
        assert group_key_for_path("/users", 3) == "users"

    def test_root_path(self):
        assert group_key_for_path("/", 1) == "root"

    def test_full_depth_matches_path_key(self):
        # depth <= 0 must reproduce OpenApiPath.path_key so per-path
        # grouping keeps historical namespaces.
        for raw in ("/users/{user-id}", "/admin/stats", "/", "/items"):
            api_path = OpenApiPath(path=raw, operations=[])
            assert group_key_for_path(raw, 0) == api_path.path_key


class TestGrouping:
    def test_depth_zero_one_group_per_path(self):
        paths = _paths("/users/{user-id}", "/users/export")
        groups = group_paths_by_prefix(paths, depth=0)
        assert [g.key for g in groups] == ["users_user-id", "users_export"]
        assert all(len(g.paths) == 1 for g in groups)

    def test_depth_one_groups_siblings(self):
        paths = _paths("/users/{user-id}", "/users/export", "/admin/stats")
        groups = group_paths_by_prefix(paths, depth=1)
        assert [g.key for g in groups] == ["users", "admin"]
        assert [p.path for p in groups[0].paths] == [
            "/users/{user-id}",
            "/users/export",
        ]

    def test_first_seen_order_preserved(self):
        paths = _paths("/b/x", "/a/y", "/b/z")
        groups = group_paths_by_prefix(paths, depth=1)
        assert [g.key for g in groups] == ["b", "a"]
        assert [p.path for p in groups[0].paths] == ["/b/x", "/b/z"]

    def test_group_operations_flatten_member_paths(self):
        p1 = OpenApiPath(path="/u/a", operations=[])
        p2 = OpenApiPath(path="/u/b", operations=[])
        group = PathGroup(key="u", paths=(p1, p2))
        assert group.operations == []


OPENAPI_DOC = {
    "openapi": "3.0.0",
    "info": {"title": "t", "version": "1"},
    "paths": {
        "/users/{user-id}": {
            "get": {"operationId": "getUser", "responses": {"200": {}}},
        },
        "/users/export": {
            "get": {"operationId": "exportUsers", "responses": {"200": {}}},
        },
        "/admin/stats": {
            "get": {"operationId": "adminStats", "responses": {"200": {}}},
        },
    },
}


def _make_context(tmp_path: Path) -> WorkflowContext:
    (tmp_path / "app.py").write_text("def handler():\n    pass\n")

    artifact_service = MagicMock(spec=BaseArtifactService)

    async def load_artifact(*, app_name, user_id, filename):
        if filename == "oas-openapi-building":
            return types.Part.from_text(text=yaml.safe_dump(OPENAPI_DOC))
        return None

    artifact_service.load_artifact = AsyncMock(side_effect=load_artifact)
    artifact_service.save_artifact = AsyncMock()

    return WorkflowContext(
        project_path=tmp_path,
        folder_name="/",
        model="lm-studio-test",
        app_name="contractor-test",
        user_id="u",
        artifact_service=artifact_service,
        fs=RootedLocalFileSystem(str(tmp_path)),
    )


@pytest.mark.asyncio
class TestPathparGroupForks:
    """The fork/concurrency unit of trace-graph-pathpar follows group_depth."""

    async def _run(self, tmp_path, monkeypatch, depth: int):
        import contractor.workflows.trace_graph_pathpar.workflow as wf_mod
        from contractor.workflows.trace_graph_pathpar import (
            TraceGraphPathParWorkflow,
        )

        monkeypatch.setattr(wf_mod.CFG.budgets, "group_depth", depth)
        monkeypatch.setattr(wf_mod, "attach_graph_tools_if_local", lambda fs: [])
        monkeypatch.setattr(wf_mod, "merge_overlay_forks", lambda *a, **k: [])

        forks: list = []

        def fake_fork(fs, patch):
            fork = MagicMock()
            forks.append(fork)
            return fork

        monkeypatch.setattr(wf_mod, "fork_overlay", fake_fork)

        groups_seen: list[str] = []

        async def fake_group_analysis(
            self, *, group, overlay, runner, user_id, on_event
        ):
            groups_seen.append(group.key)

        monkeypatch.setattr(
            TraceGraphPathParWorkflow, "_run_group_analysis", fake_group_analysis
        )

        workflow = TraceGraphPathParWorkflow(_make_context(tmp_path))
        await workflow._run_impl(user_id="u", on_event=None)
        return forks, groups_seen

    async def test_depth_zero_forks_per_path(self, tmp_path, monkeypatch):
        forks, groups_seen = await self._run(tmp_path, monkeypatch, depth=0)
        assert len(forks) == 3
        assert sorted(groups_seen) == [
            "admin_stats",
            "users_export",
            "users_user-id",
        ]

    async def test_depth_one_forks_per_route_group(self, tmp_path, monkeypatch):
        forks, groups_seen = await self._run(tmp_path, monkeypatch, depth=1)
        assert len(forks) == 2
        assert sorted(groups_seen) == ["admin", "users"]
