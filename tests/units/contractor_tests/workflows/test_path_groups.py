"""Unit tests for router-prefix path grouping (coverage budgeting)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from google.adk.artifacts import BaseArtifactService, FileArtifactService
from google.genai import types

from cli.fs import RootedLocalFileSystem
from contractor.workflows import WorkflowContext
from contractor.workflows.path_groups import (
    PathGroup,
    group_key_for_path,
    group_paths_by_prefix,
)
from contractor.workflows.path_keys import (
    MAX_OPENAPI_PATH_KEY_LENGTH,
    openapi_path_key,
)
from contractor.workflows.trace_annotation import OpenApiPath, extract_openapi_paths


def _paths(*raw: str) -> list[OpenApiPath]:
    return [OpenApiPath(path=p, operations=[]) for p in raw]


def test_extract_openapi_paths_includes_every_standard_operation_method():
    methods = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
    document = {
        "openapi": "3.1.0",
        "paths": {
            "/operations": {
                method: {"operationId": method, "responses": {"200": {}}}
                for method in methods
            }
        },
    }

    paths = extract_openapi_paths(document)

    assert len(paths) == 1
    assert {operation.method for operation in paths[0].operations} == methods


def test_extract_openapi_paths_derives_id_when_operation_id_is_missing_or_null():
    document = {
        "openapi": "3.1.0",
        "paths": {
            "/items": {
                "get": {"responses": {"200": {}}},
                "post": {"operationId": None, "responses": {"200": {}}},
            }
        },
    }

    paths = extract_openapi_paths(document)

    assert [operation.operation_id for operation in paths[0].operations] == [
        "GET /items",
        "POST /items",
    ]


@pytest.mark.asyncio
async def test_long_unicode_path_key_fits_real_artifact_storage(tmp_path):
    path = "/" + "я" * 80
    key = openapi_path_key(path)
    assert len(key.encode("ascii")) <= MAX_OPENAPI_PATH_KEY_LENGTH
    assert key != openapi_path_key(path + "x")

    service = FileArtifactService(root_dir=str(tmp_path))
    filename = f"user:vulnerability-reports/trace-graph-pathpar:openapi:{key}"
    await service.save_artifact(
        app_name="contractor",
        user_id="u",
        session_id=None,
        filename=filename,
        artifact=types.Part.from_text(text="finding: {}"),
    )
    assert await service.load_artifact(
        app_name="contractor",
        user_id="u",
        session_id=None,
        filename=filename,
    ) is not None


def test_path_keys_use_version_directory_not_ambiguous_legacy_namespace():
    key = openapi_path_key("/users/{id}")

    assert key.startswith("v2/d0/")
    assert key != "users_id"


def test_path_keys_are_portable_across_case_insensitive_and_windows_filesystems():
    assert openapi_path_key("/Users") != openapi_path_key("/users")
    for device_name in ("CON", "nul", "AUX", "COM1", "lpt9"):
        assert openapi_path_key(f"/{device_name}").split("/")[-1].startswith("p-")


class TestGroupKey:
    def test_depth_one_uses_first_segment(self):
        users = group_key_for_path("/users", 1)
        assert group_key_for_path("/users/{user-id}", 1) == users
        assert group_key_for_path("/users/export", 1) == users
        assert group_key_for_path("/admin/stats", 1) == group_key_for_path(
            "/admin", 1
        )
        assert users != openapi_path_key("/users")

    def test_depth_two(self):
        assert group_key_for_path("/api/v1/users", 2) == group_key_for_path(
            "/api/v1", 2
        )

    def test_param_braces_are_collision_safe(self):
        assert group_key_for_path("/{tenant}/users", 1) == group_key_for_path(
            "/{tenant}", 1
        )
        assert group_key_for_path("/{tenant}/users", 1) != group_key_for_path(
            "/tenant/users", 1
        )

    def test_configured_depth_is_preserved_beyond_segments(self):
        assert group_key_for_path("/users", 3).startswith("v2/d3/")
        assert group_key_for_path("/users", 3) != group_key_for_path("/users", 1)

    def test_root_path(self):
        assert group_key_for_path("/", 1).startswith("v2/d1/")
        assert group_key_for_path("/", 1) != openapi_path_key("/")

    def test_distinct_paths_have_distinct_keys(self):
        colliding_under_the_legacy_slug = (
            ("/", "/root"),
            ("/users/{id}", "/users/id"),
            ("/a_b", "/a/b"),
        )
        for left, right in colliding_under_the_legacy_slug:
            assert OpenApiPath(left).path_key != OpenApiPath(right).path_key

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
        assert [g.key for g in groups] == [p.path_key for p in paths]
        assert all(len(g.paths) == 1 for g in groups)

    def test_depth_one_groups_siblings(self):
        paths = _paths("/users/{user-id}", "/users/export", "/admin/stats")
        groups = group_paths_by_prefix(paths, depth=1)
        assert [g.key for g in groups] == [
            group_key_for_path("/users", 1),
            group_key_for_path("/admin", 1),
        ]
        assert [p.path for p in groups[0].paths] == [
            "/users/{user-id}",
            "/users/export",
        ]

    def test_first_seen_order_preserved(self):
        paths = _paths("/b/x", "/a/y", "/b/z")
        groups = group_paths_by_prefix(paths, depth=1)
        assert [g.key for g in groups] == [
            group_key_for_path("/b", 1),
            group_key_for_path("/a", 1),
        ]
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
        assert sorted(groups_seen) == sorted(
            openapi_path_key(path) for path in OPENAPI_DOC["paths"]
        )

    async def test_depth_one_forks_per_route_group(self, tmp_path, monkeypatch):
        forks, groups_seen = await self._run(tmp_path, monkeypatch, depth=1)
        assert len(forks) == 2
        assert sorted(groups_seen) == sorted(
            [
                group_key_for_path("/admin", 1),
                group_key_for_path("/users", 1),
            ]
        )
