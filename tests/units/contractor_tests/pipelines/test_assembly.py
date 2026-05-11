"""Pipeline-assembly smoke tests.

These tests exercise each pipeline's ``_run_impl`` with the heavy I/O
side-effects stubbed out, then inspect the resulting ``TaskRunner.queue``.
They are the first line of defense against artifact-wiring regressions:
a typo in an upstream ``add_task(name=...)`` or a downstream
``artifacts=[...]`` reference is silently fatal at runtime, but caught
here by ``test_<pipeline>_artifact_references_resolve``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.artifacts import BaseArtifactService

from cli.pipelines import Pipeline, PipelineContext, get_pipelines
from cli.pipelines.likec4_building import LikeC4BuildingPipeline
from cli.pipelines.oas_building import OasBuildingPipeline
from cli.pipelines.oas_enrichment import OasEnrichmentPipeline
from cli.pipelines.router import RouterPipeline
from cli.pipelines.trace_annotation import TraceAnnotationPipeline
from cli.pipelines.trace_annotation_direct import TraceAnnotationDirectPipeline
from contractor.runners.task_runner import TaskRunner


# ─── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_known_keys_present(self):
        registry = get_pipelines()
        assert set(registry.keys()) == {
            "build",
            "enrich",
            "likec4",
            "trace",
            "trace-direct",
            "router",
        }

    def test_all_classes_extend_pipeline_base(self):
        for cls in get_pipelines().values():
            assert issubclass(cls, Pipeline), f"{cls.__name__} must extend Pipeline"

    def test_registry_matches_explicit_imports(self):
        # Catch the case where get_pipelines() drifts from the modules that
        # actually define the classes — e.g. someone deletes a pipeline file
        # but forgets to remove it from the registry.
        registry = get_pipelines()
        assert registry["build"] is OasBuildingPipeline
        assert registry["enrich"] is OasEnrichmentPipeline
        assert registry["likec4"] is LikeC4BuildingPipeline
        assert registry["trace"] is TraceAnnotationPipeline
        assert registry["trace-direct"] is TraceAnnotationDirectPipeline
        assert registry["router"] is RouterPipeline


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_context(*, prompt: str | None = None) -> PipelineContext:
    """A PipelineContext with all I/O surfaces mocked.

    `load_artifact` returns None by default — pipelines should treat that as
    "no prior run". Tests that need a non-empty artifact patch this further.
    """
    artifact_service = MagicMock(spec=BaseArtifactService)
    artifact_service.load_artifact = AsyncMock(return_value=None)
    artifact_service.save_artifact = AsyncMock()

    return PipelineContext(
        project_path=Path("/tmp/proj"),
        folder_name="src",
        model="lm-studio-test",
        app_name="contractor-test",
        user_id="u",
        artifact_service=artifact_service,
        fs=MagicMock(),
        prompt=prompt,
    )


async def _capture_queue(pipeline: Pipeline, *, monkeypatch) -> list:
    """Invoke a pipeline's ``_run_impl`` while suppressing the actual run.

    Returns the list of ``TaskInvocation`` objects added to the queue.
    """
    captured: dict = {}

    original_init = TaskRunner.__init__

    def capture_init(self, **kwargs):
        original_init(self, **kwargs)
        captured.setdefault("runners", []).append(self)

    async def fake_run(self, **kwargs):
        return []

    monkeypatch.setattr(TaskRunner, "__init__", capture_init)
    monkeypatch.setattr(TaskRunner, "run", fake_run)

    await pipeline._run_impl(user_id="u", on_event=None)

    queues: list = []
    for r in captured.get("runners", []):
        queues.extend(r.queue)
    return queues


def _refs(queue) -> set[str]:
    """Set of ``ref``s actually queued."""
    return {item.ref for item in queue}


def _template_keys(queue) -> set[str]:
    return {item.template_key for item in queue}


def _all_artifact_refs(queue) -> set[str]:
    refs: set[str] = set()
    for item in queue:
        refs.update(item.artifacts)
    return refs


# Artifacts that pipelines persist outside the runner before any task runs.
# Trace pipelines seed ``oas-openapi-building`` via ``persist_seed_artifact``
# and depend on it being present — they don't go through the runner queue.
_KNOWN_SEED_ARTIFACTS: set[str] = set()


def _producing_task_key(artifact_ref: str) -> str:
    """For ``<template_key>/result``, the upstream task is ``<template_key>``."""
    return artifact_ref.rsplit("/", 1)[0]


# ─── OasBuildingPipeline ──────────────────────────────────────────────────────


class TestOasBuildingPipeline:
    @pytest.mark.asyncio
    async def test_assembles_four_tasks(self, monkeypatch):
        pipeline = OasBuildingPipeline(_make_context())
        queue = await _capture_queue(pipeline, monkeypatch=monkeypatch)

        # Two code-analysis tasks + oas_update + oas_validate, all queued
        # because load_artifact returns None (no prior run).
        assert _template_keys(queue) == {
            "dependency_information",
            "project_information",
            "oas_update",
            "oas_validate",
        }

    @pytest.mark.asyncio
    async def test_artifact_references_resolve(self, monkeypatch):
        pipeline = OasBuildingPipeline(_make_context())
        queue = await _capture_queue(pipeline, monkeypatch=monkeypatch)

        produced = _template_keys(queue) | _KNOWN_SEED_ARTIFACTS
        for artifact_ref in _all_artifact_refs(queue):
            assert _producing_task_key(artifact_ref) in produced, (
                f"artifact {artifact_ref!r} is referenced but no upstream "
                f"task or known seed produces it"
            )

    @pytest.mark.asyncio
    async def test_skip_when_dependency_artifact_exists(self, monkeypatch):
        ctx = _make_context()

        # Pretend dependency_information already has a result on disk.
        async def fake_load(*, app_name, user_id, filename, **_):
            if filename == "dependency_information/result":
                return MagicMock(text="ok", inline_data=None)
            return None

        ctx.artifact_service.load_artifact = AsyncMock(side_effect=fake_load)

        pipeline = OasBuildingPipeline(ctx)
        queue = await _capture_queue(pipeline, monkeypatch=monkeypatch)
        assert "dependency_information" not in _template_keys(queue)
        # Downstream tasks still queued — they will read the existing artifact.
        assert "oas_update" in _template_keys(queue)


# ─── OasEnrichmentPipeline ────────────────────────────────────────────────────


class TestOasEnrichmentPipeline:
    @pytest.mark.asyncio
    async def test_assembles_enrich_and_validate(self, monkeypatch):
        pipeline = OasEnrichmentPipeline(_make_context())
        queue = await _capture_queue(pipeline, monkeypatch=monkeypatch)
        assert _template_keys(queue) == {"oas_enrich", "oas_validate"}

    @pytest.mark.asyncio
    async def test_artifact_references_resolve(self, monkeypatch):
        # Enrich pulls upstream `dependency_information/result` and
        # `project_information/result`, which this pipeline does NOT queue —
        # they must come from a prior `build` run. Treat them as accepted
        # external inputs.
        external_inputs = {"dependency_information", "project_information"}
        pipeline = OasEnrichmentPipeline(_make_context())
        queue = await _capture_queue(pipeline, monkeypatch=monkeypatch)

        produced = _template_keys(queue) | external_inputs
        for artifact_ref in _all_artifact_refs(queue):
            assert _producing_task_key(artifact_ref) in produced


# ─── LikeC4BuildingPipeline ───────────────────────────────────────────────────


class TestLikeC4BuildingPipeline:
    @pytest.mark.asyncio
    async def test_assembles_four_tasks(self, monkeypatch):
        # Skip the overlay seed/persist (they hit ctx.fs which is a MagicMock).
        monkeypatch.setattr(
            LikeC4BuildingPipeline,
            "_seed_overlay_from_artifact",
            AsyncMock(),
        )
        monkeypatch.setattr(
            LikeC4BuildingPipeline,
            "_persist_overlay_to_artifact",
            AsyncMock(),
        )

        pipeline = LikeC4BuildingPipeline(_make_context())
        queue = await _capture_queue(pipeline, monkeypatch=monkeypatch)
        assert _template_keys(queue) == {
            "dependency_information",
            "project_information",
            "likec4_build",
            "likec4_validate",
        }

    @pytest.mark.asyncio
    async def test_artifact_references_resolve(self, monkeypatch):
        monkeypatch.setattr(
            LikeC4BuildingPipeline,
            "_seed_overlay_from_artifact",
            AsyncMock(),
        )
        monkeypatch.setattr(
            LikeC4BuildingPipeline,
            "_persist_overlay_to_artifact",
            AsyncMock(),
        )

        pipeline = LikeC4BuildingPipeline(_make_context())
        queue = await _capture_queue(pipeline, monkeypatch=monkeypatch)

        produced = _template_keys(queue) | _KNOWN_SEED_ARTIFACTS
        for artifact_ref in _all_artifact_refs(queue):
            assert _producing_task_key(artifact_ref) in produced


# ─── TraceAnnotationPipeline ──────────────────────────────────────────────────


class TestTraceAnnotationPipeline:
    """Per-path trace pipeline. Test the per-path task assembly directly
    using ``_run_path_analysis`` since the outer loop iterates over an
    OpenAPI artifact we'd otherwise need to fabricate."""

    @pytest.mark.asyncio
    async def test_per_path_task_assembles(self, monkeypatch):
        from cli.pipelines.trace_annotation import OpenApiOperation, OpenApiPath

        api_path = OpenApiPath(
            path="/items/{id}",
            operations=[
                OpenApiOperation(
                    operation_id="getItem",
                    method="get",
                    path="/items/{id}",
                    schema={},
                )
            ],
        )

        pipeline = TraceAnnotationPipeline(_make_context())
        queue: list = []
        original_init = TaskRunner.__init__

        def capture_init(self, **kwargs):
            original_init(self, **kwargs)
            queue.append(self)

        monkeypatch.setattr(TaskRunner, "__init__", capture_init)
        monkeypatch.setattr(TaskRunner, "run", AsyncMock())

        await pipeline._run_path_analysis(api_path, user_id="u")

        assert len(queue) == 1
        runner = queue[0]
        assert len(runner.queue) == 1
        item = runner.queue[0]
        assert item.template_key == "trace_annotation"
        assert "items_id" in item.ref
        assert item.skills == ["trace"]
        # No artifacts — the trace template reads from the overlay-FS instead.
        assert item.artifacts == []


# ─── TraceAnnotationDirectPipeline ────────────────────────────────────────────


class TestTraceAnnotationDirectPipeline:
    def test_template_loads_at_class_init(self):
        # The "trace_annotation" template MUST be loadable for this pipeline
        # to import successfully. Construction triggers TaskTemplate.load —
        # a missing manifest or body file fails the test.
        pipeline = TraceAnnotationDirectPipeline(_make_context())
        assert pipeline._template.key == "trace_annotation"
        assert pipeline._template.version  # any non-empty version is fine
