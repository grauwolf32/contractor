import logging
from functools import partial
from typing import Any, Optional

from google.adk.models import LiteLlm
from google.genai import types

from contractor.agents.likec4_builder_agent.agent import build_likec4_builder_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.tools.fs import MemoryOverlayFileSystem
from contractor.tools.likec4 import DEFAULT_LIKEC4_PATH

from cli.pipelines import Pipeline

logger = logging.getLogger(__name__)


# Artifact key for the canonical LikeC4 source. Kept as a single text artifact
# so reruns of the pipeline can pick up where the previous one left off.
LIKEC4_ARTIFACT_FILENAME: str = "likec4-architecture.c4"


class LikeC4BuildingPipeline(Pipeline):
    """Builds a LikeC4 architecture model for the project, focused on
    security-relevant facts (external interactions, trust boundaries,
    data classes).

    Stages:
      1. dependency_information   — SWE worker
      2. project_information      — SWE worker (consumes deps)
      3. likec4_build             — LikeC4 builder worker (consumes both)
      4. likec4_validate          — LikeC4 builder worker, repair-only pass

    Persistence: the LikeC4 source lives in a single overlay-backed file at
    :data:`contractor.tools.likec4.DEFAULT_LIKEC4_PATH`. Before the pipeline
    runs, that file is seeded from the ``likec4-architecture.c4`` artifact if
    it exists; after the pipeline finishes, the file is read back and saved
    as that artifact.
    """

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> Any:
        ctx = self.ctx
        runner = TaskRunner(
            name="likec4_builder",
            artifact_service=ctx.artifact_service,
        )

        llm = LiteLlm(model=ctx.model)
        # Single overlay shared across the build and validate tasks so the
        # validate worker sees what the build worker wrote.
        overlay_fs = MemoryOverlayFileSystem(fs=ctx.fs)

        await self._seed_overlay_from_artifact(overlay_fs, user_id=user_id)

        swe_builder = partial(
            build_swe_agent, name="swe_agent", fs=ctx.fs, model=llm, max_tokens=100_000
        )
        likec4_builder = partial(
            build_likec4_builder_agent,
            name="likec4_builder",
            fs=overlay_fs,
            model=llm,
            max_tokens=120_000,
        )

        runner.add_variable(name="project_path", value=ctx.folder_name)

        # Code-analysis tasks (`dependency_information`, `project_information`)
        # are expensive and idempotent — if a previous run already produced
        # the artifact downstream tasks consume, skip the regeneration.
        if not await self.artifact_exists(
            user_id=user_id, filename="dependency_information/result"
        ):
            runner.add_task(
                name="dependency_information",
                worker_builder=swe_builder,
                iterations=1,
                max_attempts=3,
                max_steps=20,
                namespace="dependency_information",
                model=llm,
            )
        else:
            await self.emit_task_skipped(on_event, "dependency_information")

        if not await self.artifact_exists(
            user_id=user_id, filename="project_information/result"
        ):
            runner.add_task(
                name="project_information_experimental",
                worker_builder=swe_builder,
                iterations=1,
                max_attempts=3,
                max_steps=20,
                artifacts=["dependency_information/result"],
                namespace="project_information",
                model=llm,
            )
        else:
            await self.emit_task_skipped(on_event, "project_information_experimental")

        runner.add_task(
            name="likec4_build",
            worker_builder=likec4_builder,
            iterations=3,
            max_attempts=6,
            max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
            ],
            namespace="likec4-building",
            model=llm,
        )

        runner.add_task(
            name="likec4_validate",
            worker_builder=likec4_builder,
            iterations=1,
            max_attempts=2,
            max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
                "likec4_build/result",
            ],
            namespace="likec4-building",
            model=llm,
        )

        try:
            return await runner.run(user_id=user_id, on_event=on_event)
        finally:
            await self._persist_overlay_to_artifact(overlay_fs, user_id=user_id)

    async def _seed_overlay_from_artifact(
        self,
        overlay_fs: MemoryOverlayFileSystem,
        *,
        user_id: str,
    ) -> None:
        """Write the previously-saved LikeC4 source to ``DEFAULT_LIKEC4_PATH``.

        No-op when no prior artifact exists (first run on this project).
        """
        ctx = self.ctx
        artifact = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=LIKEC4_ARTIFACT_FILENAME,
        )
        if not artifact or not artifact.text:
            return
        overlay_fs.write_text(DEFAULT_LIKEC4_PATH, artifact.text, encoding="utf-8")
        logger.info(
            "seeded %s from artifact %r (%d chars)",
            DEFAULT_LIKEC4_PATH,
            LIKEC4_ARTIFACT_FILENAME,
            len(artifact.text),
        )

    async def _persist_overlay_to_artifact(
        self,
        overlay_fs: MemoryOverlayFileSystem,
        *,
        user_id: str,
    ) -> None:
        """Read ``DEFAULT_LIKEC4_PATH`` and save it as the LikeC4 artifact.

        If the file is missing (e.g. the pipeline failed before any agent
        wrote it) the previous artifact is left untouched.
        """
        if not overlay_fs.exists(DEFAULT_LIKEC4_PATH):
            logger.info(
                "no %s on overlay — skipping artifact save", DEFAULT_LIKEC4_PATH
            )
            return
        content = overlay_fs.read_text(DEFAULT_LIKEC4_PATH, encoding="utf-8")
        ctx = self.ctx
        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=LIKEC4_ARTIFACT_FILENAME,
            artifact=types.Part.from_text(text=content),
        )
        logger.info(
            "saved %s to artifact %r (%d chars)",
            DEFAULT_LIKEC4_PATH,
            LIKEC4_ARTIFACT_FILENAME,
            len(content),
        )
