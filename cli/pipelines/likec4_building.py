from functools import partial
from typing import Any, Optional

from google.adk.models import LiteLlm

from contractor.agents.likec4_builder_agent.agent import build_likec4_builder_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler

from cli.pipelines import Pipeline


class LikeC4BuildingPipeline(Pipeline):
    """Builds a LikeC4 architecture model for the project, focused on
    security-relevant facts (external interactions, trust boundaries,
    data classes).

    Stages:
      1. dependency_information   — SWE worker
      2. project_information      — SWE worker (consumes deps)
      3. likec4_build             — LikeC4 builder worker (consumes both)
      4. likec4_validate          — LikeC4 builder worker, repair-only pass
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
        fs = ctx.fs

        swe_builder = partial(
            build_swe_agent, name="swe_agent", fs=fs, model=llm, max_tokens=100_000
        )
        likec4_builder = partial(
            build_likec4_builder_agent,
            name="likec4_builder",
            fs=fs,
            model=llm,
            max_tokens=120_000,
        )

        runner.add_variable(name="project_path", value=ctx.folder_name)

        runner.add_task(
            name="dependency_information",
            worker_builder=swe_builder,
            iterations=1,
            max_attempts=3,
            max_steps=20,
            namespace="dependency_information",
            model=llm,
        )

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

        runner.add_task(
            name="likec4_build",
            worker_builder=likec4_builder,
            iterations=3,
            max_attempts=9,
            max_steps=30,
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

        return await runner.run(user_id=user_id, on_event=on_event)
