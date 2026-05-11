from functools import partial
from typing import Any, Optional

from google.adk.models import LiteLlm

from cli.pipelines import Pipeline
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler


class OasBuildingPipeline(Pipeline):
    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> Any:
        ctx = self.ctx
        runner = TaskRunner(
            name="oas_builder",
            artifact_service=ctx.artifact_service,
        )

        llm = LiteLlm(model=ctx.model)
        fs = ctx.fs
        swe_builder = partial(
            build_swe_agent, name="swe_agent", fs=fs, model=llm, max_tokens=100_000
        )
        oas_builder = partial(
            build_oas_builder_agent,
            name="oas_builder",
            fs=fs,
            model=llm,
            max_tokens=100_000,
        )
        oas_linter = partial(
            build_oas_linter_agent,
            name="oas_validator",
            fs=fs,
            model=llm,
            max_tokens=100_000,
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
                max_attempts=2,
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
                name="project_information",
                worker_builder=swe_builder,
                iterations=1,
                max_attempts=2,
                max_steps=20,
                artifacts=["dependency_information/result"],
                namespace="project_information",
                model=llm,
            )
        else:
            await self.emit_task_skipped(on_event, "project_information")

        runner.add_task(
            name="oas_update",
            worker_builder=oas_builder,
            iterations=2,
            max_attempts=4,
            max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
            ],
            namespace="openapi-building",
            model=llm,
        )

        runner.add_task(
            name="oas_validate",
            worker_builder=oas_linter,
            iterations=1,
            max_attempts=1,
            max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
                "oas_update/result",
            ],
            namespace="openapi-building",
            model=llm,
        )

        return await runner.run(user_id=user_id, on_event=on_event)
