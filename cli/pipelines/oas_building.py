from functools import partial
from typing import Any, Optional

from cli.pipelines import Pipeline
from cli.pipelines.config import OAS_BUILDING as CFG
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model


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
            checkpoint_path=ctx.checkpoint_path,
        )

        llm = build_model(ctx.model, ctx.timeout)
        fs = ctx.fs
        swe_builder = partial(
            build_swe_agent, name="swe_agent", fs=fs, model=llm,
            max_tokens=CFG.swe_max_tokens,
        )
        oas_builder = partial(
            build_oas_builder_agent,
            name="oas_builder",
            fs=fs,
            model=llm,
            max_tokens=CFG.builder_max_tokens,
        )
        oas_linter = partial(
            build_oas_linter_agent,
            name="oas_validator",
            fs=fs,
            model=llm,
            max_tokens=CFG.validator_max_tokens,
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
                **CFG.dependency_information.as_kwargs(),
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
                **CFG.project_information.as_kwargs(),
                artifacts=["dependency_information/result"],
                namespace="project_information",
                model=llm,
            )
        else:
            await self.emit_task_skipped(on_event, "project_information")

        runner.add_task(
            name="oas_update",
            worker_builder=oas_builder,
            **CFG.oas_update.as_kwargs(),
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
            **CFG.oas_validate.as_kwargs(),
            artifacts=[
                "dependency_information/result",
                "project_information/result",
                "oas_update/result",
            ],
            namespace="openapi-building",
            model=llm,
        )

        return await runner.run(user_id=user_id, on_event=on_event)
