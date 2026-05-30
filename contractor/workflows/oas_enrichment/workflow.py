from functools import partial
from typing import Any, Optional

from cli.pipelines import Pipeline, persist_seed_artifact
from cli.pipelines.config import OAS_ENRICHMENT as CFG
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model


class OasEnrichmentPipeline(Pipeline):
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

        await persist_seed_artifact(ctx, filename="oas-openapi-building")

        runner.add_variable(name="project_path", value=ctx.folder_name)

        runner.add_task(
            name="oas_enrich",
            worker_builder=oas_builder,
            **CFG.oas_enrich.as_kwargs(),
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
                "oas_enrich/result",
            ],
            namespace="openapi-building",
            model=llm,
        )

        return await runner.run(user_id=user_id, on_event=on_event)
