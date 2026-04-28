from functools import partial
from typing import Any, Optional

from google.adk.models import LiteLlm

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.tools.fs import RootedLocalFileSystem

from cli.pipelines import Pipeline, persist_seed_artifact


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
        )

        llm = LiteLlm(model=ctx.model)
        fs = RootedLocalFileSystem(root_path=ctx.project_path)
        oas_builder = partial(
            build_oas_builder_agent,
            name="oas_builder",
            fs=fs,
            model=llm,
            max_tokens=120_000,
        )
        oas_linter = partial(
            build_oas_linter_agent,
            name="oas_validator",
            fs=fs,
            model=llm,
            max_tokens=120_000,
        )

        await persist_seed_artifact(ctx, filename="oas-openapi-building")

        runner.add_variable(name="project_path", value=ctx.folder_name)

        runner.add_task(
            name="oas_enrich_experimental",
            worker_builder=oas_builder,
            iterations=3,
            max_attempts=9,
            max_steps=30,
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
            iterations=2,
            max_attempts=6,
            max_steps=30,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
                "oas_enrich/result",
            ],
            namespace="openapi-building",
            model=llm,
        )

        return await runner.run(user_id=user_id, on_event=on_event)
