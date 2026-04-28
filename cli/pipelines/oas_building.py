from functools import partial
from typing import Any, Optional

from google.adk.models import LiteLlm

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.tools.fs import RootedLocalFileSystem

from cli.pipelines import Pipeline


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
        fs = RootedLocalFileSystem(root_path=ctx.project_path)
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
            name="oas_update_experimental",
            worker_builder=oas_builder,
            iterations=3,
            max_attempts=9,
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
            max_attempts=3,
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
