from pathlib import Path
from functools import partial
from contractor.tools.fs import RootedLocalFileSystem
from google.adk.artifacts import BaseArtifactService
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner
from google.adk.models import LiteLlm


async def oas_building_pipeline(
    *,
    project_path: Path,
    folder_name: str,
    model: str,
    artifact_service: BaseArtifactService,
    **kwargs,
) -> TaskRunner:
    runner = TaskRunner(
        name="oas_builder",
        artifact_service=artifact_service,
    )

    llm = LiteLlm(model=model)
    fs = RootedLocalFileSystem(root_path=project_path)
    swe_builder = partial(build_swe_agent, name="swe_agent", fs=fs, model=llm)
    oas_builder = partial(build_oas_builder_agent, name="oas_builder", fs=fs, model=llm)
    oas_linter = partial(build_oas_linter_agent, name="oas_validator", fs=fs, model=llm)

    runner.add_variable(name="project_path", value=folder_name)

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
        name="project_information",
        worker_builder=swe_builder,
        iterations=1,
        max_attempts=3,
        max_steps=20,
        artifacts=["dependency_information/result"],
        namespace="project_information",
        model=llm,
    )

    runner.add_task(
        name="oas_update",
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

    # runner.transfer... (transfer memories) ?

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
            "oas_update/summary"
        ],
        namespace="openapi-building",
        model=llm
    )

    return runner
