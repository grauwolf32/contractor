from pathlib import Path
from typing import Optional
from functools import partial
from contractor.tools.fs import RootedLocalFileSystem
from google.adk.artifacts import BaseArtifactService
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.runners.task_runner import TaskRunner
from google.adk.models import LiteLlm
from cli.utils import save_artifact


async def oas_enrichment_pipeline(
    *,
    project_path: Path,
    folder_name: str,
    model: str,
    app_name: str,
    user_id: str,
    artifact_service: BaseArtifactService,
    artifact: Optional[str] = None,
    **kwargs,
) -> TaskRunner:
    runner = TaskRunner(
        name="oas_builder",
        artifact_service=artifact_service,
    )

    llm = LiteLlm(model=model)
    fs = RootedLocalFileSystem(root_path=project_path)
    oas_builder = partial(build_oas_builder_agent, name="oas_builder", fs=fs, model=llm)

    if artifact:
        artifact_text = types.Part.from_text(text=artifact)
        await artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            filename="oas-openapi-building",
            artifact=artifact_text,
        )

    runner.add_variable(name="project_path", value=folder_name)

    runner.add_task(
        name="oas_enrich",
        worker_builder=oas_builder,
        iterations=1,
        max_attempts=3,
        max_steps=20,
        artifacts=[
            "dependency_information/result",
            "project_information/result",
            "oas-openapi-building",
        ],
        namespace="openapi-building",
        model=llm,
    )

    return runner
