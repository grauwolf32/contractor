import asyncio
import json
import logging
from functools import partial
from pathlib import Path

import click
from dotenv import load_dotenv
from google.adk.artifacts import FileArtifactService

from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.runners.task_runner import TaskRunner
from contractor.tools.fs import RootedLocalFileSystem
from contractor.utils.formatting import handle_event, make_jsonable

load_dotenv()


def turn_off_logger() -> None:
    names = [
        "httpcore",
        "fsspec",
        "google_adk",
        "google_genai",
        "openai",
        "litellm",
        "LiteLLM",
        "asyncio",
    ]
    for name in names:
        logging.getLogger(name).setLevel(logging.CRITICAL)


turn_off_logger()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

ARTIFACTS_DIR: Path = Path(__file__).parent.parent / "artifacts"


def oas_builder(
    *,
    project_path: Path,
    folder_name: str,
) -> TaskRunner:
    artifact_service = FileArtifactService(root_dir=ARTIFACTS_DIR)
    runner = TaskRunner(
        name="oas_builder",
        artifact_service=artifact_service,
    )

    fs = RootedLocalFileSystem(root_path=project_path)
    swe_builder = partial(build_swe_agent, name="swe_agent", fs=fs)
    oas_builder_fn = partial(build_oas_builder_agent, name="oas_builder", fs=fs)

    runner.add_variable(name="project_path", value=folder_name)

    runner.add_task(
        name="dependency_information",
        worker_builder=swe_builder,
        iterations=1,
        max_attempts=3,
        namespace="dependency_information",
    )

    runner.add_task(
        name="project_information",
        worker_builder=swe_builder,
        iterations=1,
        max_attempts=3,
        artifacts=["dependency_information/result"],
        namespace="project_information",
    )

    runner.add_task(
        name="oas_bootstrap",
        worker_builder=oas_builder_fn,
        iterations=1,
        max_attempts=3,
        artifacts=[
            "dependency_information/result",
            "project_information/result",
        ],
        namespace="openapi-building",
    )

    runner.add_task(
        name="oas_update",
        worker_builder=oas_builder_fn,
        iterations=3,
        max_attempts=9,
        artifacts=[
            "dependency_information/result",
            "project_information/result",
        ],
        namespace="openapi-building",
    )

    return runner


async def async_main(project_path: Path, folder_name: str, user_id: str) -> None:
    runner = oas_builder(
        project_path=project_path,
        folder_name=folder_name,
    )

    results = await runner.run(
        user_id=user_id,
        on_event=handle_event,
    )

    results = make_jsonable(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


@click.command(name="contractor")
@click.option(
    "--project-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to the project directory",
)
@click.option(
    "--folder-name",
    type=str,
    default="/",
    show_default=True,
    help="Project-relative folder path used inside task templates",
)
@click.option(
    "--user-id",
    type=str,
    default="cli-user",
    show_default=True,
    help="User id for ADK session runner",
)
@click.option(
    "--model",
    type=str,
    default="gpt-4-1106-preview",
    show_default=True,
    help="Model name to use for the task",
)
def main(project_path: Path, folder_name: str, user_id: str) -> None:
    """Run contractor task pipeline for a project."""
    asyncio.run(async_main(project_path, folder_name, user_id))


if __name__ == "__main__":
    main()
