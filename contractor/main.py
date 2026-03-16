import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Awaitable, Optional

import click
from dotenv import load_dotenv
from google.adk.artifacts import FileArtifactService
from google.adk.models import LiteLlm

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.swe_agent.agent import build_swe_agent
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


@dataclass(frozen=True)
class PipelineSpec:
    builder: Callable[..., Awaitable[TaskRunner]]
    requires_artifact: bool = False


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_folder_name(folder_name: str) -> str:
    if folder_name in ("", "/"):
        return "/"

    normalized = Path(folder_name.lstrip("/")).as_posix().strip("/")
    return f"/{normalized}" if normalized else "/"


def _validate_project_path(project_path: Path) -> Path:
    project_path = project_path.resolve()

    if not project_path.exists():
        raise click.BadParameter(
            f"Directory does not exist: {project_path}",
            param_hint="--project-path",
        )

    if not project_path.is_dir():
        raise click.BadParameter(
            f"Path is not a directory: {project_path}",
            param_hint="--project-path",
        )

    return project_path


def _validate_folder_name(project_path: Path, folder_name: str) -> str:
    normalized_folder = _normalize_folder_name(folder_name)

    if normalized_folder == "/":
        target_dir = project_path
    else:
        target_dir = (project_path / normalized_folder.lstrip("/")).resolve()

    try:
        target_dir.relative_to(project_path)
    except ValueError as exc:
        raise click.BadParameter(
            "--folder-name must point to a directory inside --project-path",
            param_hint="--folder-name",
        ) from exc

    if not target_dir.exists():
        raise click.BadParameter(
            f"Directory does not exist: {target_dir}",
            param_hint="--folder-name",
        )

    if not target_dir.is_dir():
        raise click.BadParameter(
            f"Path is not a directory: {target_dir}",
            param_hint="--folder-name",
        )

    return normalized_folder


def _read_artifact_file(artifact_path: Optional[Path]) -> Optional[str]:
    if artifact_path is None:
        return None

    artifact_path = artifact_path.resolve()

    if not artifact_path.exists():
        raise click.BadParameter(
            f"File does not exist: {artifact_path}",
            param_hint="--artifact",
        )

    if not artifact_path.is_file():
        raise click.BadParameter(
            f"Path is not a file: {artifact_path}",
            param_hint="--artifact",
        )

    try:
        return artifact_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise click.BadParameter(
            f"Artifact file must be a UTF-8 text file: {artifact_path}",
            param_hint="--artifact",
        ) from exc


async def oas_building_pipeline(
    *,
    project_path: Path,
    folder_name: str,
    model: str,
) -> TaskRunner:
    _ensure_artifacts_dir()

    artifact_service = FileArtifactService(root_dir=ARTIFACTS_DIR)
    runner = TaskRunner(
        name="oas_builder",
        artifact_service=artifact_service,
    )

    llm = LiteLlm(model=model)
    fs = RootedLocalFileSystem(root_path=project_path)
    swe_builder = partial(build_swe_agent, name="swe_agent", fs=fs, model=llm)
    oas_builder = partial(build_oas_builder_agent, name="oas_builder", fs=fs, model=llm)

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

    return runner


async def oas_enrichment_pipeline(
    *,
    project_path: Path,
    folder_name: str,
    model: str,
    artifact: Optional[str]=None,
) -> TaskRunner:
    _ensure_artifacts_dir()

    artifact_service = FileArtifactService(root_dir=ARTIFACTS_DIR)
    runner = TaskRunner(
        name="oas_builder",
        artifact_service=artifact_service,
    )

    llm = LiteLlm(model=model)
    fs = RootedLocalFileSystem(root_path=project_path)
    swe_builder = partial(build_swe_agent, name="swe_agent", fs=fs, model=llm)
    oas_builder = partial(build_oas_builder_agent, name="oas_builder", fs=fs, model=llm)

    if artifact:
        artifact = types.Part.from_text(text=artifact) 
        await artifact_service.save_artifact(app_name=app_name, user_id=user_id, filename="oas-openapi-building")

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


def get_pipelines() -> dict[str, PipelineSpec]:
    return {
        "build": PipelineSpec(
            builder=oas_building_pipeline,
            requires_artifact=False,
        ),
        "enrich": PipelineSpec(
            builder=oas_enrichment_pipeline,
            requires_artifact=True,
        ),
    }


def get_pipeline_names() -> list[str]:
    return sorted(get_pipelines().keys())


async def async_main(
    project_path: Path,
    folder_name: str,
    user_id: str,
    model: str,
    pipeline: str,
    artifact: Optional[str],
) -> None:
    pipeline = pipeline.lower()
    pipelines = get_pipelines()
    spec = pipelines.get(pipeline)

    if spec is None:
        available = ", ".join(sorted(pipelines))
        raise click.UsageError(
            f"Unsupported pipeline: {pipeline}. Available: {available}"
        )

    builder_kwargs = {
        "project_path": project_path,
        "folder_name": folder_name,
        "model": model,
    }

    if spec.requires_artifact and artifact:
        builder_kwargs["artifact"] = artifact

    runner = await spec.builder(**builder_kwargs)

    _ = await runner.run(
        user_id=user_id,
        on_event=handle_event,
    )

@click.command(name="contractor")
@click.option(
    "--pipeline",
    type=click.Choice(get_pipeline_names(), case_sensitive=False),
    default="build",
    show_default=True,
    help="Pipeline to run",
)
@click.option(
    "--project-path",
    required=True,
    type=click.Path(path_type=Path),
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
    "--artifact",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to existing OpenAPI artifact file for pipelines that require it",
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
    default="lm-studio-qwen3.5",
    show_default=True,
    help="Model name to use for the task",
)
def main(
    pipeline: str,
    project_path: Path,
    folder_name: str,
    artifact: Optional[Path],
    user_id: str,
    model: str,
) -> None:
    """Run contractor task pipeline for a project."""
    pipeline = pipeline.lower()
    project_path = _validate_project_path(project_path)
    folder_name = _validate_folder_name(project_path, folder_name)
    artifact_text = _read_artifact_file(artifact)

    pipelines = get_pipelines()
    spec = pipelines[pipeline]

    if artifact is not None and not spec.requires_artifact:
        raise click.UsageError(
            f"--artifact cannot be used with --pipeline {pipeline}"
        )

    asyncio.run(
        async_main(
            project_path=project_path,
            folder_name=folder_name,
            user_id=user_id,
            model=model,
            pipeline=pipeline,
            artifact=artifact_text,
        )
    )


if __name__ == "__main__":
    main()