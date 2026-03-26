from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import click
from dotenv import load_dotenv
from google.adk.artifacts import FileArtifactService
from google.adk.models import LiteLlm
from google.genai import types

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEvent
from contractor.tools.fs import RootedLocalFileSystem
from contractor.utils.formatting import render_event

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
_METRICS_LOCK = asyncio.Lock()


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:
                pass

    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass

    return repr(value)


def _metrics_file(output_dir: Path) -> Path:
    return output_dir / "metrics.jsonl"


def _event_to_metrics_record(event: TaskRunnerEvent) -> dict[str, Any]:
    payload = _jsonable(getattr(event, "payload", {}) or {})

    return {
        "ts": _utc_now_iso(),
        "type": getattr(event, "type", None),
        "task_name": getattr(event, "task_name", None),
        "task_id": getattr(event, "task_id", None),
        "payload": payload,
        "iteration": payload.get("iteration") if isinstance(payload, dict) else None,
        "session_id": payload.get("session_id") if isinstance(payload, dict) else None,
        "invocation_id": payload.get("invocation_id") if isinstance(payload, dict) else None,
        "agent_name": payload.get("agent_name") if isinstance(payload, dict) else None,
        "tool_name": payload.get("tool_name") if isinstance(payload, dict) else None,
    }


def _append_jsonl_sync(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


async def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    await asyncio.to_thread(_append_jsonl_sync, path, record)


def _is_metrics_event(event: TaskRunnerEvent) -> bool:
    event_type = getattr(event, "type", "") or ""
    return event_type.startswith("metrics_")


def build_handle_event(output_dir: Path) -> Callable[[TaskRunnerEvent], Awaitable[None]]:
    metrics_path = _metrics_file(output_dir)

    async def handle_event(event: TaskRunnerEvent) -> None:
        if _is_metrics_event(event):
            record = _event_to_metrics_record(event)

            async with _METRICS_LOCK:
                await _append_jsonl(metrics_path, record)

            return

        await render_event(event=event)

    return handle_event


async def oas_building_pipeline(
    *,
    project_path: Path,
    folder_name: str,
    model: str,
    **kwargs,
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
    artifact: Optional[str] = None,
    app_name: str,
    user_id: str,
    **kwargs,
) -> TaskRunner:
    _ensure_artifacts_dir()

    artifact_service = FileArtifactService(root_dir=ARTIFACTS_DIR)
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


async def save_artifact(
    app_name: str,
    user_id: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_service = FileArtifactService(root_dir=ARTIFACTS_DIR)
    artifact_keys = await artifact_service.list_artifact_keys(
        app_name=app_name,
        user_id=user_id,
    )

    for filename in artifact_keys:
        upload_path = output_dir / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = await artifact_service.load_artifact(
            app_name=app_name,
            user_id=user_id,
            filename=filename,
        )
        text = artifact.text or ""
        with open(upload_path, "w", encoding="utf-8") as f:
            f.write(text)


async def remove_artifacts(
    app_name: str,
    user_id: str,
) -> None:
    artifact_service = FileArtifactService(root_dir=ARTIFACTS_DIR)
    artifact_keys = await artifact_service.list_artifact_keys(
        app_name=app_name,
        user_id=user_id,
    )
    for filename in artifact_keys:
        await artifact_service.delete_artifact(
            app_name=app_name,
            user_id=user_id,
            filename=filename,
        )


async def async_main(
    project_path: Path,
    folder_name: str,
    user_id: str,
    model: str,
    pipeline: str,
    artifact: Optional[str],
    output_dir: Path,
    rm_artifacts: bool,
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
        "user_id": user_id,
        "app_name": "contractor",
    }

    if spec.requires_artifact and artifact:
        builder_kwargs["artifact"] = artifact

    runner = await spec.builder(**builder_kwargs)
    event_handler = build_handle_event(output_dir)

    _ = await runner.run(
        user_id=user_id,
        on_event=event_handler,
    )

    await save_artifact(
        app_name="contractor",
        user_id=user_id,
        output_dir=output_dir,
    )

    if rm_artifacts:
        await remove_artifacts(
            app_name="contractor",
            user_id=user_id,
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
@click.option(
    "--rm",
    is_flag=True,
    help="Remove artifacts after completion",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save the artifacts",
)
def main(
    pipeline: str,
    project_path: Path,
    folder_name: str,
    artifact: Optional[Path],
    output: Optional[Path],
    user_id: str,
    model: str,
    rm: bool,
) -> None:
    """Run contractor task pipeline for a project."""
    pipeline = pipeline.lower()
    project_path = _validate_project_path(project_path)
    folder_name = _validate_folder_name(project_path, folder_name)
    artifact_text = _read_artifact_file(artifact)
    output_dir = output if output else project_path / ".contractor"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipelines = get_pipelines()
    spec = pipelines[pipeline]

    if artifact is not None and not spec.requires_artifact:
        raise click.UsageError(f"--artifact cannot be used with --pipeline {pipeline}")

    asyncio.run(
        async_main(
            project_path=project_path,
            folder_name=folder_name,
            user_id=user_id,
            model=model,
            pipeline=pipeline,
            artifact=artifact_text,
            rm_artifacts=rm,
            output_dir=output_dir,
        )
    )


if __name__ == "__main__":
    main()