from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
import tree_sitter_language_pack as ts_pack
from dotenv import load_dotenv
from google.adk.artifacts import FileArtifactService

from contractor.runners.task_runner import (
    TaskRunnerEvent,
    TaskRunnerEventHandler,
)

from cli.metrics import MetricsSink
from cli.pipelines import PipelineContext, get_pipelines
from cli.render import _render_event
from cli.ui import LiveRenderer, interactive_prompt, render_artifact_summary
from cli.utils import (
    remove_artifacts,
    save_artifact,
    validate_folder_name,
    validate_project_path,
)

PROMPT_REQUIRED_PIPELINES = frozenset({"router"})

load_dotenv()

APP_NAME = "contractor"
ARTIFACTS_DIR: Path = Path(__file__).parent.parent / "artifacts"

_QUIET_LOGGERS = (
    "httpcore",
    "fsspec",
    "google_adk",
    "google_genai",
    "openai",
    "litellm",
    "LiteLLM",
    "asyncio",
    "contractor",
    "urllib3",
    "opentelemetry",
)

_UI_STOP_EVENTS = frozenset({"run_finished", "task_failed", "pipeline_finished"})

# High-volume / non-user-facing events. Persisted to metrics.jsonl when they
# match, but never forwarded to the live UI (they would just flood it).
_UI_SKIP_EVENT_TYPES = frozenset(
    {"adk_before_run", "adk_after_run", "adk_event"}
)


def _setup_logging() -> None:
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _read_artifact_file(
    _ctx: click.Context, _param: click.Parameter, value: Optional[Path]
) -> Optional[str]:
    """Click callback: validate --artifact path and return its UTF-8 contents."""
    if value is None:
        return None

    artifact_path = value.resolve()

    if not artifact_path.exists():
        raise click.BadParameter(
            f"File does not exist: {artifact_path}", param_hint="--artifact"
        )

    if not artifact_path.is_file():
        raise click.BadParameter(
            f"Path is not a file: {artifact_path}", param_hint="--artifact"
        )

    try:
        return artifact_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise click.BadParameter(
            f"Artifact file must be a UTF-8 text file: {artifact_path}",
            param_hint="--artifact",
        ) from exc


def _build_event_handler(
    output_dir: Path,
    pipeline: str,
    enable_ui: bool,
) -> TaskRunnerEventHandler:
    metrics = MetricsSink(output_dir)
    ui = LiveRenderer(pipeline_name=pipeline) if enable_ui else None
    if ui is not None:
        ui.start()

    async def handle(event: TaskRunnerEvent) -> None:
        event_type = getattr(event, "type", "") or ""

        if metrics.matches(event):
            await metrics.write(event)

        if event_type.startswith("metrics_") or event_type in _UI_SKIP_EVENT_TYPES:
            return

        if ui is not None:
            ui.on_event(event)
            if event_type in _UI_STOP_EVENTS:
                ui.stop()
            return

        if output := _render_event(event):
            print(output)

    return handle


async def async_main(
    *,
    project_path: Path,
    folder_name: str,
    user_id: str,
    model: str,
    pipeline: str,
    artifact: Optional[str],
    prompt: Optional[str],
    output_dir: Path,
    rm_artifacts: bool,
    enable_ui: bool,
) -> None:
    ts_pack.init({"cache_dir": ts_pack.cache_dir()})

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_service = FileArtifactService(root_dir=ARTIFACTS_DIR)

    if rm_artifacts:
        await remove_artifacts(
            app_name=APP_NAME,
            user_id=user_id,
            artifact_service=artifact_service,
        )

    pipeline_cls = get_pipelines()[pipeline]
    ctx = PipelineContext(
        project_path=project_path,
        folder_name=folder_name,
        model=model,
        app_name=APP_NAME,
        user_id=user_id,
        artifact_service=artifact_service,
        artifact=artifact,
        prompt=prompt,
    )

    runner = pipeline_cls(ctx)
    handler = _build_event_handler(output_dir, pipeline, enable_ui=enable_ui)

    await runner.run(user_id=user_id, on_event=handler)

    saved_paths = await save_artifact(
        app_name=APP_NAME,
        user_id=user_id,
        output_dir=output_dir,
        artifact_service=artifact_service,
    )
    render_artifact_summary(output_dir, saved_paths)


@click.command(name=APP_NAME)
@click.option(
    "--pipeline",
    type=click.Choice(sorted(get_pipelines().keys()), case_sensitive=False),
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
    callback=_read_artifact_file,
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
    "--prompt",
    type=str,
    default=None,
    help=(
        "User prompt for prompt-driven pipelines (e.g. router). "
        "Required with --no-ui; otherwise an interactive input screen is shown."
    ),
)
@click.option("--rm", is_flag=True, help="Remove previous artifacts")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save the artifacts",
)
@click.option("--no-ui", is_flag=True, help="disable ui")
def main(
    pipeline: str,
    project_path: Path,
    folder_name: str,
    artifact: Optional[str],
    prompt: Optional[str],
    output: Optional[Path],
    user_id: str,
    model: str,
    rm: bool,
    no_ui: bool,
) -> None:
    """Run contractor task pipeline for a project."""
    _setup_logging()

    pipeline = pipeline.lower()
    project_path = validate_project_path(project_path)
    folder_name = validate_folder_name(project_path, folder_name)
    output_dir = output if output else project_path / ".contractor"
    output_dir.mkdir(parents=True, exist_ok=True)

    if pipeline in PROMPT_REQUIRED_PIPELINES and not prompt:
        if no_ui:
            raise click.UsageError(
                f"--prompt is required for pipeline '{pipeline}' when --no-ui is set"
            )
        prompt = interactive_prompt(pipeline_name=pipeline)

    asyncio.run(
        async_main(
            project_path=project_path,
            folder_name=folder_name,
            user_id=user_id,
            model=model,
            pipeline=pipeline,
            artifact=artifact,
            prompt=prompt,
            rm_artifacts=rm,
            output_dir=output_dir,
            enable_ui=not no_ui,
        )
    )


if __name__ == "__main__":
    main()
