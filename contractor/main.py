import argparse
import asyncio
import json
import logging
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from google.adk.artifacts import FileArtifactService

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contractor",
        description="Run contractor task pipeline for a project",
    )

    parser.add_argument(
        "--project-path",
        type=Path,
        required=True,
        help="Path to the project directory",
    )
    parser.add_argument(
        "--folder-name",
        type=str,
        default="/",
        help="Project-relative folder path used inside task templates",
    )
    parser.add_argument(
        "--user-id",
        default="cli-user",
        help="User id for ADK session runner",
    )

    return parser.parse_args()


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

    runner.add_variable(name="project_path", value=folder_name)
    
    runner.add_task(
        name="dependency_information",
        worker_builder=swe_builder,
        iterations=1,
        max_attempts=3,
    )

    return runner


async def async_main() -> None:
    args = parse_args()

    runner = oas_builder(
        project_path=args.project_path,
        folder_name=args.folder_name,
    )

    results = await runner.run(
        user_id=args.user_id,
        on_event=handle_event,
    )

    results = make_jsonable(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
