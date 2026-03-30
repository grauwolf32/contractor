import json
import fsspec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any
from functools import partial
from contractor.tools.fs import RootedLocalFileSystem, MemoryOverlayFileSystem
from google.adk.artifacts import BaseArtifactService
from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.task_runner import (
    TaskRunner,
    TaskRunnerEventHandler,
)
from google.adk.models import LiteLlm


async def trace_annotation_runner(
    *,
    folder_name: str,
    llm: LiteLlm,
    fs: fsspec.AbstractFileSystem,
    operation_id: str,
    operation_schema: str,
    artifact_service: BaseArtifactService,
    **kwargs,
) -> TaskRunner:
    runner = TaskRunner(
        name="contractor",
        artifact_service=artifact_service,
    )

    trace_builder = partial(
        build_trace_agent,
        name="trace_agent",
        fs=fs,
        model=llm,
        max_tokens=120000,
        enable_vuln_reporting=True,
    )

    runner.add_variable(name="project_path", value=folder_name)
    runner.add_variable(name="operation_id", value=operation_id)
    if operation_schema:
        runner.add_variable(name="operation_schema", value=operation_schema)

    runner.add_task(
        name="trace_annotation",
        worker_builder=trace_builder,
        iterations=1,
        max_attempts=3,
        max_steps=20,
        artifacts=["oas-openapi-building"],
        namespace="trace-annotation",
        model=llm,
    )

    return runner


TEST_DATA_PATH = (
    Path(__file__).parent.parent.parent / "tests" / "data" / "fakeproj" / "2"
)


@dataclass
class MockRunner:
    fs: MemoryOverlayFileSystem
    task_runner: TaskRunner

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> list[dict[str, Any]]:
        await self.task_runner.run(user_id=user_id, on_event=on_event)
        patch = self.fs.save()
        with open(str(TEST_DATA_PATH / "patch.json"), "w") as f:
            f.write(json.dumps(patch))


# Mock pipeline for tests
async def trace_annotation_pipeline(
    project_path: Path,
    folder_name: str,
    model: str,
    app_name: str,
    user_id: str,
    artifact_service: BaseArtifactService,
    artifact: Optional[str] = None,
    **kwargs,
) -> TaskRunner:
    llm = LiteLlm(model=model)
    base_fs = RootedLocalFileSystem(root_path=project_path)
    overlay_fs = MemoryOverlayFileSystem(fs=base_fs)

    if artifact:
        artifact_text = types.Part.from_text(text=artifact)
        await artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            filename="oas-openapi-building",
            artifact=artifact_text,
        )

    operation_id = "lostPasswordResetForm"
    spec_path = TEST_DATA_PATH / "spec.yml"

    raw: str = ""
    with open(spec_path) as f:
        raw = f.read()

    task_runner = await trace_annotation_runner(
        folder_name=folder_name,
        llm=llm,
        fs=overlay_fs,
        operation_id=operation_id,
        operation_schema=raw,
        artifact_service=artifact_service,
    )

    return MockRunner(fs=overlay_fs, task_runner=task_runner)
