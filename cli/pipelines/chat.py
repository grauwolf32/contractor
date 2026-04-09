from __future__ import annotations

from pathlib import Path

from google.adk.artifacts import BaseArtifactService

from contractor.runners.chat_runner import ChatRunner


async def advanced_chat_pipeline(
    *,
    project_path: Path,
    folder_name: str,
    model: str,
    app_name: str,
    user_id: str,
    artifact_service: BaseArtifactService,
    **_: object,
) -> ChatRunner:
    return ChatRunner(
        project_path=str(project_path),
        folder_name=folder_name,
        user_id=user_id,
        model=model,
        app_name=app_name,
        artifact_service=artifact_service,
    )
