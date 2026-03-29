from pathlib import Path
from datetime import datetime
from google.adk.artifacts import BaseArtifactService
from datetime import datetime, timezone

async def save_artifact(
    app_name: str,
    user_id: str,
    output_dir: Path,
    artifact_service: BaseArtifactService,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    artifact_service: BaseArtifactService,
) -> None:
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

def normalize_folder_name(folder_name: str) -> str:
    if folder_name in ("", "/"):
        return "/"

    normalized = Path(folder_name.lstrip("/")).as_posix().strip("/")
    return f"/{normalized}" if normalized else "/"

def validate_folder_name(project_path: Path, folder_name: str) -> str:
    normalized_folder = normalize_folder_name(folder_name)

    if normalized_folder == "/":
        target_dir = project_path
    else:
        target_dir = (project_path / normalized_folder.lstrip("/")).resolve()

    try:
        target_dir.relative_to(project_path)
    except ValueError as exc:
        raise ValueError(
            "--folder-name must point to a directory inside --project-path"
            ) from exc

    if not target_dir.exists():
        raise ValueError(
            f"Directory does not exist: {target_dir}",
        )

    if not target_dir.is_dir():
        raise ValueError(
            f"Path is not a directory: {target_dir}",
        )

    return normalized_folder

def validate_project_path(project_path: Path) -> Path:
    project_path = project_path.resolve()

    if not project_path.exists():
        raise ValueError(
            f"Directory does not exist: {project_path}",
        )

    if not project_path.is_dir():
        raise ValueError(
            f"Path is not a directory: {project_path}",
        )

    return project_path

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()