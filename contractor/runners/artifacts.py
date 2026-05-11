from __future__ import annotations

import json
from typing import Any

from google.adk.artifacts import BaseArtifactService
from google.genai import types

from contractor.runners.models import ArtifactKind

ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("result", "summary", "records")


class InvalidArtifactKeyError(ValueError):
    """Raised when an artifact key is invalid."""


def validate_artifact_key(key: str) -> str:
    """Return a cleaned artifact key or raise."""
    cleaned = (key or "").strip().strip("/")
    if not cleaned:
        raise InvalidArtifactKeyError("artifact key must not be empty")
    if ".." in cleaned.split("/"):
        raise InvalidArtifactKeyError(
            "artifact key must not contain path traversal segments"
        )
    return cleaned


def artifact_filename(key: str, kind: ArtifactKind) -> str:
    return f"{validate_artifact_key(key)}/{kind}"


def artifact_names_for_key(key: str) -> dict[ArtifactKind, str]:
    return {kind: artifact_filename(key, kind) for kind in ARTIFACT_KINDS}


def _records_to_text(records: Any) -> str:
    if isinstance(records, str):
        return records
    return json.dumps(records or [], ensure_ascii=False)


async def save_result_artifacts(
    *,
    artifact_service: BaseArtifactService,
    app_name: str,
    user_id: str,
    key: str,
    result: str = "",
    summary: str = "",
    records: Any = None,
) -> dict[ArtifactKind, str]:
    """Persist ``result``/``summary``/``records`` artifacts under ``{key}/{kind}``.

    Returns the ``{kind: filename}`` mapping for the saved artifacts.
    """
    payloads: dict[ArtifactKind, str] = {
        "result": result or "",
        "summary": summary or "",
        "records": _records_to_text(records),
    }

    names: dict[ArtifactKind, str] = {}
    for kind, text in payloads.items():
        filename = artifact_filename(key, kind)
        await artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=None,
            filename=filename,
            artifact=types.Part.from_text(text=text),
        )
        names[kind] = filename
    return names
