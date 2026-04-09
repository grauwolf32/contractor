from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RouteTarget(str, Enum):
    DIRECT = "direct"
    REPO = "repo"
    OAS_BUILD = "oas_build"
    OAS_ENRICH = "oas_enrich"
    TRACE = "trace"
    ARTIFACTS = "artifacts"
    CODE_EDIT = "code_edit"


@dataclass(slots=True)
class ChatContext:
    project_path: str
    folder_name: str
    user_id: str
    app_name: str
    model: str
    session_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatTurn:
    user_text: str
    assistant_text: str = ""
    plan: str = ""
    route: RouteTarget = RouteTarget.DIRECT
    artifacts: list[str] = field(default_factory=list)