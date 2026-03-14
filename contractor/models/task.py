from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

TASKS_BASE_DIR = Path(__file__).parent.parent / "tasks"


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return normalized or "task"


@dataclass(slots=True, frozen=True)
class TaskTemplate:
    key: str
    title: str

    objective: str
    instructions: str
    output_format: str

    default_artifacts: list[str] = field(default_factory=list)
    default_iterations: int = 1
    format: str = "json"

    @classmethod
    def load(cls, name: str) -> "TaskTemplate":
        template_key = Path(name).stem
        fname = TASKS_BASE_DIR / f"{template_key}.yml"
        if not os.path.exists(fname):
            raise ValueError(f"Task template '{template_key}' not found")

        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw = data["task"]

        return cls(
            key=template_key,
            title=raw.get("name", template_key) or template_key,
            objective=raw["objective"],
            instructions=raw["instructions"],
            output_format=raw["output_format"],
            default_artifacts=list(raw.get("artifacts", []) or []),
            default_iterations=int(raw.get("iterations", 1) or 1),
            format=raw.get("format", "json") or "json",
        )


@dataclass(slots=True, frozen=True)
class RenderedTask:
    key: str
    title: str

    objective: str
    instructions: str
    output_format: str
    format: str

    @classmethod
    def from_template(
        cls,
        template: TaskTemplate,
        *,
        variables: Mapping[str, Any],
        params: Mapping[str, Any],
        artifacts: Mapping[str, Mapping[str, str]],
    ) -> "RenderedTask":
        scope: dict[str, Any] = dict(variables)
        scope.update(params)

        scope["artifacts"] = yaml.safe_dump(
            {k: dict(v) for k, v in artifacts.items()},
            allow_unicode=True,
            sort_keys=False,
        )

        for producer_task_key, payloads in artifacts.items():
            producer_key = _normalize_name(producer_task_key)
            for kind, value in payloads.items():
                scope[f"artifact__{producer_key}__{kind}"] = value

        return cls(
            key=template.key,
            title=template.title,
            objective=template.objective.format(**scope),
            instructions=template.instructions.format(**scope),
            output_format=template.output_format.format(**scope),
            format=template.format,
        )
