import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

PROMPTS_BASE_DIR = Path(__file__).parent / "yml"
DEFAULT_ROLE: Final[str] = "Professional Software Engineer"


@dataclass
class TaskDescription:
    objective: str
    instructions: str = ""
    examples: str = ""

    def format(self) -> str:
        return (
            f"OBJECTIVE:\n{self.objective}\n\nINSTRUCTIONS:\n{self.instructions}\n\n"
            if self.instructions
            else f"EXAMPLES:\n{self.examples}\n\n"
            if self.examples
            else ""
        )


@dataclass
class SectionPrompts:
    fmt: str = ""
    role: str = ""

    tasks: dict[str, TaskDescription] = field(default_factory=dict)

    def format_task(self, task_description: TaskDescription) -> str:
        return (
            "ROLE:\n{self.role}\n\n"
            f"{task_description.format()}"
            "OUTPUT FORMAT:\n{self.fmt}\n\n"
        )

    def format(self, name: str = "general") -> str:
        task_description = self.tasks[name]
        return self.format_task(task_description=task_description)

    def load(self, name: str) -> SectionPrompts:
        fname = PROMPTS_BASE_DIR / f"{name}.yml"
        if not os.path.exists(fname):
            raise ValueError(f"prompt not found: {name}.yml")

        raw: dict[str, Any] = {}
        with open(fname) as f:
            raw = yaml.safe_load(f)

        if fmt := raw.pop("format", ""):
            self.fmt = fmt
        if role := raw.pop("role", DEFAULT_ROLE):
            self.role = role

        self.tasks = {k: TaskDescription(**v) for k, v in raw.items}
        return self
