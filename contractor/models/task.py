import os
import yaml

from pathlib import Path
from typing import Any

TASKS_BASE_DIR = Path(__file__).parent.parent / "tasks"


class Task:
    name: str
    objective: str
    instructions: str
    output_format: str
    _max_iterations: int = 1
    _format: str = "json"

    def load(name: str, variables: dict[str, Any]):
        fname = TASKS_BASE_DIR / f"{name}.yml"
        if not os.path.exists(fname):
            raise ValueError(f"Task {name} not found")
        with open(fname, "r") as f:
            data = yaml.safe_load(f)

        task = Task(**data["task"])
        task.objective = task.objective.format(**variables)
        task.instructions = task.instructions.format(**variables)
        task.output_format = task.output_format.format(**variables)
        return task
