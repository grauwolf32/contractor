import os
from pathlib import Path

import yaml

TASKS_BASE_DIR = Path(__file__).parent.parent / "tasks"


class Task:
    name: str
    objective: str
    instructions: str
    output_format: str
    _repeats: int = 1
    _format: str = "json"

    def load(name: str):
        fname = TASKS_BASE_DIR / f"{name}.yml"
        if not os.path.exists(fname):
            raise ValueError(f"Task {name} not found")
        with open(fname, "r") as f:
            data = yaml.safe_load(f)
        return Task(**data["task"])
