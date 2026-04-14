import yaml

from pydantic import BaseModel
from typing import Any
from pathlib import Path

AGENTS_PATH = Path(__file__).parent.parent / "agents"


class _Prompt(BaseModel):
    text: str


class _PromptItem(BaseModel):
    prompt: _Prompt


def load_prompt(agent_name: str):
    fpath = AGENTS_PATH / agent_name / "prompt.yml"
    if not fpath.exists():
        raise ValueError(f"Could not find prompt for {agent_name} in {fpath}")

    raw: dict[str, Any] = {}
    with open(fpath) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Prompt file {fpath} should contain list")

    item: _PromptItem = _PromptItem.model_validate(raw[0])
    return item.prompt.text
