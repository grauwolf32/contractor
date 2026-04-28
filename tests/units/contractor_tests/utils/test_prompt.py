from pathlib import Path

import pytest

from contractor.utils import prompt as prompt_module


@pytest.fixture()
def fake_agents_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(prompt_module, "AGENTS_PATH", tmp_path)
    return tmp_path


def test_load_prompt_returns_text_from_first_item(fake_agents_dir: Path):
    agent_dir = fake_agents_dir / "demo"
    agent_dir.mkdir()
    (agent_dir / "prompt.yml").write_text(
        "- prompt:\n    text: |\n      hello world\n",
        encoding="utf-8",
    )

    text = prompt_module.load_prompt("demo")
    assert text.strip() == "hello world"


def test_load_prompt_picks_first_item_when_multiple(fake_agents_dir: Path):
    agent_dir = fake_agents_dir / "demo"
    agent_dir.mkdir()
    (agent_dir / "prompt.yml").write_text(
        "- prompt:\n"
        "    text: first\n"
        "- prompt:\n"
        "    text: second\n",
        encoding="utf-8",
    )

    assert prompt_module.load_prompt("demo") == "first"


def test_load_prompt_raises_when_dir_missing(fake_agents_dir: Path):
    with pytest.raises(ValueError, match="Could not find prompt"):
        prompt_module.load_prompt("does_not_exist")


def test_load_prompt_raises_on_non_list_root(fake_agents_dir: Path):
    agent_dir = fake_agents_dir / "demo"
    agent_dir.mkdir()
    (agent_dir / "prompt.yml").write_text(
        "prompt:\n  text: not a list\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="should contain list"):
        prompt_module.load_prompt("demo")


def test_load_prompt_raises_on_invalid_item_shape(fake_agents_dir: Path):
    agent_dir = fake_agents_dir / "demo"
    agent_dir.mkdir()
    (agent_dir / "prompt.yml").write_text("- not_a_prompt: 1\n", encoding="utf-8")

    # Pydantic validation error bubbles up.
    with pytest.raises(Exception):
        prompt_module.load_prompt("demo")
