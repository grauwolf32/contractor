from pathlib import Path

import pytest

from contractor.utils import prompt as prompt_module


@pytest.fixture()
def fake_agents_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(prompt_module, "AGENTS_PATH", tmp_path)
    return tmp_path


def _write_agent(
    agents_dir: Path,
    name: str,
    *,
    manifest: str,
    files: dict[str, str],
) -> Path:
    agent_dir = agents_dir / name
    (agent_dir / "prompts").mkdir(parents=True)
    (agent_dir / "prompt.yml").write_text(manifest, encoding="utf-8")
    for rel, body in files.items():
        (agent_dir / rel).write_text(body, encoding="utf-8")
    return agent_dir


def test_load_prompt_returns_active_version_text(fake_agents_dir: Path):
    _write_agent(
        fake_agents_dir,
        "demo",
        manifest=(
            "active: v1\n"
            "versions:\n"
            "  v1:\n"
            "    file: prompts/v1.md\n"
        ),
        files={"prompts/v1.md": "hello world\n"},
    )

    assert prompt_module.load_prompt("demo") == "hello world\n"


def test_load_prompt_resolves_active_among_multiple_versions(fake_agents_dir: Path):
    _write_agent(
        fake_agents_dir,
        "demo",
        manifest=(
            "active: v2\n"
            "versions:\n"
            "  v2:\n"
            "    file: prompts/v2.md\n"
            "  v1:\n"
            "    file: prompts/v1.md\n"
        ),
        files={
            "prompts/v1.md": "first\n",
            "prompts/v2.md": "second\n",
        },
    )

    assert prompt_module.load_prompt("demo") == "second\n"


def test_load_prompt_can_pin_explicit_version(fake_agents_dir: Path):
    _write_agent(
        fake_agents_dir,
        "demo",
        manifest=(
            "active: v2\n"
            "versions:\n"
            "  v2:\n"
            "    file: prompts/v2.md\n"
            "  v1:\n"
            "    file: prompts/v1.md\n"
        ),
        files={
            "prompts/v1.md": "first\n",
            "prompts/v2.md": "second\n",
        },
    )

    assert prompt_module.load_prompt("demo", version="v1") == "first\n"


def test_load_prompt_raises_when_manifest_missing(fake_agents_dir: Path):
    with pytest.raises(ValueError, match="Could not find prompt manifest"):
        prompt_module.load_prompt("does_not_exist")


def test_load_prompt_raises_on_non_mapping_manifest(fake_agents_dir: Path):
    agent_dir = fake_agents_dir / "demo"
    agent_dir.mkdir()
    (agent_dir / "prompt.yml").write_text("- not: a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        prompt_module.load_prompt("demo")


def test_load_prompt_raises_when_version_unknown(fake_agents_dir: Path):
    _write_agent(
        fake_agents_dir,
        "demo",
        manifest=(
            "active: v1\n"
            "versions:\n"
            "  v1:\n"
            "    file: prompts/v1.md\n"
        ),
        files={"prompts/v1.md": "hi\n"},
    )

    with pytest.raises(ValueError, match="not declared"):
        prompt_module.load_prompt("demo", version="v99")


def test_load_prompt_raises_when_referenced_file_missing(fake_agents_dir: Path):
    agent_dir = fake_agents_dir / "demo"
    (agent_dir / "prompts").mkdir(parents=True)
    (agent_dir / "prompt.yml").write_text(
        "active: v1\nversions:\n  v1:\n    file: prompts/v1.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing file"):
        prompt_module.load_prompt("demo")


def test_load_prompt_with_version_returns_resolved_active(fake_agents_dir: Path):
    _write_agent(
        fake_agents_dir,
        "demo",
        manifest=(
            "active: v2\n"
            "versions:\n"
            "  v2:\n"
            "    file: prompts/v2.md\n"
            "  v1:\n"
            "    file: prompts/v1.md\n"
        ),
        files={"prompts/v1.md": "first\n", "prompts/v2.md": "second\n"},
    )

    text, resolved = prompt_module.load_prompt_with_version("demo")
    assert text == "second\n"
    assert resolved == "v2"


def test_load_prompt_with_version_returns_pinned(fake_agents_dir: Path):
    _write_agent(
        fake_agents_dir,
        "demo",
        manifest=(
            "active: v2\n"
            "versions:\n"
            "  v2:\n"
            "    file: prompts/v2.md\n"
            "  v1:\n"
            "    file: prompts/v1.md\n"
        ),
        files={"prompts/v1.md": "first\n", "prompts/v2.md": "second\n"},
    )

    text, resolved = prompt_module.load_prompt_with_version("demo", version="v1")
    assert text == "first\n"
    assert resolved == "v1"
