import yaml

from pydantic import BaseModel
from pathlib import Path

AGENTS_PATH = Path(__file__).parent.parent / "agents"


class _Version(BaseModel):
    file: str


class _Manifest(BaseModel):
    active: str
    versions: dict[str, _Version]


def _read_manifest(agent_name: str) -> tuple[Path, _Manifest]:
    agent_dir = AGENTS_PATH / agent_name
    manifest_path = agent_dir / "prompt.yml"
    if not manifest_path.exists():
        raise ValueError(
            f"Could not find prompt manifest for {agent_name} at {manifest_path}"
        )
    with open(manifest_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt manifest {manifest_path} must be a mapping")
    return agent_dir, _Manifest.model_validate(raw)


def load_prompt_with_version(
    agent_name: str, version: str | None = None
) -> tuple[str, str]:
    """Return `(text, resolved_version)` for the requested or active prompt."""
    agent_dir, manifest = _read_manifest(agent_name)
    selected = version or manifest.active
    if selected not in manifest.versions:
        raise ValueError(
            f"Prompt version {selected!r} not declared in {agent_dir / 'prompt.yml'}"
        )
    text_path = agent_dir / manifest.versions[selected].file
    if not text_path.exists():
        raise ValueError(
            f"Prompt version {selected!r} references missing file {text_path}"
        )
    return text_path.read_text(encoding="utf-8"), selected


def load_prompt(agent_name: str, version: str | None = None) -> str:
    return load_prompt_with_version(agent_name, version)[0]


def all_active_prompt_versions() -> dict[str, str]:
    """Snapshot ``{agent_name: active_version}`` for every agent with a manifest.

    Used by the task runner to emit prompt-version provenance on ``run_started``
    so metrics consumers can slice runs by which prompt was active.
    Agents without a ``prompt.yml`` or with broken manifests are skipped silently.
    """
    versions: dict[str, str] = {}
    if not AGENTS_PATH.is_dir():
        return versions
    for entry in sorted(AGENTS_PATH.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "prompt.yml").exists():
            continue
        try:
            _, resolved = load_prompt_with_version(entry.name)
        except Exception:
            continue
        versions[entry.name] = resolved
    return versions
