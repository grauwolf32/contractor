from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from test_settings import base_arguments

from contractor_runtime.podman_settings import PODMAN_LIMIT_CEILINGS, PodmanSettings
from contractor_runtime.settings import parse_settings

IMAGE = "localhost/contractor-execution@sha256:" + "a" * 64


def test_podman_defaults_are_disabled_immutable_and_private(tmp_path: Path) -> None:
    settings = parse_settings(base_arguments(tmp_path), {})
    assert settings.podman == PodmanSettings()
    assert not settings.podman.enabled
    assert settings.podman.cpus == 2
    assert settings.podman.memory_bytes == 2 << 30
    assert settings.podman.pids == 256
    assert settings.podman.tmpfs_bytes == 256 << 20
    assert settings.podman.command_max_seconds == 300
    assert settings.podman.prepare_max_seconds == 30
    assert settings.podman.stop_grace_seconds == 5
    assert settings.podman.preview_bytes == 32 << 10
    assert settings.podman.output_max_bytes == 1 << 20
    with pytest.raises(FrozenInstanceError):
        settings.podman.enabled = True  # type: ignore[misc]
    policy = PodmanSettings(enabled=True, image=IMAGE, owner="service-private")
    assert IMAGE not in repr(policy) and "service-private" not in repr(policy)


def test_podman_cli_overrides_environment_without_provisioning(tmp_path: Path) -> None:
    environment = {
        "CONTRACTOR_PODMAN_ENABLED": "true",
        "CONTRACTOR_PODMAN_IMAGE": IMAGE,
        "CONTRACTOR_PODMAN_OWNER": "runtime-a",
        "CONTRACTOR_WORKSPACE_STORAGE": "local",
        "CONTRACTOR_WORKSPACE_WORK_ROOT": str(tmp_path / "project"),
        "CONTRACTOR_PODMAN_CPUS": "3.5",
        "CONTRACTOR_PODMAN_PIDS": "300",
    }
    settings = parse_settings([*base_arguments(tmp_path), "--podman-cpus", "4"], environment)
    assert settings.podman.enabled
    assert settings.podman.image == IMAGE
    assert settings.podman.owner == "runtime-a"
    assert settings.podman.cpus == 4
    assert settings.podman.pids == 300
    assert not (tmp_path / "project").exists()
    disabled = parse_settings([*base_arguments(tmp_path), "--podman-enabled", "false"], environment)
    assert not disabled.podman.enabled


@pytest.mark.parametrize("field", PODMAN_LIMIT_CEILINGS)
@pytest.mark.parametrize("kind", ["zero", "negative", "over", "boolean", "nan", "infinity"])
def test_every_podman_limit_is_finite_typed_and_bounded(field: str, kind: str) -> None:
    value = {
        "zero": 0,
        "negative": -1,
        "over": PODMAN_LIMIT_CEILINGS[field] + 1,
        "boolean": True,
        "nan": float("nan"),
        "infinity": float("inf"),
    }[kind]
    with pytest.raises(ValueError):
        replace(PodmanSettings(), **{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": True},
        {"enabled": True, "image": IMAGE},
        {"enabled": True, "owner": "service"},
        {"enabled": 1},
        {"image": "repo:latest"},
        {"image": "repo@sha256:abc"},
        {"image": "repo@sha256:" + "A" * 64},
        {"image": "/tmp/image"},
        {"image": "https://user:password@host/image"},
        {"image": "repo\n@sha256:" + "a" * 64},
        {"owner": ""},
        {"owner": "../runtime"},
        {"owner": "x" * 64},
        {"command_max_seconds": 59},
        {"preview_bytes": 1 << 20},
        {"memory_bytes": 1024},
        {"pids": 2.5},
    ],
)
def test_invalid_podman_policy_is_rejected_even_when_disabled(changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(PodmanSettings(), **changes)


@pytest.mark.parametrize("storage", [None, "memory"])
def test_enabled_podman_requires_local_storage(tmp_path: Path, storage: str | None) -> None:
    env = {
        "CONTRACTOR_PODMAN_ENABLED": "true",
        "CONTRACTOR_PODMAN_IMAGE": IMAGE,
        "CONTRACTOR_PODMAN_OWNER": "runtime-a",
    }
    if storage is not None:
        env["CONTRACTOR_WORKSPACE_STORAGE"] = storage
    with pytest.raises(SystemExit):
        parse_settings(base_arguments(tmp_path), env)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--podman-enabled", "yes"],
        ["--podman-cpus", "nan"],
        ["--podman-memory-bytes", "0"],
        ["--podman-network", "host"],
        ["--podman-engine", "remote"],
        ["--podman-mount", "/"],
        ["--podman-user", "root"],
    ],
)
def test_invalid_or_unsupported_podman_options_fail(tmp_path: Path, arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_settings([*base_arguments(tmp_path), *arguments], {})
