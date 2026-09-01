from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from contractor_runtime.settings import (
    LOCAL_WORKSPACE_DEFAULTS,
    MEMORY_WORKSPACE_DEFAULTS,
    WORKSPACE_LIMIT_CEILINGS,
    parse_settings,
)


def test_omitted_workspace_configuration_disables_provider(tmp_path: Path) -> None:
    settings = parse_settings(base_arguments(tmp_path), {})
    assert settings.workspace is None


@pytest.mark.parametrize(
    ("storage", "defaults"),
    [("local", LOCAL_WORKSPACE_DEFAULTS), ("memory", MEMORY_WORKSPACE_DEFAULTS)],
)
def test_workspace_defaults_are_storage_specific_and_safe(
    tmp_path: Path, storage: str, defaults: tuple[int, int, int, int]
) -> None:
    arguments = [*base_arguments(tmp_path), "--workspace-storage", storage]
    private_root = tmp_path / "recognizable-private-workspace-root"
    if storage == "local":
        arguments.extend(("--workspace-work-root", str(private_root)))
    settings = parse_settings(arguments, {})
    assert settings.workspace is not None
    assert settings.workspace.storage == storage
    limits = settings.workspace.limits
    assert (
        limits.max_files,
        limits.max_expanded_bytes,
        limits.max_managed_text_bytes,
        limits.max_file_bytes,
    ) == defaults
    assert str(private_root) not in repr(settings)
    with pytest.raises(FrozenInstanceError):
        settings.workspace.storage = "memory"  # type: ignore[misc]


def test_workspace_environment_and_cli_limit_override(tmp_path: Path) -> None:
    environment = {
        "CONTRACTOR_WORKSPACE_STORAGE": "memory",
        "CONTRACTOR_WORKSPACE_MAX_FILES": "20",
        "CONTRACTOR_WORKSPACE_MAX_EXPANDED_BYTES": "400",
        "CONTRACTOR_WORKSPACE_MAX_MANAGED_TEXT_BYTES": "300",
        "CONTRACTOR_WORKSPACE_MAX_FILE_BYTES": "100",
    }
    settings = parse_settings(
        [*base_arguments(tmp_path), "--workspace-max-files", "21"], environment
    )
    assert settings.workspace is not None
    assert settings.workspace.storage == "memory"
    assert settings.workspace.limits.max_files == 21
    assert settings.workspace.limits.max_expanded_bytes == 400


@pytest.mark.parametrize(
    "extra",
    [
        ["--workspace-max-files", "10"],
        ["--workspace-storage", "local"],
        ["--workspace-storage", "local", "--workspace-work-root", "relative/path"],
        ["--workspace-storage", "local", "--workspace-work-root", "/"],
        ["--workspace-storage", "memory", "--workspace-work-root", "/tmp/not-used"],
        ["--workspace-storage", "memory", "--workspace-max-files", "0"],
        [
            "--workspace-storage",
            "memory",
            "--workspace-max-expanded-bytes",
            "10",
            "--workspace-max-file-bytes",
            "11",
        ],
        [
            "--workspace-storage",
            "memory",
            "--workspace-max-files",
            str(WORKSPACE_LIMIT_CEILINGS[0] + 1),
        ],
    ],
)
def test_invalid_workspace_configuration_is_rejected(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_settings([*base_arguments(tmp_path), *extra], {})


def test_local_workspace_root_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "workspace-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(SystemExit):
        parse_settings(
            [
                *base_arguments(tmp_path),
                "--workspace-storage",
                "local",
                "--workspace-work-root",
                str(link),
            ],
            {},
        )


def base_arguments(tmp_path: Path) -> list[str]:
    ca, certificate, key = settings_files(tmp_path)
    return [
        "--control-plane-url",
        "https://localhost:8443",
        "--advertised-control-url",
        "https://localhost:9443",
        "--advertised-a2a-url",
        "https://localhost:9444",
        "--ca-file",
        str(ca),
        "--certificate-file",
        str(certificate),
        "--private-key-file",
        str(key),
        "--work-root",
        str(tmp_path / "sandbox-work"),
    ]


def settings_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = (tmp_path / "ca.crt", tmp_path / "agent.crt", tmp_path / "agent.key")
    for path in paths:
        path.write_text("test", encoding="utf-8")
    paths[2].chmod(0o600)
    return paths
