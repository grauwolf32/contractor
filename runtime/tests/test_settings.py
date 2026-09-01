from __future__ import annotations

from pathlib import Path

import pytest

from contractor_runtime.settings import parse_settings


def test_settings_use_environment_and_cli_override(tmp_path: Path) -> None:
    files = settings_files(tmp_path)
    environment = {
        "CONTRACTOR_CONTROL_PLANE_URL": "https://control.example:8443",
        "CONTRACTOR_ADVERTISED_CONTROL_URL": "https://agent.example:9443",
        "CONTRACTOR_ADVERTISED_A2A_URL": "https://agent.example:9444",
        "CONTRACTOR_CA_FILE": str(files[0]),
        "CONTRACTOR_CERTIFICATE_FILE": str(files[1]),
        "CONTRACTOR_PRIVATE_KEY_FILE": str(files[2]),
        "CONTRACTOR_RUNTIME_LISTEN": "127.0.0.1:9000",
        "CONTRACTOR_WORK_ROOT": str(tmp_path / "work"),
        "CONTRACTOR_INITIAL_LABELS": "debug,caido",
        "CONTRACTOR_RUNTIME_ADAPTERS": "otlp-http@1,http-proxy@1",
    }
    settings = parse_settings(
        [
            "--listen",
            "127.0.0.1:9001",
            "--log-level",
            "debug",
            "--initial-label",
            "site_proxy",
            "--runtime-adapter",
            "otlp-http@1",
        ],
        environment,
    )
    assert settings.listen_address == "127.0.0.1:9001"
    assert settings.log_level == "debug"
    assert settings.control_plane_url == "https://control.example:8443"
    assert settings.initial_labels == ("site_proxy",)
    assert settings.enabled_runtime_adapters == ("otlp-http@1",)
    assert str(files[2]) not in repr(settings)


def test_initial_labels_are_sorted_and_environment_is_used_without_cli(tmp_path: Path) -> None:
    settings = parse_settings(
        base_arguments(tmp_path), {"CONTRACTOR_INITIAL_LABELS": "debug,caido"}
    )
    assert settings.initial_labels == ("caido", "debug")


def test_runtime_adapter_environment_is_sorted_and_unset_means_all(tmp_path: Path) -> None:
    configured = parse_settings(
        base_arguments(tmp_path),
        {"CONTRACTOR_RUNTIME_ADAPTERS": "otlp-http@1,http-proxy@1"},
    )
    defaulted = parse_settings(base_arguments(tmp_path), {})
    assert configured.enabled_runtime_adapters == ("http-proxy@1", "otlp-http@1")
    assert defaulted.enabled_runtime_adapters is None


def test_caido_runtime_adapter_can_be_selected_explicitly(tmp_path: Path) -> None:
    settings = parse_settings(
        [*base_arguments(tmp_path), "--runtime-adapter", "caido-graphql@1"],
        {},
    )
    assert settings.enabled_runtime_adapters == ("caido-graphql@1",)


@pytest.mark.parametrize(
    "value",
    ["", "unknown@1", "otlp-http@1,otlp-http@1", "otlp-http@1,"],
)
def test_invalid_runtime_adapter_allowlist_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_settings(
            base_arguments(tmp_path),
            {"CONTRACTOR_RUNTIME_ADAPTERS": value},
        )


@pytest.mark.parametrize(
    "labels",
    [
        ["default"],
        ["Debug"],
        ["debug", "debug"],
        ["x" * 64],
        [f"label_{index}" for index in range(33)],
    ],
)
def test_invalid_initial_labels_are_rejected(tmp_path: Path, labels: list[str]) -> None:
    arguments = base_arguments(tmp_path)
    for label in labels:
        arguments.extend(("--initial-label", label))
    with pytest.raises(SystemExit):
        parse_settings(arguments, {})


def test_required_private_settings_are_enforced() -> None:
    with pytest.raises(SystemExit):
        parse_settings([], {})


@pytest.mark.parametrize("listen", ["", "localhost", ":9000", "localhost:not-a-port", "x:70000"])
def test_invalid_listen_is_rejected(tmp_path: Path, listen: str) -> None:
    with pytest.raises(SystemExit):
        parse_settings([*base_arguments(tmp_path), "--listen", listen], {})


@pytest.mark.parametrize(
    "option,value",
    [
        ("--control-plane-url", "http://control.example"),
        ("--control-plane-url", "https://user:secret@control.example"),
        ("--advertised-control-url", "https://agent.example/path?query=yes"),
        ("--advertised-a2a-url", "not-a-url"),
    ],
)
def test_private_urls_must_be_credential_free_https(
    tmp_path: Path, option: str, value: str
) -> None:
    arguments = base_arguments(tmp_path)
    index = arguments.index(option)
    arguments[index + 1] = value
    with pytest.raises(SystemExit):
        parse_settings(arguments, {})


def test_private_key_permissions_and_timing_are_strict(tmp_path: Path) -> None:
    arguments = base_arguments(tmp_path)
    key = Path(arguments[arguments.index("--private-key-file") + 1])
    key.chmod(0o644)
    with pytest.raises(SystemExit):
        parse_settings(arguments, {})
    key.chmod(0o600)
    with pytest.raises(SystemExit):
        parse_settings(
            [*arguments, "--heartbeat-interval-seconds", "60", "--confirmed-lease-seconds", "10"],
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
        str(tmp_path / "work"),
    ]


def settings_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = (tmp_path / "ca.crt", tmp_path / "agent.crt", tmp_path / "agent.key")
    for path in paths:
        if not path.exists():
            path.write_text("test", encoding="utf-8")
    paths[2].chmod(0o600)
    return paths
