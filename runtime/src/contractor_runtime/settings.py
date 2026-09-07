"""Strict process settings for the single-slot Runtime Agent."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from contractor_runtime.contracts import WorkspaceStorageV2
from contractor_runtime.podman_settings import PodmanSettings, add_podman_arguments, podman_settings

DEFAULT_WORKSPACE_OPERATION_TIMEOUT_SECONDS = 30.0

LOCAL_WORKSPACE_DEFAULTS = (50_000, 2 << 30, 256 << 20, 16 << 20)
MEMORY_WORKSPACE_DEFAULTS = (10_000, 256 << 20, 64 << 20, 4 << 20)
WORKSPACE_LIMIT_CEILINGS = (1_000_000, 64 << 30, 16 << 30, 2 << 30)


@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    max_files: int
    max_expanded_bytes: int
    max_managed_text_bytes: int
    max_file_bytes: int


@dataclass(frozen=True, slots=True)
class WorkspaceSettings:
    storage: WorkspaceStorageV2
    limits: WorkspaceLimits
    work_root: Path | None = field(default=None, repr=False)
    operation_timeout_seconds: float = DEFAULT_WORKSPACE_OPERATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not math.isfinite(self.operation_timeout_seconds) or self.operation_timeout_seconds <= 0:
            raise ValueError("workspace operation timeout must be finite and positive")


@dataclass(frozen=True, slots=True)
class Settings:
    control_plane_url: str
    advertised_control_url: str
    advertised_a2a_url: str
    ca_file: Path
    certificate_file: Path
    private_key_file: Path = field(repr=False)
    initial_labels: tuple[str, ...] = ()
    enabled_runtime_adapters: tuple[str, ...] | None = None
    workspace: WorkspaceSettings | None = None
    podman: PodmanSettings = field(default_factory=PodmanSettings)
    host: str = "127.0.0.1"
    port: int = 9443
    heartbeat_interval_seconds: float = 10.0
    confirmed_lease_seconds: float = 60.0
    request_timeout_seconds: float = 5.0
    shutdown_grace_seconds: float = 10.0
    work_root: Path = Path(".local/runtime/work")
    log_level: str = "info"

    @property
    def listen_address(self) -> str:
        return f"{self.host}:{self.port}"


def parse_settings(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    values = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(prog="contractor-runtime")
    add_podman_arguments(parser, values)
    parser.add_argument("--control-plane-url", default=values.get("CONTRACTOR_CONTROL_PLANE_URL"))
    parser.add_argument(
        "--advertised-control-url", default=values.get("CONTRACTOR_ADVERTISED_CONTROL_URL")
    )
    parser.add_argument("--advertised-a2a-url", default=values.get("CONTRACTOR_ADVERTISED_A2A_URL"))
    parser.add_argument("--ca-file", default=values.get("CONTRACTOR_CA_FILE"))
    parser.add_argument("--certificate-file", default=values.get("CONTRACTOR_CERTIFICATE_FILE"))
    parser.add_argument("--private-key-file", default=values.get("CONTRACTOR_PRIVATE_KEY_FILE"))
    parser.add_argument("--initial-label", action="append", default=None)
    parser.add_argument("--runtime-adapter", action="append", default=None)
    parser.add_argument(
        "--workspace-storage",
        choices=("local", "memory"),
        default=values.get("CONTRACTOR_WORKSPACE_STORAGE"),
    )
    parser.add_argument(
        "--workspace-work-root", default=values.get("CONTRACTOR_WORKSPACE_WORK_ROOT")
    )
    parser.add_argument(
        "--workspace-operation-timeout-seconds",
        type=float,
        default=values.get("CONTRACTOR_WORKSPACE_OPERATION_TIMEOUT_SECONDS"),
    )
    parser.add_argument(
        "--workspace-max-files",
        type=int,
        default=values.get("CONTRACTOR_WORKSPACE_MAX_FILES"),
    )
    parser.add_argument(
        "--workspace-max-expanded-bytes",
        type=int,
        default=values.get("CONTRACTOR_WORKSPACE_MAX_EXPANDED_BYTES"),
    )
    parser.add_argument(
        "--workspace-max-managed-text-bytes",
        type=int,
        default=values.get("CONTRACTOR_WORKSPACE_MAX_MANAGED_TEXT_BYTES"),
    )
    parser.add_argument(
        "--workspace-max-file-bytes",
        type=int,
        default=values.get("CONTRACTOR_WORKSPACE_MAX_FILE_BYTES"),
    )
    parser.add_argument(
        "--listen",
        default=values.get("CONTRACTOR_RUNTIME_LISTEN", "127.0.0.1:9443"),
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=values.get("CONTRACTOR_HEARTBEAT_INTERVAL_SECONDS", "10"),
    )
    parser.add_argument(
        "--confirmed-lease-seconds",
        type=float,
        default=values.get("CONTRACTOR_CONFIRMED_LEASE_SECONDS", "60"),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=values.get("CONTRACTOR_REQUEST_TIMEOUT_SECONDS", "5"),
    )
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=float,
        default=values.get("CONTRACTOR_SHUTDOWN_GRACE_SECONDS", "10"),
    )
    parser.add_argument(
        "--work-root",
        default=values.get("CONTRACTOR_WORK_ROOT", ".local/runtime/work"),
    )
    parser.add_argument(
        "--log-level",
        default=values.get("CONTRACTOR_RUNTIME_LOG_LEVEL", "info"),
        choices=("debug", "info", "warning", "error"),
    )
    args = parser.parse_args(argv)

    required = {
        "--control-plane-url": args.control_plane_url,
        "--advertised-control-url": args.advertised_control_url,
        "--advertised-a2a-url": args.advertised_a2a_url,
        "--ca-file": args.ca_file,
        "--certificate-file": args.certificate_file,
        "--private-key-file": args.private_key_file,
    }
    for option, value in required.items():
        if not isinstance(value, str) or not value.strip():
            parser.error(f"{option} is required")

    host, port = _parse_listen(parser, args.listen)
    control_plane_url = _private_url(parser, "--control-plane-url", args.control_plane_url)
    advertised_control_url = _private_url(
        parser, "--advertised-control-url", args.advertised_control_url
    )
    advertised_a2a_url = _private_url(parser, "--advertised-a2a-url", args.advertised_a2a_url)
    ca_file = _regular_file(parser, "--ca-file", args.ca_file)
    certificate_file = _regular_file(parser, "--certificate-file", args.certificate_file)
    private_key_file = _regular_file(parser, "--private-key-file", args.private_key_file)
    if private_key_file.stat().st_mode & 0o077:
        parser.error("--private-key-file must not be accessible by group or other users")

    heartbeat = _positive(parser, "--heartbeat-interval-seconds", args.heartbeat_interval_seconds)
    lease = _positive(parser, "--confirmed-lease-seconds", args.confirmed_lease_seconds)
    request_timeout = _positive(parser, "--request-timeout-seconds", args.request_timeout_seconds)
    shutdown_grace = _positive(parser, "--shutdown-grace-seconds", args.shutdown_grace_seconds)
    if lease <= heartbeat:
        parser.error("--confirmed-lease-seconds must exceed --heartbeat-interval-seconds")
    work_root = Path(args.work_root).expanduser().resolve()
    if work_root == Path(work_root.anchor):
        parser.error("--work-root must not be a filesystem root")
    initial_labels = _initial_labels(parser, args.initial_label, values)
    enabled_runtime_adapters = _runtime_adapters(parser, args.runtime_adapter, values)
    workspace = _workspace_settings(parser, args)
    podman = podman_settings(parser, args)
    if podman.enabled and (workspace is None or workspace.storage != "local"):
        parser.error("enabled podman requires --workspace-storage local")

    return Settings(
        control_plane_url=control_plane_url,
        advertised_control_url=advertised_control_url,
        advertised_a2a_url=advertised_a2a_url,
        ca_file=ca_file,
        certificate_file=certificate_file,
        private_key_file=private_key_file,
        initial_labels=initial_labels,
        enabled_runtime_adapters=enabled_runtime_adapters,
        workspace=workspace,
        podman=podman,
        host=host,
        port=port,
        heartbeat_interval_seconds=heartbeat,
        confirmed_lease_seconds=lease,
        request_timeout_seconds=request_timeout,
        shutdown_grace_seconds=shutdown_grace,
        work_root=work_root,
        log_level=args.log_level,
    )


def _parse_listen(parser: argparse.ArgumentParser, raw: str) -> tuple[str, int]:
    try:
        host, port_text = raw.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, AttributeError) as error:
        parser.error(f"invalid --listen value: {type(error).__name__}")
    if not host:
        parser.error("--listen host must not be empty")
    if not 0 <= port <= 65535:
        parser.error("--listen port must be between 0 and 65535")
    return host, port


def _private_url(parser: argparse.ArgumentParser, option: str, raw: str) -> str:
    parsed = urlsplit(raw)
    try:
        parsed_port = parsed.port
    except ValueError:
        parser.error(f"{option} has an invalid port")
    if (
        raw != raw.strip()
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (parsed_port is not None and not 1 <= parsed_port <= 65535)
    ):
        parser.error(f"{option} must be an HTTPS URL without credentials, query, or fragment")
    return raw.rstrip("/")


def _regular_file(parser: argparse.ArgumentParser, option: str, raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        parser.error(f"{option} must name an existing regular file")
    return path


def _positive(parser: argparse.ArgumentParser, option: str, value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        parser.error(f"{option} must be positive")
    return value


_RUNTIME_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _initial_labels(
    parser: argparse.ArgumentParser,
    cli_values: list[str] | None,
    environ: Mapping[str, str],
) -> tuple[str, ...]:
    if cli_values is None:
        raw = environ.get("CONTRACTOR_INITIAL_LABELS", "")
        candidates = [] if raw == "" else raw.split(",")
    else:
        candidates = cli_values
    if len(candidates) > 32:
        parser.error("at most 32 --initial-label values are allowed")
    if len(candidates) != len(set(candidates)):
        parser.error("--initial-label values must be unique")
    for value in candidates:
        if (
            not 1 <= len(value) <= 63
            or _RUNTIME_LABEL_PATTERN.fullmatch(value) is None
            or value == "default"
        ):
            parser.error("--initial-label must match [a-z][a-z0-9_-]* and not be default")
    return tuple(sorted(candidates))


_KNOWN_RUNTIME_ADAPTERS = frozenset({"caido-graphql@1", "http-proxy@1", "otlp-http@1"})


def _runtime_adapters(
    parser: argparse.ArgumentParser,
    cli_values: list[str] | None,
    environ: Mapping[str, str],
) -> tuple[str, ...] | None:
    """Return the immutable enabled factory set, or None for every built-in."""

    if cli_values is None:
        raw = environ.get("CONTRACTOR_RUNTIME_ADAPTERS")
        if raw is None:
            return None
        candidates = raw.split(",")
    else:
        candidates = cli_values
    if not candidates or any(not value for value in candidates):
        parser.error("--runtime-adapter requires a non-empty adapter ref")
    if len(candidates) != len(set(candidates)):
        parser.error("--runtime-adapter values must be unique")
    unknown = sorted(set(candidates) - _KNOWN_RUNTIME_ADAPTERS)
    if unknown:
        parser.error("--runtime-adapter must name a supported built-in adapter")
    return tuple(sorted(candidates))


def _workspace_settings(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> WorkspaceSettings | None:
    related = (
        args.workspace_operation_timeout_seconds,
        args.workspace_work_root,
        args.workspace_max_files,
        args.workspace_max_expanded_bytes,
        args.workspace_max_managed_text_bytes,
        args.workspace_max_file_bytes,
    )
    if args.workspace_storage is None:
        if any(value is not None for value in related):
            parser.error("--workspace-storage is required when workspace options are set")
        return None

    storage: WorkspaceStorageV2 = args.workspace_storage
    work_root: Path | None = None
    if storage == "local":
        raw_root = args.workspace_work_root
        if not isinstance(raw_root, str) or not raw_root:
            parser.error("--workspace-work-root is required for local workspace storage")
        candidate = Path(raw_root).expanduser()
        if not candidate.is_absolute():
            parser.error("--workspace-work-root must be absolute")
        if candidate.is_symlink():
            parser.error("--workspace-work-root must not be a symlink")
        work_root = candidate.resolve(strict=False)
        if work_root == Path(work_root.anchor):
            parser.error("--workspace-work-root must not be a filesystem root")
    elif args.workspace_work_root is not None:
        parser.error("--workspace-work-root is valid only for local workspace storage")

    operation_timeout = args.workspace_operation_timeout_seconds
    if operation_timeout is not None and storage != "local":
        parser.error(
            "--workspace-operation-timeout-seconds is valid only for local workspace storage"
        )
    if operation_timeout is None:
        operation_timeout = DEFAULT_WORKSPACE_OPERATION_TIMEOUT_SECONDS
    if not math.isfinite(operation_timeout) or operation_timeout <= 0:
        parser.error("--workspace-operation-timeout-seconds must be finite and positive")

    defaults = LOCAL_WORKSPACE_DEFAULTS if storage == "local" else MEMORY_WORKSPACE_DEFAULTS
    values = (
        args.workspace_max_files,
        args.workspace_max_expanded_bytes,
        args.workspace_max_managed_text_bytes,
        args.workspace_max_file_bytes,
    )
    selected = tuple(
        default if value is None else value for default, value in zip(defaults, values, strict=True)
    )
    names = (
        "--workspace-max-files",
        "--workspace-max-expanded-bytes",
        "--workspace-max-managed-text-bytes",
        "--workspace-max-file-bytes",
    )
    for name, value, ceiling in zip(names, selected, WORKSPACE_LIMIT_CEILINGS, strict=True):
        if not isinstance(value, int) or value <= 0 or value > ceiling:
            parser.error(f"{name} must be positive and within the implementation ceiling")
    max_files, max_expanded, max_managed, max_file = selected
    if max_managed > max_expanded or max_file > max_expanded:
        parser.error("workspace managed/file byte limits must not exceed expanded bytes")
    return WorkspaceSettings(
        storage=storage,
        operation_timeout_seconds=operation_timeout,
        work_root=work_root,
        limits=WorkspaceLimits(
            max_files=max_files,
            max_expanded_bytes=max_expanded,
            max_managed_text_bytes=max_managed,
            max_file_bytes=max_file,
        ),
    )
