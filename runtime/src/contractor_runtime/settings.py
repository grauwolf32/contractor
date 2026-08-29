"""Strict process settings for the single-slot Runtime Agent."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class Settings:
    control_plane_url: str
    advertised_control_url: str
    advertised_a2a_url: str
    ca_file: Path
    certificate_file: Path
    private_key_file: Path = field(repr=False)
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
    parser.add_argument("--control-plane-url", default=values.get("CONTRACTOR_CONTROL_PLANE_URL"))
    parser.add_argument(
        "--advertised-control-url", default=values.get("CONTRACTOR_ADVERTISED_CONTROL_URL")
    )
    parser.add_argument("--advertised-a2a-url", default=values.get("CONTRACTOR_ADVERTISED_A2A_URL"))
    parser.add_argument("--ca-file", default=values.get("CONTRACTOR_CA_FILE"))
    parser.add_argument("--certificate-file", default=values.get("CONTRACTOR_CERTIFICATE_FILE"))
    parser.add_argument("--private-key-file", default=values.get("CONTRACTOR_PRIVATE_KEY_FILE"))
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

    return Settings(
        control_plane_url=control_plane_url,
        advertised_control_url=advertised_control_url,
        advertised_a2a_url=advertised_a2a_url,
        ca_file=ca_file,
        certificate_file=certificate_file,
        private_key_file=private_key_file,
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
