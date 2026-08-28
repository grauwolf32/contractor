"""Bootstrap process settings for the Runtime Agent."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    log_level: str

    @property
    def listen_address(self) -> str:
        return f"{self.host}:{self.port}"


def parse_settings(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    values = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(prog="contractor-runtime")
    parser.add_argument(
        "--listen",
        default=values.get("CONTRACTOR_RUNTIME_LISTEN", "127.0.0.1:9080"),
    )
    parser.add_argument(
        "--log-level",
        default=values.get("CONTRACTOR_RUNTIME_LOG_LEVEL", "info"),
        choices=("debug", "info", "warning", "error"),
    )
    args = parser.parse_args(argv)

    try:
        host, port_text = args.listen.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, AttributeError) as exc:
        parser.error(f"invalid --listen value: {exc}")
    if not host:
        parser.error("--listen host must not be empty")
    if not 0 <= port <= 65535:
        parser.error("--listen port must be between 0 and 65535")

    return Settings(host=host, port=port, log_level=args.log_level)
