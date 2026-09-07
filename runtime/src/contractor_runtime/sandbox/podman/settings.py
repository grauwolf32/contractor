"""Immutable operator policy. Parsing never invokes Podman or provisions images."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

PODMAN_LIMIT_CEILINGS = MappingProxyType(
    {
        "cpus": 64,
        "memory_bytes": 64 << 30,
        "pids": 4096,
        "tmpfs_bytes": 4 << 30,
        "command_max_seconds": 3600,
        "prepare_max_seconds": 120,
        "stop_grace_seconds": 30,
        "preview_bytes": 1 << 20,
        "output_max_bytes": 16 << 20,
    }
)
_IMAGE = re.compile(r"[a-z0-9]+(?:[._:/-][a-z0-9]+)*@sha256:[0-9a-f]{64}")
_OWNER = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")


@dataclass(frozen=True, slots=True)
class PodmanSettings:
    enabled: bool = False
    image: str | None = field(default=None, repr=False)
    owner: str | None = field(default=None, repr=False)
    cpus: float = 2.0
    memory_bytes: int = 2 << 30
    pids: int = 256
    tmpfs_bytes: int = 256 << 20
    command_max_seconds: int = 300
    prepare_max_seconds: int = 30
    stop_grace_seconds: int = 5
    preview_bytes: int = 32 << 10
    output_max_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("podman enabled must be boolean")
        if self.image is not None and (
            not isinstance(self.image, str)
            or len(self.image) > 512
            or _IMAGE.fullmatch(self.image) is None
        ):
            raise ValueError("podman image must be a bounded repository@sha256 digest reference")
        if self.owner is not None and (
            not isinstance(self.owner, str) or _OWNER.fullmatch(self.owner) is None
        ):
            raise ValueError("podman owner must be a stable lowercase service identifier")
        if self.enabled and (self.image is None or self.owner is None):
            raise ValueError("enabled podman requires image and owner")
        for name, ceiling in PODMAN_LIMIT_CEILINGS.items():
            value = getattr(self, name)
            valid_type = type(value) in (int, float) if name == "cpus" else type(value) is int
            if not valid_type or not 0 < value <= ceiling or not math.isfinite(value):
                raise ValueError(f"podman {name} must be positive and at most {ceiling}")
        if self.command_max_seconds < 60:
            raise ValueError("podman command maximum must accommodate the 60 second default")
        if 2 * self.preview_bytes > self.output_max_bytes:
            raise ValueError("podman combined output maximum must cover both previews")
        if self.tmpfs_bytes > self.memory_bytes:
            raise ValueError("podman tmpfs maximum must not exceed memory maximum")


def add_podman_arguments(parser: argparse.ArgumentParser, values: Mapping[str, str]) -> None:
    defaults = PodmanSettings()
    parser.add_argument(
        "--podman-enabled",
        type=_boolean,
        default=values.get("CONTRACTOR_PODMAN_ENABLED", "false"),
    )
    for name in ("image", "owner", *PODMAN_LIMIT_CEILINGS):
        parser.add_argument(
            "--podman-" + name.replace("_", "-"),
            type=float if name == "cpus" else (str if name in {"image", "owner"} else int),
            default=values.get("CONTRACTOR_PODMAN_" + name.upper(), getattr(defaults, name)),
        )


def podman_settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> PodmanSettings:
    try:
        return PodmanSettings(
            **{
                name: getattr(args, "podman_" + name)
                for name in ("enabled", "image", "owner", *PODMAN_LIMIT_CEILINGS)
            }
        )
    except ValueError as error:
        parser.error(str(error))


def _boolean(raw: str) -> bool:
    if raw not in {"true", "false"}:
        raise argparse.ArgumentTypeError("must be true or false")
    return raw == "true"
