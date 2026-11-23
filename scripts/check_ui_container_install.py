#!/usr/bin/env python3
"""Validate the UI image's frozen-install layer using its actual COPY inputs."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def instructions(containerfile: Path) -> list[str]:
    result: list[str] = []
    pending = ""
    for raw in containerfile.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        result.append(pending)
        pending = ""
    if pending:
        raise ValueError("Containerfile ends with an unfinished instruction")
    return result


def copied_install_inputs(containerfile: Path) -> list[Path]:
    sources: list[Path] = []
    for instruction in instructions(containerfile):
        operation, _, arguments = instruction.partition(" ")
        if operation.upper() == "RUN" and "pnpm install --frozen-lockfile" in arguments:
            if not sources:
                raise ValueError("no inputs were copied before the frozen install")
            return sources
        if operation.upper() != "COPY":
            continue
        parts = shlex.split(arguments)
        if len(parts) < 2 or parts[-1] != "./" or any(part.startswith("--") for part in parts):
            raise ValueError(f"unsupported pre-install COPY: {instruction}")
        for source in parts[:-1]:
            source_path = Path(source)
            if source_path.parts[:1] != ("ui",) or not (ROOT / source_path).is_file():
                raise ValueError(f"pre-install COPY source is not a UI file: {source}")
            sources.append(source_path)
    raise ValueError("Containerfile has no frozen install")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "containerfile",
        nargs="?",
        type=Path,
        default=ROOT / "deploy/ui/Containerfile",
    )
    args = parser.parse_args()
    try:
        sources = copied_install_inputs(args.containerfile)
        with tempfile.TemporaryDirectory(prefix="contractor-ui-install-") as temp:
            workdir = Path(temp)
            for source in sources:
                shutil.copy2(ROOT / source, workdir / source.name)
            subprocess.run(
                [
                    "corepack",
                    "pnpm",
                    "install",
                    "--frozen-lockfile",
                    "--lockfile-only",
                    "--offline",
                    "--ignore-scripts",
                ],
                cwd=workdir,
                check=True,
            )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"UI image install layer is invalid: {error}", file=sys.stderr)
        return 1
    print("UI image frozen-install inputs are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
