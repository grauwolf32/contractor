#!/usr/bin/env python3
"""Encrypted backup/restore of generated deployment secrets using Ansible Vault."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_STATE, secure_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--vault-id", default="contractor@prompt")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("seal")
    p.add_argument("output", type=Path)
    p = commands.add_parser("restore")
    p.add_argument("source", type=Path)
    args = parser.parse_args()
    state = args.state.resolve()
    command = ["uvx", "--from", "ansible-core==2.21.5", "ansible-vault"]
    if args.command == "seal":
        if args.output.exists():
            raise RuntimeError("backup exists; choose a new output filename")
        values = {}
        for directory in ("secrets", "wireguard"):
            for path in sorted((state / directory).rglob("*")):
                if path.is_symlink():
                    raise RuntimeError("refusing a symlink in the secret state")
                if path.is_file():
                    if path.stat().st_mode & 0o077:
                        raise RuntimeError("secret state must be owner-only")
                    values[str(path.relative_to(state))] = base64.b64encode(
                        path.read_bytes()
                    ).decode()
        if not values:
            raise RuntimeError("nothing to back up")
        subprocess.run(
            [*command, "encrypt", "--vault-id", args.vault_id, "--output", str(args.output), "-"],
            input=json.dumps({"version": 1, "files": values}),
            text=True,
            check=True,
        )
        os.chmod(args.output, 0o600)
        print(f"Encrypted backup created: {args.output}")
    else:
        # Capture decrypted content, never emit it to the terminal.
        result = subprocess.run(
            [*command, "view", "--vault-id", args.vault_id, str(args.source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("could not unlock the backup")
        document = json.loads(result.stdout)
        if document.get("version") != 1:
            raise ValueError("unsupported backup version")
        restored = {}
        for name, encoded in document["files"].items():
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) < 2
                or relative.parts[0] not in {"secrets", "wireguard"}
            ):
                raise ValueError("backup path is outside secret state")
            target = state / relative
            if target.exists() or any(parent.is_symlink() for parent in [target, *target.parents]):
                raise RuntimeError("refusing to overwrite existing secret state")
            restored[target] = base64.b64decode(encoded, validate=True).decode()
        for target, content in restored.items():
            secure_write(target, content, replace=False)
        print("Secret state restored; no values were printed")


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError):
        print("secret operation failed; no decrypted values were logged", file=sys.stderr)
        sys.exit(1)
