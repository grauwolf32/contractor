#!/usr/bin/env python3
"""Persistent cloud LiteLLM behind the separate WireGuard egress namespace."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

from common import (
    DEFAULT_STATE,
    ROOT,
    apply_remote,
    load_state,
    secret,
    secure_write,
    ssh,
)


def pinned_images():
    source = (ROOT / "deploy/litellm/run.sh").read_text()
    result = {}
    for name in ("litellm", "postgres"):
        match = re.search(r"^" + name + r"_image=\$\{[^:]+:-([^}]+)\}", source, re.MULTILINE)
        if not match or not re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", match[1]):
            raise RuntimeError("could not read the existing digest-pinned image")
        result[name] = match[1]
    return result


def environment(values):
    for value in values.values():
        if "\n" in value or "\r" in value or "\x00" in value:
            raise ValueError("a secret contains an unsupported line break")
    return "".join(key + "=" + value + "\n" for key, value in values.items())


def unit(name, command, requires, after):
    return (
        "[Unit]\nDescription=Contractor hybrid " + name + "\n"
        "Requires=" + requires + "\nAfter=" + after + "\n\n"
        "[Service]\nType=exec\nKillMode=mixed\nUMask=0077\nExecStart=" + command + "\n"
        "ExecStop=/usr/bin/podman stop -t 30 contractor-hybrid-" + name + "\n"
        "Restart=on-failure\nRestartSec=5\nTimeoutStartSec=300\nTimeoutStopSec=45\n\n"
        "[Install]\nWantedBy=multi-user.target\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("apply")
    p.add_argument("--openrouter-key-file", type=Path, required=True)
    commands.add_parser("check")
    args = parser.parse_args()
    state, doc = load_state(args.state)
    host = doc["gateway_host"]
    if args.command == "check":
        print(ssh(host, "sudo -n /opt/contractor-hybrid/bin/gateway.sh check", timeout=90).rstrip())
        return
    if not (state / "egress.json").exists():
        raise RuntimeError("apply deploy/wireguard/egress.py first")
    for name, prefix in (
        ("litellm-master-key", "sk-"),
        ("litellm-salt-key", "sk-"),
        ("litellm-postgres-password", ""),
    ):
        secure_write(
            state / "secrets" / name,
            prefix + secrets.token_urlsafe(32) + "\n",
            replace=False,
        )
    upstream = secret(args.openrouter_key_file)
    if not upstream.startswith("sk-or-v1-"):
        raise ValueError("provide an OpenRouter inference key")
    master = secret(state / "secrets/litellm-master-key")
    salt = secret(state / "secrets/litellm-salt-key")
    password = secret(state / "secrets/litellm-postgres-password")
    images = pinned_images()
    proxy_env = environment(
        {
            "LITELLM_MASTER_KEY": master,
            "LITELLM_SALT_KEY": salt,
            "OPENROUTER_API_KEY": upstream,
            "DATABASE_URL": "postgresql://litellm:" + password + "@172.30.91.1:54329/litellm",
        }
    )
    db_env = environment(
        {
            "POSTGRES_USER": "litellm",
            "POSTGRES_DB": "litellm",
            "POSTGRES_PASSWORD": password,
        }
    )
    public = {
        "images": images,
        "dns": json.loads((state / "egress.json").read_text())["dns"],
    }
    files = {
        "/etc/contractor-hybrid/litellm.env": (proxy_env, 0o600),
        "/etc/contractor-hybrid/postgres.env": (db_env, 0o600),
        "/etc/contractor-hybrid/litellm_config.yaml": (
            (ROOT / "deploy/litellm/litellm_config.yaml").read_text(),
            0o644,
        ),
        "/etc/contractor-hybrid/gateway.json": (json.dumps(public), 0o644),
        "/opt/contractor-hybrid/bin/gateway.sh": (
            (Path(__file__).parent / "gateway.sh").read_text(),
            0o700,
        ),
        "/etc/systemd/system/contractor-hybrid-postgres.service": (
            unit(
                "postgres",
                "/opt/contractor-hybrid/bin/gateway.sh postgres",
                "contractor-wg-egress.service",
                "contractor-wg-egress.service",
            ),
            0o644,
        ),
        "/etc/systemd/system/contractor-hybrid-litellm.service": (
            unit(
                "litellm",
                "/opt/contractor-hybrid/bin/gateway.sh litellm",
                "contractor-wg-egress.service contractor-hybrid-postgres.service",
                "contractor-wg-egress.service contractor-hybrid-postgres.service",
            ),
            0o644,
        ),
    }
    print(
        apply_remote(
            host,
            files,
            ["/opt/contractor-hybrid/bin/gateway.sh", "install"],
            timeout=1200,
        ).rstrip()
    )
    print(f"{host}: persistent LiteLLM installed; provider keys were not printed")


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"gateway: {exc}", file=sys.stderr)
        sys.exit(1)
