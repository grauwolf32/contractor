#!/usr/bin/env python3
"""Independent WireGuard mesh setup. Does not change the host default route."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid"))
from common import (
    DEFAULT_STATE,
    apply_remote,
    load_state,
    run,
    secure_write,
    ssh,
    ssh_hostname,
)


def key_pair(directory):
    private = directory / "private.key"
    if not private.exists():
        secure_write(private, run(["wg", "genkey"]), replace=False)
    if private.is_symlink() or private.stat().st_mode & 0o077:
        raise RuntimeError("WireGuard private key must be a regular 0600 file")
    return private.read_text().strip(), run(["wg", "pubkey"], data=private.read_text()).strip()


def init(args):
    state = args.state.resolve()
    if (state / "deployment.json").exists():
        raise RuntimeError("state already exists; use render/apply without regenerating identities")
    if len(set(args.hosts)) != len(args.hosts) or not 1 <= len(args.hosts) <= 16:
        raise ValueError("provide 1..16 distinct SSH aliases")
    network = ipaddress.ip_network(args.network)
    if network.version != 4 or network.prefixlen != 24 or not network.is_private:
        raise ValueError("mesh network must be a private IPv4 /24")
    if not 1024 <= args.port <= 65535:
        raise ValueError("invalid mesh UDP port")
    hosts = [
        {"ssh": alias, "address": str(network.network_address + 11 + index), "slots": 1}
        for index, alias in enumerate(args.hosts)
    ]
    endpoints = [ssh_hostname(host["ssh"]) for host in hosts]
    if len(set(endpoints)) != len(hosts):
        raise ValueError("SSH aliases resolve to one machine")
    doc = {
        "version": 1,
        "network": str(network),
        "port": args.port,
        "controller_address": str(network.network_address + 10),
        "hosts": hosts,
        "gateway_host": hosts[0]["ssh"],
    }
    secure_write(state / "deployment.json", json.dumps(doc, indent=2) + "\n", replace=False)
    render(state, doc)
    print(f"Mesh state prepared: {state}; private keys were not printed")


def render(state, doc):
    nodes = [
        {"ssh": "controller", "address": doc["controller_address"], "endpoint": None},
        *[{**host, "endpoint": ssh_hostname(host["ssh"])} for host in doc["hosts"]],
    ]
    keys = {node["ssh"]: key_pair(state / "wireguard" / node["ssh"]) for node in nodes}
    for node in nodes:
        lines = [
            "# Managed by Contractor WireGuard mesh",
            "[Interface]",
            f"PrivateKey = {keys[node['ssh']][0]}",
            f"Address = {node['address']}/32",
            f"ListenPort = {doc['port']}",
            "MTU = 1380",
        ]
        for peer in nodes:
            if peer["ssh"] == node["ssh"]:
                continue
            lines += [
                "",
                "[Peer]",
                f"PublicKey = {keys[peer['ssh']][1]}",
                f"AllowedIPs = {peer['address']}/32",
                "PersistentKeepalive = 25",
            ]
            if peer["endpoint"]:
                endpoint = peer["endpoint"]
                if ":" in endpoint:
                    endpoint = "[" + endpoint + "]"
                lines.append(f"Endpoint = {endpoint}:{doc['port']}")
        secure_write(state / "wireguard" / node["ssh"] / "wg-mesh.conf", "\n".join(lines) + "\n")


def apply(args):
    state, doc = load_state(args.state)
    render(state, doc)
    script = (Path(__file__).parent / "linux-mesh.sh").read_text()
    if args.ubuntu_mirror:
        parsed = urlsplit(args.ubuntu_mirror)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("--ubuntu-mirror must be a plain HTTPS mirror URL")
    selected = args.hosts or [host["ssh"] for host in doc["hosts"]]
    if not set(selected) <= {host["ssh"] for host in doc["hosts"]}:
        raise ValueError("host is absent from deployment.json")
    for alias in selected:
        ssh(
            alias,
            "test ! -e /etc/wireguard/wg-mesh.conf || "
            "sudo -n grep -q '^# Managed by Contractor WireGuard mesh$' "
            "/etc/wireguard/wg-mesh.conf",
        )
        apply_remote(
            alias,
            {
                "/etc/wireguard/wg-mesh.conf": (
                    (state / "wireguard" / alias / "wg-mesh.conf").read_text(),
                    0o600,
                ),
                "/opt/contractor-hybrid/bin/mesh.sh": (script, 0o700),
            },
            [
                "/bin/bash",
                "/opt/contractor-hybrid/bin/mesh.sh",
                "up",
                args.ubuntu_mirror,
            ],
            timeout=600,
        )
        print(f"{alias}: wg-mesh enabled (SSH/default route unchanged)")
    print("Local step (requires administrator privileges):")
    print(
        shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--state",
                str(state),
                "local-up",
            ]
        )
    )


def local_up(args):
    state, _ = load_state(args.state)
    config = state / "wireguard" / "controller" / "wg-mesh.conf"
    executable = shutil.which("wg-quick")
    if not executable:
        raise RuntimeError("install WireGuard tools and wireguard-go locally first")
    cmd = [executable, "up", str(config)]
    if os.geteuid() != 0:
        cmd = ["sudo", "env", "PATH=" + os.environ.get("PATH", ""), *cmd]
    subprocess.run(cmd, check=True)


def status(args):
    _, doc = load_state(args.state)
    for host in doc["hosts"]:
        output = ssh(
            host["ssh"],
            "sudo -n wg show wg-mesh latest-handshakes; ip -brief address show wg-mesh",
        )
        print(host["ssh"] + ":\n" + output.rstrip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("init")
    p.add_argument("hosts", nargs="+")
    p.add_argument("--network", default="10.77.0.0/24")
    p.add_argument("--port", type=int, default=51821)
    p = commands.add_parser("apply")
    p.add_argument("hosts", nargs="*")
    p.add_argument(
        "--ubuntu-mirror",
        default="",
        help="Replace the Cloud.ru Ubuntu mirror; retain Ubuntu signature verification",
    )
    commands.add_parser("render")
    commands.add_parser("local-up")
    commands.add_parser("status")
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "init":
        init(args)
    elif args.command == "render":
        render(*load_state(args.state))
    else:
        {"apply": apply, "local-up": local_up, "status": status}[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"mesh: {exc}", file=sys.stderr)
        sys.exit(1)
