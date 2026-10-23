#!/usr/bin/env python3
"""Install one provider WireGuard profile in an isolated LiteLLM namespace."""

from __future__ import annotations

import argparse
import base64
import configparser
import ipaddress
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid"))
from common import DEFAULT_STATE, apply_remote, load_state, secure_write, ssh


def parse_profile(path):
    config = configparser.ConfigParser(interpolation=None, strict=True)
    config.read_string(Path(path).read_text())
    if set(config.sections()) != {"Interface", "Peer"}:
        raise ValueError("expected exactly one Interface and one Peer")
    interface, peer = config["Interface"], config["Peer"]
    if set(interface) - {"privatekey", "address", "dns", "mtu"}:
        raise ValueError("interface hooks and additional directives are not accepted")
    if set(peer) - {
        "publickey",
        "presharedkey",
        "allowedips",
        "endpoint",
        "persistentkeepalive",
    }:
        raise ValueError("unexpected provider Peer directive")
    for value in [
        interface["privatekey"],
        peer["publickey"],
        peer.get("presharedkey", ""),
    ]:
        if value and len(base64.b64decode(value, validate=True)) != 32:
            raise ValueError("invalid WireGuard key")
    address = str(ipaddress.ip_interface(interface["address"]))
    if ipaddress.ip_interface(address).version != 4:
        raise ValueError("this launcher currently requires one IPv4 tunnel address")
    allowed = [str(ipaddress.ip_network(x.strip())) for x in peer["allowedips"].split(",")]
    if "0.0.0.0/0" not in allowed:
        raise ValueError("provider profile must carry the IPv4 default route")
    dns = [str(ipaddress.ip_address(x.strip())) for x in interface.get("dns", "1.1.1.1").split(",")]
    mtu = int(interface.get("mtu", "1380"))
    if not 1280 <= mtu <= 1420:
        raise ValueError("MTU must be between 1280 and 1420")
    endpoint = peer["endpoint"]
    if not re.fullmatch(r"[A-Za-z0-9_.:\[\]-]+:[0-9]{1,5}", endpoint):
        raise ValueError("invalid provider endpoint")
    if not 1 <= int(endpoint.rsplit(":", 1)[1]) <= 65535:
        raise ValueError("invalid provider port")
    keepalive = int(peer.get("persistentkeepalive", "25"))
    if not 0 <= keepalive <= 65535:
        raise ValueError("invalid keepalive")
    lines = [
        "[Interface]",
        "PrivateKey = " + interface["privatekey"],
        "",
        "[Peer]",
        "PublicKey = " + peer["publickey"],
        "AllowedIPs = " + ", ".join(allowed),
        "Endpoint = " + endpoint,
        "PersistentKeepalive = " + str(keepalive),
    ]
    if peer.get("presharedkey"):
        lines.append("PresharedKey = " + peer["presharedkey"])
    return "\n".join(lines) + "\n", {"address": address, "dns": dns, "mtu": mtu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("apply")
    p.add_argument("--config", type=Path, required=True)
    commands.add_parser("check")
    commands.add_parser("check-kill-switch")
    args = parser.parse_args()
    state, doc = load_state(args.state)
    host = next(x for x in doc["hosts"] if x["ssh"] == doc["gateway_host"])
    if args.command != "apply":
        command = "/opt/contractor-hybrid/bin/egress.sh " + args.command
        print(ssh(host["ssh"], "sudo -n " + command, timeout=90).rstrip())
        return
    config, metadata = parse_profile(args.config)
    metadata.update(mesh_address=host["address"], mesh_network=doc["network"])
    secure_write(state / "secrets" / "wg-egress.conf", config)
    secure_write(state / "egress.json", json.dumps(metadata, indent=2) + "\n")
    unit = """[Unit]
Description=Contractor LiteLLM WireGuard egress namespace
Requires=wg-quick@wg-mesh.service
After=network-online.target wg-quick@wg-mesh.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/contractor-hybrid/bin/egress.sh up
ExecStop=/opt/contractor-hybrid/bin/egress.sh down

[Install]
WantedBy=multi-user.target
"""
    output = apply_remote(
        host["ssh"],
        {
            "/etc/contractor-hybrid/wg-egress.conf": (config, 0o600),
            "/etc/contractor-hybrid/egress.json": (json.dumps(metadata), 0o600),
            "/opt/contractor-hybrid/bin/egress.sh": (
                (Path(__file__).parent / "linux-egress.sh").read_text(),
                0o700,
            ),
            "/etc/systemd/system/contractor-wg-egress.service": (unit, 0o644),
        },
        ["/opt/contractor-hybrid/bin/egress.sh", "install"],
        timeout=600,
    )
    print(f"{host['ssh']}: provider profile installed in contractor-llm only")
    print(output.rstrip())


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        configparser.Error,
        subprocess.SubprocessError,
    ):
        # Configparser and decoder exceptions can include secret input lines.
        print(
            "egress: configuration or remote operation failed; no key material was logged",
            file=sys.stderr,
        )
        sys.exit(1)
