#!/usr/bin/env python3
"""Deploy cloud Runtime Agents with the existing Ansible role and local PKI."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shlex
import ssl
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_STATE, ROOT, load_state, run, secure_write, ssh

ANSIBLE = ["uvx", "--from", "ansible-core==2.21.5", "ansible-playbook"]


def inventory(args, state, doc):
    binary = args.contractor.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("provide --contractor with the local release's executable")
    pki = args.pki_root.resolve()
    if not (pki / "ca.crt").is_file() or not (pki / "ca.key").is_file():
        raise ValueError("the existing deployment CA must be on the local PKI host")
    version = run(["git", "-C", ROOT, "rev-parse", args.version + "^{commit}"]).strip()
    run(
        [
            "git",
            "-C",
            ROOT,
            "cat-file",
            "-e",
            version + ":runtime/scripts/prefetch_parsers.py",
        ]
    )
    capabilities = (
        {}
        if args.full_capabilities
        else {
            "workspace_storage": "local",
            "openapi_validator": False,
            "likec4_validator": False,
            "scanners": [],
            "podman": False,
            "podman_network": False,
        }
    )
    hosts = {}
    for host in doc["hosts"]:
        instances = [
            {"name": host["ssh"] + "-" + str(i + 1), "port": 9443 + i} for i in range(host["slots"])
        ]
        hosts[host["ssh"]] = {
            "ansible_host": host["ssh"],
            "ansible_python_interpreter": "/usr/bin/python3",
            "contractor_runtime_listen_host": host["address"],
            "contractor_runtime_advertise_host": host["address"],
            "contractor_runtime_instances": instances,
        }
    document = {
        "contractor_runtime": {
            "vars": {
                "ansible_connection": "ssh",
                "contractor_runtime_source_repo": str(ROOT),
                "contractor_runtime_version": version,
                "contractor_runtime_control_plane_url": "https://"
                + doc["controller_address"]
                + ":8443",
                "contractor_runtime_pki_host": "localhost",
                "contractor_runtime_pki_root": str(pki),
                "contractor_runtime_pki_command": str(binary),
                "contractor_runtime_capabilities": capabilities,
                "contractor_runtime_restart": not args.defer_restarts,
                "contractor_runtime_verify": True,
                "contractor_runtime_labels": args.labels,
            },
            "hosts": hosts,
        }
    }
    path = state / "runtime-inventory.yml"
    secure_write(path, json.dumps(document, indent=2) + "\n")
    return path


def preflight(args, doc):
    ids = []
    for host in doc["hosts"]:
        machine_id = ssh(host["ssh"], "cat /etc/machine-id").strip()
        if machine_id in ids:
            raise RuntimeError("two SSH aliases connect to the same machine")
        ids.append(machine_id)
        target = shlex.quote(doc["controller_address"])
        ssh(
            host["ssh"],
            (
                "python3 -c "
                + shlex.quote(
                    "import socket; "
                    f"socket.create_connection(({doc['controller_address']!r},8443),8).close()"
                )
            ),
        )
        print(f"{host['ssh']}: local Control Plane TCP endpoint reachable at {target}:8443")


def verify(args, doc):
    """Check every listener, including unchanged units that Ansible skips."""
    pki = args.pki_root.resolve()
    context = ssl.create_default_context(cafile=str(pki / "ca.crt"))
    context.load_cert_chain(pki / "control-plane.crt", pki / "control-plane.key")
    for host in doc["hosts"]:
        for number in range(host["slots"]):
            connection = http.client.HTTPSConnection(
                host["address"], 9443 + number, context=context, timeout=10
            )
            try:
                connection.request("GET", "/readyz")
                response = connection.getresponse()
                body = response.read(4097)
                if response.status != 200 or len(body) > 4096:
                    raise RuntimeError("a cloud Runtime is not ready after deployment")
                if json.loads(body).get("status") != "ready":
                    raise RuntimeError("a cloud Runtime returned unexpected readiness")
            finally:
                connection.close()
            print(f"{host['ssh']}-{number + 1}: registered, mTLS readiness passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--contractor", type=Path, required=True)
    parser.add_argument("--pki-root", type=Path, required=True)
    parser.add_argument(
        "--version",
        required=True,
        help="Exact Runtime source revision supported by this Ansible role",
    )
    parser.add_argument("--full-capabilities", action="store_true")
    parser.add_argument("--defer-restarts", action="store_true")
    parser.add_argument("--labels", nargs="+", default=["cloud-llm"])
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--syntax-check", action="store_true")
    args = parser.parse_args()
    state, doc = load_state(args.state)
    path = inventory(args, state, doc)
    if args.inventory_only:
        print(f"Inventory written: {path}")
        return
    if not args.syntax_check:
        preflight(args, doc)
    env = dict(os.environ, ANSIBLE_CONFIG=str(ROOT / "deploy/ansible/ansible.cfg"))
    argv = [*ANSIBLE, "-i", str(path), str(ROOT / "deploy/ansible/runtime.yml")]
    if args.syntax_check:
        argv.append("--syntax-check")
    result = subprocess.run(argv, cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"Ansible failed (exit {result.returncode}); deployment is incomplete")
    if not args.syntax_check:
        verify(args, doc)


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"runtime deploy: {exc}", file=sys.stderr)
        sys.exit(1)
