#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["PyYAML==6.0.3"]
# ///
"""Prepare a local Server profile; preserve the existing loopback API and CA."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml
from bootstrap import API
from common import DEFAULT_STATE, ROOT, load_state, run, secret, secure_write

PATH_FIELDS = (
    "operatorConfigRoot",
    "managedConfigRoot",
    "caFile",
    "certificateFile",
    "privateKeyFile",
    "artifactBlobPath",
    "gitKnownHostsFile",
    "localAuthFile",
    "credentialMasterKeyFile",
    "llmGatewayAdminBindingsFile",
)


def prepare(args, state, doc):
    source = args.config.resolve()
    config = yaml.safe_load(source.read_text())
    spec = config["spec"]
    if spec["privateListen"] not in {"127.0.0.1:8443", "localhost:8443"}:
        raise ValueError("this profile requires the existing private listener on 127.0.0.1:8443")
    for name in PATH_FIELDS:
        if spec.get(name):
            spec[name] = str((source.parent / spec[name]).resolve())
    ca = (args.pki_root / "ca.crt").read_bytes()
    if Path(spec["caFile"]).read_bytes() != ca:
        raise ValueError("PKI root does not match the existing Server CA")
    pki = state / "server-pki"
    leaf = pki / "control-plane.crt"
    if leaf.exists():
        if (pki / "ca.crt").read_bytes() != ca:
            raise ValueError("prepared Server leaf belongs to another CA")
        run(
            [
                "openssl",
                "verify",
                "-CAfile",
                pki / "ca.crt",
                "-verify_ip",
                doc["controller_address"],
                leaf,
            ]
        )
    if not leaf.exists():
        # Issue a separate CP leaf under the existing CA. Do not replace the
        # original CP identity, Runtime keys, business data or migration state.
        secure_write(pki / "ca.crt", (args.pki_root / "ca.crt").read_text(), replace=False)
        secure_write(pki / "ca.key", secret(args.pki_root / "ca.key") + "\n")
        try:
            run(
                [
                    args.contractor,
                    "pki",
                    "issue-control-plane",
                    "--root",
                    pki,
                    "--dns",
                    "localhost",
                    "--ip",
                    "127.0.0.1,::1," + doc["controller_address"],
                ]
            )
        finally:
            (pki / "ca.key").unlink(missing_ok=True)
    spec["certificateFile"] = str(leaf)
    spec["privateKeyFile"] = str(pki / "control-plane.key")
    spec["privateUrl"] = "https://" + doc["controller_address"] + ":8443"
    # Both endpoints terminate at the same original Server. The mesh ingress
    # is a raw TCP relay: Runtime mTLS identity is preserved end-to-end.
    spec["privateListen"] = "127.0.0.1:8443"
    binding = json.loads((state / "gateway-admin-bindings.json").read_text())
    old_path = spec.get("llmGatewayAdminBindingsFile")
    if old_path:
        old = yaml.safe_load(Path(old_path).read_text())
        exact = {tuple(sorted(x["llmGateway"].items())) for x in binding["bindings"]}
        binding["bindings"] += [
            x for x in old["bindings"] if tuple(sorted(x["llmGateway"].items())) not in exact
        ]
    secure_write(state / "server-admin-bindings.json", json.dumps(binding, indent=2) + "\n")
    spec["llmGatewayAdminBindingsFile"] = str(state / "server-admin-bindings.json")
    secure_write(state / "server.yaml", yaml.safe_dump(config, sort_keys=False))
    secure_write(
        state / "server-previous.json",
        json.dumps(
            {
                "config": str(source),
                "binary": str(args.contractor.resolve()),
            },
            indent=2,
        )
        + "\n",
    )
    print(
        f"Server profile prepared: {state / 'server.yaml'}; existing CA/leaf and database unchanged"
    )


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def launch(argv, log_path, *, env=None):
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(x) for x in argv],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    return process


def proxy(state, doc):
    pidfile = state / "ingress.pid"
    if pidfile.exists() and alive(int(pidfile.read_text())):
        return
    process = launch(
        [
            sys.executable,
            Path(__file__).parent / "ingress.py",
            "--listen",
            doc["controller_address"],
        ],
        state / "ingress.log",
    )
    time.sleep(0.3)
    if process.poll() is not None:
        raise RuntimeError("mesh ingress failed; inspect its protected log")
    secure_write(pidfile, str(process.pid) + "\n")


def restart(args, state, doc):
    # Do not stop the live Server before the operator has brought up local WG.
    with socket.socket() as probe:
        probe.bind((doc["controller_address"], 0))
    api = API(args.server_url, args.token_file)
    queue = list(api.pages("/v1/queue"))
    agents = list(api.pages("/v1/operations/runtime-agents"))
    if queue or any(agent["slotState"] != "idle" for agent in agents):
        raise RuntimeError("Server has queued or active work; drain it before restarting")
    pid = int(args.pid_file.read_text())
    command = run(["ps", "-p", str(pid), "-o", "command="]).strip()
    if str(args.contractor.resolve()) not in command or "server run" not in command:
        raise RuntimeError("pid file does not identify the expected Server process")
    document = json.loads(secret(args.environment))
    env = {key: value for key, value in os.environ.items() if not key.startswith("CONTRACTOR_")}
    env.update(document.get("values", {}))
    for name, path in document.get("files", {}).items():
        env[name] = secret(path)
    os.kill(pid, signal.SIGTERM)
    for _unused in range(50):
        if not alive(pid):
            break
        time.sleep(1)
    else:
        raise RuntimeError("old Server did not stop; no replacement was started")
    process = launch(
        [args.contractor.resolve(), "server", "run", "--config", state / "server.yaml"],
        state / "server.log",
        env=env,
    )
    secure_write(args.pid_file, str(process.pid) + "\n")
    for _unused in range(40):
        if process.poll() is not None:
            break
        try:
            api.request("GET", "/readyz")
            proxy(state, doc)
            print("Local Server ready; loopback clients preserved; mesh-only mTLS ingress active")
            return
        except (OSError, RuntimeError):
            time.sleep(1)
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=45)
    previous = json.loads((state / "server-previous.json").read_text())
    rollback = launch(
        [previous["binary"], "server", "run", "--config", previous["config"]],
        state / "server-rollback.log",
        env=env,
    )
    secure_write(args.pid_file, str(rollback.pid) + "\n")
    raise RuntimeError("new Server was not ready; original Server configuration was restarted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pki-root", type=Path, required=True)
    p.add_argument("--contractor", type=Path, required=True)
    p = commands.add_parser("restart")
    p.add_argument("--contractor", type=Path, required=True)
    p.add_argument("--pid-file", type=Path, required=True)
    p.add_argument("--environment", type=Path, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    commands.add_parser("proxy")
    args = parser.parse_args()
    state, doc = load_state(args.state)
    if args.command == "prepare":
        prepare(args, state, doc)
    elif args.command == "restart":
        restart(args, state, doc)
    else:
        proxy(state, doc)


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"server profile: {exc}", file=sys.stderr)
        sys.exit(1)
