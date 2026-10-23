"""Small deployment primitives; secret inputs and subprocess output stay private."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / ".local" / "hybrid"


def run(argv, *, data=None, timeout=120, show=False, env=None):
    result = subprocess.run(
        [str(x) for x in argv],
        input=data,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{Path(str(argv[0])).name} failed (exit {result.returncode}); "
            "inspect the service locally, without publishing secret-bearing logs"
        )
    if show and result.stdout:
        print(result.stdout.rstrip())
    return result.stdout


def secure_write(path, content, *, replace=True):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink: {path}")
    if not replace and path.exists():
        return
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def secret(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a regular secret file: {path}")
    if path.stat().st_mode & 0o077:
        raise RuntimeError(f"secret file needs mode 0600: {path}")
    return path.read_text().strip()


def ssh_target(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("SSH alias must be a plain host alias")
    return name


def ssh(host, command, *, data=None, timeout=120, show=False):
    return run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            ssh_target(host),
            command,
        ],
        data=data,
        timeout=timeout,
        show=show,
    )


def ssh_hostname(host):
    result = run(["ssh", "-G", ssh_target(host)])
    values = dict(line.split(" ", 1) for line in result.splitlines() if " " in line)
    hostname = values["hostname"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}", hostname):
        raise ValueError("SSH hostname is not suitable for a WireGuard endpoint")
    return hostname


def apply_remote(host, files, command, *, timeout=120):
    """Send files over encrypted stdin, never through argv, environment or logs."""
    bootstrap = (
        "import base64,json,os,pathlib,subprocess,sys,tempfile;"
        "os.umask(0o077);p=json.load(sys.stdin);"
        "\nfor name,item in p['files'].items():\n"
        " path=pathlib.Path(name)\n"
        " if path.is_symlink(): raise RuntimeError('refusing symlink')\n"
        " path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)\n"
        " fd,tmp=tempfile.mkstemp(dir=path.parent)\n"
        " with os.fdopen(fd,'wb') as f: f.write(base64.b64decode(item['data']))\n"
        " os.chmod(tmp,item['mode']);os.replace(tmp,path)\n"
        "\nr=subprocess.run(p['command'],check=False);sys.exit(r.returncode)"
    )
    payload = {
        "files": {
            str(path): {
                "data": base64.b64encode(content.encode()).decode(),
                "mode": mode,
            }
            for path, (content, mode) in files.items()
        },
        "command": command,
    }
    return ssh(
        host,
        "sudo -n python3 -c " + shlex.quote(bootstrap),
        data=json.dumps(payload),
        timeout=timeout,
    )


def load_state(path):
    path = Path(path).resolve()
    document = json.loads((path / "deployment.json").read_text())
    if document.get("version") != 1:
        raise ValueError("unsupported deployment state version")
    aliases = [ssh_target(host["ssh"]) for host in document["hosts"]]
    endpoints = [ssh_hostname(alias) for alias in aliases]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("SSH aliases resolve to the same host; fix the inventory")
    return path, document
