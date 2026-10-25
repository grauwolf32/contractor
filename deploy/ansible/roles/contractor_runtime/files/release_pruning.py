"""Plan release deletion only after a bounded, complete ownership check.

Never prints unit contents, command lines or environment values. The role is
an exclusive host operation: do not concurrently edit its units or releases.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path

MAX_ENTRIES = 4096
MAX_UNITS = 256
MAX_BYTES = 65536
SCAN_SECONDS = 30
COMMIT = re.compile(r"[0-9a-f]{40}")
UNIT = re.compile(r"contractor-runtime-([a-z0-9][a-z0-9_-]{0,62})\.service")


class UnsafePruning(ValueError):
    """A complete release reference set could not be established."""


def entries(path: Path, *, optional: bool = False) -> list[Path]:
    if path.is_symlink():
        raise UnsafePruning("discovery directory is a symlink")
    try:
        with os.scandir(path) as scan:
            found = []
            for entry in scan:
                if len(found) == MAX_ENTRIES:
                    raise UnsafePruning("directory entry bound exceeded (4096)")
                found.append(Path(entry.path))
            return found
    except FileNotFoundError:
        if optional:
            return []
        raise UnsafePruning("required discovery directory is missing") from None


def read_small(path: Path) -> bytes:
    if path.is_symlink():
        raise UnsafePruning("reference metadata is a symlink")
    with path.open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise UnsafePruning("reference metadata byte bound exceeded (65536)")
    return data


def unit_releases(path: Path, releases: Path) -> set[Path]:
    # Only the literal, single ExecStart shape emitted by this role is owned.
    text = read_small(path).decode("utf-8").replace("\\\n", " ")
    commands = re.findall(r"^ExecStart=(.*)$", text, re.MULTILINE)
    if len(commands) != 1:
        raise UnsafePruning("unit must have exactly one literal ExecStart")
    executable = commands[0].split()[0] if commands[0].split() else ""
    expression = (
        re.escape(str(releases))
        + r"/([0-9a-f]{40})/runtime/\.venv/bin/contractor-runtime"
    )
    match = re.fullmatch(expression, executable)
    if match is None:
        raise UnsafePruning("unit ExecStart is outside the managed release layout")
    # Literal arguments and environment can also name resources in a release.
    # Conservative extra protection is preferable to deleting a referenced tree.
    reference = re.escape(str(releases)) + r"/([0-9a-f]{40})/"
    return {releases / commit for commit in re.findall(reference, text)}


def process_releases(pid: int, releases: Path, proc: Path) -> set[Path]:
    # Python's argv includes the installed console-script path. If the process
    # disappears or has an unexpected command line, retry instead of guessing.
    data = read_small(proc / str(pid) / "cmdline")
    expression = re.escape(os.fsencode(str(releases))) + rb"/([0-9a-f]{40})/"
    found = {releases / match.decode("ascii") for match in re.findall(expression, data)}
    if not found:
        raise UnsafePruning(
            "running unit has no identifiable managed release; retry after restart"
        )
    return found


def manager_output(argv: list[str], timeout: float) -> bytes:
    """Bound both output streams while reading; kill and join on every exit."""
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        output = bytearray()
        total = 0
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stderr, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    for key, _ in selector.select(remaining):
                        data = os.read(key.fd, min(4096, MAX_BYTES + 1 - total))
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        total += len(data)
                        if total > MAX_BYTES:
                            raise UnsafePruning(
                                "manager metadata byte bound exceeded (65536)"
                            )
                        if key.fileobj is process.stdout:
                            output.extend(data)
            process.wait(timeout=max(0, deadline - time.monotonic()))
            if process.returncode:
                raise UnsafePruning(
                    "manager query failed; check the user manager and retry"
                )
            return bytes(output)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def manager_state(argv: list[str], unit: Path, remaining: float) -> int:
    # Never request ExecStart or Environment, which can contain credentials.
    output = manager_output(
        argv + ["show", unit.name, "--property=MainPID,FragmentPath,DropInPaths"],
        min(5, remaining),
    )
    state = dict(line.split("=", 1) for line in output.decode().splitlines())
    if set(state) != {"MainPID", "FragmentPath", "DropInPaths"}:
        raise UnsafePruning("manager metadata is incomplete")
    if state["FragmentPath"] != str(unit) or state["DropInPaths"]:
        raise UnsafePruning(
            "unit has an unowned fragment or drop-in; reconcile before pruning"
        )
    pid = int(state["MainPID"])
    if pid < 0:
        raise UnsafePruning("invalid manager process identity")
    return pid


def plan(
    config: dict, *, proc: Path = Path("/proc"), state_reader=manager_state
) -> dict:
    deadline = time.monotonic() + SCAN_SECONDS
    releases = Path(config["prefix"]) / "releases"
    current = Path(config["current"])
    if current.parent != releases or COMMIT.fullmatch(current.name) is None:
        raise UnsafePruning("current release is outside the managed release layout")
    keep = int(config["keep"])
    if keep < 1:
        raise UnsafePruning("retention target must be positive")
    expected = set(config["instances"])
    units = []
    for path in entries(Path(config["home"]) / ".config/systemd/user"):
        if not path.name.startswith("contractor-runtime-"):
            continue
        match = UNIT.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise UnsafePruning(
                "unsupported Contractor unit metadata; reconcile before pruning"
            )
        if match[1] not in expected:
            raise UnsafePruning(
                "inventory omits a retained Contractor unit; include it or explicitly retire it"
            )
        units.append(path)
        if len(units) > MAX_UNITS:
            raise UnsafePruning("Contractor unit bound exceeded (256)")
    if {UNIT.fullmatch(path.name)[1] for path in units} != expected:
        raise UnsafePruning("inventory and installed Contractor units disagree")
    markers = entries(Path(config["config_dir"]) / "restart-pending", optional=True)
    if any(
        path.name not in expected
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != 0
        for path in markers
    ):
        raise UnsafePruning("unsupported restart marker; reconcile before pruning")
    if markers:
        # A new unit can already name the new release while its old process
        # still needs the previous one. Do not infer that release from the unit.
        return {"remove": [], "deferred": True}
    found = entries(releases)
    if any(
        path.is_symlink() or not path.is_dir() or COMMIT.fullmatch(path.name) is None
        for path in found
    ):
        raise UnsafePruning(
            "unsupported release directory entry; reconcile before pruning"
        )
    protected = {current}
    for unit in units:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UnsafePruning("release discovery deadline exceeded (30 seconds)")
        protected.update(unit_releases(unit, releases))
        pid = state_reader(config["systemctl"], unit, remaining)
        if pid:
            protected.update(process_releases(pid, releases, proc))
    if not protected <= set(found):
        raise UnsafePruning("a retained unit or process references a missing release")
    newest = sorted(
        (path for path in found if path not in protected),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    # keep is the TOTAL target, not an additional number of unreferenced copies.
    remove = newest[max(0, keep - len(protected)) :]
    return {"remove": [str(path) for path in remove], "deferred": False}


def main() -> int:
    try:
        data = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise UnsafePruning("pruning input byte bound exceeded")
        result = plan(json.loads(data))
    except UnsafePruning as error:
        print(f"Release cleanup refused: {error}.", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        print(
            "Release cleanup refused: discovery failed; check units, manager and process metadata, then retry.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
