#!/usr/bin/env python3
"""Inspect local test storage; mutations require an explicit --apply.

No service or automatic cleanup is installed. PostgreSQL uses only the explicitly
configured CONTRACTOR_TEST_DATABASE_URL; subprocess diagnostics never expose it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

GIB = 1024**3
MAX_CACHE_ENTRIES = 500_000
MAX_DATABASES = 1000
CACHE_NAME = re.compile(r"[0-9a-f]{64}-[ad]")
TEST_DATABASE = re.compile(
    r"contractor_(?:test(?:_[a-z0-9_]+)?|v[0-9][a-z0-9_]*|"
    r"(?:budget|runservice|standby|lease|erd|restore|backup|completion)_[0-9a-f]{8,16})"
)
TEST_CONTAINER = re.compile(
    r"contractor-(?:litellm-contract-(?:db|proxy)-[0-9]+|"
    r"langfuse-bootstrap-test-[0-9]+|test-storage-[0-9a-f]{16})"
)
TEST_VOLUME = re.compile(
    r"contractor-(?:litellm-contract-data-[0-9]+|test-storage-[0-9a-f]{16})"
)
TEST_NETWORK = re.compile(
    r"contractor-(?:litellm-contract-[0-9]+|test-storage-[0-9a-f]{16})"
)


def run(command: list[str], *, sql: str | None = None, env: dict | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            input=sql,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(f"{command[0]} is required on PATH") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{command[0]} timed out; inspect storage before retrying"
        ) from None
    if result.returncode:
        # Connection errors can contain passwords, URLs or environment values.
        raise RuntimeError(
            f"{command[0]} failed; check connectivity, privileges and concurrent use"
        )
    return result.stdout


def cache_root() -> Path:
    value = run(["go", "env", "GOCACHE"]).strip()
    root = Path(value)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            "GOCACHE must name an existing absolute, non-symlink directory"
        )
    return root


def fingerprint(path: Path) -> tuple:
    value = path.lstat()
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


@dataclass
class Entry:
    path: Path
    files: tuple[Path, ...]
    stamps: tuple[tuple, ...]
    size: int
    used: float


def cache_entries(root: Path) -> list[Entry]:
    entries = []
    scanned = 0
    for shard_number in range(256):
        shard = root / f"{shard_number:02x}"
        if shard.is_symlink() or not shard.is_dir():
            continue
        for path in shard.iterdir():
            scanned += 1
            if scanned > MAX_CACHE_ENTRIES:
                raise RuntimeError(
                    "cache inventory exceeds 500000 entries; no pruning performed"
                )
            if not CACHE_NAME.fullmatch(path.name) or path.name[:2] != shard.name:
                continue
            try:
                info = path.lstat()
                if stat.S_ISREG(info.st_mode):
                    files = (path,)
                elif stat.S_ISDIR(info.st_mode) and path.name.endswith("-d"):
                    # Go executable entries contain a single regular file. Never
                    # recursively remove unknown layouts, symlinks or fuzz corpora.
                    files = tuple(islice(path.iterdir(), 2))
                    if len(files) != 1 or not stat.S_ISREG(files[0].lstat().st_mode):
                        continue
                else:
                    continue
                stamps = tuple(fingerprint(p) for p in (path, *files))
                size = sum(p.lstat().st_size for p in files)
                used = max(stamp[-1] / 1e9 for stamp in stamps)
            except FileNotFoundError:
                continue  # Go or another cleanup may have removed an entry.
            entries.append(Entry(path, files, stamps, size, used))
    return entries


def prune_cache(
    entries: list[Entry], target: int, cutoff: float, apply: bool
) -> tuple[int, int]:
    remaining = sum(entry.size for entry in entries)
    reclaimed = 0
    for entry in sorted(entries, key=lambda entry: (entry.used, str(entry.path))):
        if remaining <= target:
            break
        if entry.used >= cutoff:
            continue
        try:
            if (
                tuple(fingerprint(p) for p in (entry.path, *entry.files))
                != entry.stamps
            ):
                continue
            if apply:
                for path in entry.files:
                    path.unlink()
                if entry.files != (entry.path,):
                    entry.path.rmdir()
        except FileNotFoundError:
            continue
        remaining -= entry.size
        reclaimed += entry.size
    return reclaimed, remaining


class Postgres:
    def __init__(self) -> None:
        url = os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "")
        if not url:
            raise RuntimeError("CONTRACTOR_TEST_DATABASE_URL is required")
        self.command = [
            "psql",
            "-X",
            "-w",
            "-q",
            "--dbname",
            url,
            "-At",
            "-v",
            "ON_ERROR_STOP=1",
        ]
        self.env = {
            **os.environ,
            "PGCONNECT_TIMEOUT": "5",
            "PGOPTIONS": "-c statement_timeout=10000 -c lock_timeout=1000",
        }

    def sql(self, sql: str) -> str:
        return run(self.command, sql=sql, env=self.env)

    def inventory(self) -> dict[str, dict]:
        rows = json.loads(
            self.sql(f"""
SELECT coalesce(json_agg(d), '[]') FROM (
 SELECT oid::bigint, datname AS name, datistemplate AS template,
        datname = current_database() AS current,
        pg_database_size(oid) AS bytes,
        (SELECT count(*) FROM pg_stat_activity a WHERE a.datid = pg_database.oid) AS sessions
 FROM pg_database ORDER BY datname LIMIT {MAX_DATABASES + 1}
) d;
""")
        )
        if len(rows) > MAX_DATABASES:
            raise RuntimeError(
                "database inventory exceeds 1000 rows; no deletion performed"
            )
        return {row["name"]: row for row in rows}

    def drop(self, expected: dict) -> None:
        # Each separate DROP gets a fresh identity/session check. No FORCE: a
        # connection arriving after this read makes PostgreSQL reject the DROP.
        current = self.inventory().get(expected["name"])
        if current is None or current["oid"] != expected["oid"]:
            raise RuntimeError(
                "database identity changed; inspect storage before retrying"
            )
        require_disposable(current)
        self.sql('DROP DATABASE "' + current["name"] + '";')


def require_disposable(row: dict) -> None:
    if not TEST_DATABASE.fullmatch(row["name"]) or row["template"] or row["current"]:
        raise RuntimeError(f"protected database: {row['name']}")
    if row["sessions"]:
        raise RuntimeError(f"database has active connections: {row['name']}")


def gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / GIB:.2f} GiB"


def podman_inventory(kind: str) -> list[dict]:
    commands = {
        "container": ["ps", "-a"],
        "image": ["images", "--filter", "dangling=true"],
        "volume": ["volume", "ls"],
        "network": ["network", "ls"],
    }
    rows = json.loads(run(["podman", *commands[kind], "--format", "json"]))
    if len(rows) > 1000:
        raise RuntimeError(
            "Podman inventory exceeds 1000 resources; no removal performed"
        )
    return rows


def podman_select(kind: str, names: list[str]) -> list[dict]:
    rows = podman_inventory(kind)
    selected = []
    patterns = {
        "container": TEST_CONTAINER,
        "volume": TEST_VOLUME,
        "network": TEST_NETWORK,
    }
    for name in names:
        if kind == "image":
            if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", name):
                raise RuntimeError("select a full dangling image ID")
            found = next(
                (
                    row
                    for row in rows
                    if row["Id"].removeprefix("sha256:") == name.removeprefix("sha256:")
                ),
                None,
            )
            if found is None or found.get("Containers") or found.get("Names"):
                raise RuntimeError(
                    "image is missing, tagged or used; no removal performed"
                )
            if any(
                row["ImageID"].removeprefix("sha256:")
                == found["Id"].removeprefix("sha256:")
                for row in podman_inventory("container")
            ):
                raise RuntimeError("image is used by a container")
        else:
            if not patterns[kind].fullmatch(name):
                raise RuntimeError(f"protected Podman {kind}: {name}")
            found = next(
                (
                    row
                    for row in rows
                    if name in row.get("Names", [])
                    or name == row.get("Name")
                    or name == row.get("name")
                ),
                None,
            )
            if found is None:
                raise RuntimeError(f"Podman {kind} does not exist: {name}")
            if kind == "container" and found["State"] not in (
                "exited",
                "created",
                "stopped",
            ):
                raise RuntimeError(f"Podman container is active: {name}")
            if kind in ("volume", "network"):
                users = json.loads(
                    run(
                        [
                            "podman",
                            "ps",
                            "-a",
                            "--filter",
                            f"{kind}={name}",
                            "--format",
                            "json",
                        ]
                    )
                )
                if users:
                    raise RuntimeError(f"Podman {kind} is still attached: {name}")
        selected.append(found)
    return selected


def podman_clean(kind: str, names: list[str], apply: bool) -> None:
    if len(names) > 100 or len(set(names)) != len(names):
        raise RuntimeError("select at most 100 distinct Podman resources")
    selected = podman_select(kind, names)
    for name, expected in zip(names, selected, strict=True):
        if apply:
            current = podman_select(kind, [name])[0]
            identity = (
                "CreatedAt" if kind == "volume" else "id" if kind == "network" else "Id"
            )
            if current[identity] != expected[identity]:
                raise RuntimeError(
                    "Podman resource identity changed; inspect before retrying"
                )
            if kind == "container":
                command = ["rm", "--volumes", current["Id"]]
            elif kind == "image":
                command = ["image", "rm", "--no-prune", current["Id"]]
            else:
                command = [kind, "rm", name]
            run(["podman", *command])
        print(f"{'Removed' if apply else 'Would remove'} Podman {kind}: {name}")


def podman_report() -> None:
    print(run(["podman", "system", "df"]).strip())
    print(
        "Image sizes share layers; system df reclaimable is an estimate, not a deletion plan."
    )
    for kind in ("container", "volume", "network", "image"):
        for row in podman_inventory(kind):
            if kind == "container":
                print(f"container {','.join(row['Names'])}: {row['State']}")
            elif kind == "image":
                print(f"dangling image {row['Id']}: {gib(row['Size'])}")
            else:
                print(f"{kind} {row.get('Name', row.get('name'))}")
    print(
        "VM capacity and host disk allocation are separate: inspect podman machine list and guest df before maintenance."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("report")
    commands.add_parser("podman-report")
    podman = commands.add_parser("podman-clean")
    podman.add_argument("kind", choices=("container", "image", "volume", "network"))
    podman.add_argument("names", nargs="+")
    podman.add_argument("--apply", action="store_true")
    prune = commands.add_parser("prune-cache")
    prune.add_argument("--max-gib", type=float, default=20)
    prune.add_argument("--min-age-hours", type=float, default=24)
    prune.add_argument("--apply", action="store_true")
    drop = commands.add_parser("drop-databases")
    drop.add_argument("names", nargs="+")
    drop.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "podman-report":
            podman_report()
            return 0
        if args.command == "podman-clean":
            podman_clean(args.kind, args.names, args.apply)
            return 0
        if args.command in ("report", "prune-cache"):
            entries = cache_entries(cache_root())
            print(
                f"Go cache: {gib(sum(entry.size for entry in entries))}, {len(entries)} entries"
            )
            if args.command == "prune-cache":
                if not math.isfinite(args.max_gib * GIB) or args.max_gib < 0:
                    raise RuntimeError("--max-gib must be finite and nonnegative")
                if not math.isfinite(args.min_age_hours) or args.min_age_hours < 1:
                    raise RuntimeError("--min-age-hours must be finite and at least 1")
                # Go updates last-use timestamps at most once an hour.
                cutoff = time.time() - (args.min_age_hours + 1) * 3600
                reclaimed, remaining = prune_cache(
                    entries, int(args.max_gib * GIB), cutoff, args.apply
                )
                print(
                    f"{'Reclaimed' if args.apply else 'Would reclaim'} {gib(reclaimed)}; remaining {gib(remaining)}"
                )
                if remaining > args.max_gib * GIB:
                    print("Target not reached: recent or changed entries were retained")
                return 0
            if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
                print("PostgreSQL: not inspected (CONTRACTOR_TEST_DATABASE_URL unset)")
                return 0
        postgres = Postgres()
        databases = postgres.inventory()
        if args.command == "report":
            reported = []
            for row in databases.values():
                if row["name"].startswith("contractor_"):
                    reported.append(row)
                    eligible = (
                        bool(TEST_DATABASE.fullmatch(row["name"]))
                        and not row["template"]
                        and not row["current"]
                    )
                    print(
                        f"{row['name']}: {gib(row['bytes'])}, connections={row['sessions']}, {'test name' if eligible else 'protected'}"
                    )
            print(
                f"PostgreSQL: {len(reported)} Contractor databases, {gib(sum(row['bytes'] or 0 for row in reported))} known bytes"
            )
            return 0
        if len(args.names) > 100 or len(set(args.names)) != len(args.names):
            raise RuntimeError("select at most 100 distinct database names")
        selected = []
        for name in args.names:
            if name not in databases:
                raise RuntimeError(f"database does not exist: {name}")
            require_disposable(databases[name])
            selected.append(databases[name])
        for row in selected:
            if args.apply:
                postgres.drop(row)
            print(
                f"{'Dropped' if args.apply else 'Would drop'} {row['name']}: {gib(row['bytes'])}"
            )
        return 0
    except (RuntimeError, OSError, ValueError) as error:
        print(f"test-storage: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
