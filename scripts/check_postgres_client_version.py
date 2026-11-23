#!/usr/bin/env python3
"""Check pg_dump and pg_restore against the test server's major version.

make test-backup-restore runs this before it creates any database. pg_dump
refuses a server of a newer major version, so an older client would otherwise
fail the rehearsal in the middle of its setup with only the tool's message. The
server version comes from SHOW server_version_num, read with psql from
CONTRACTOR_TEST_DATABASE_URL; the requirement is stated in
docs/testing/README.md and docs/operations/artifact-blob-storage.md.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

TOOLS = ("pg_dump", "pg_restore")


def server_version(version_num: str) -> tuple[int, str]:
    """Return the major version and the readable form of server_version_num."""
    if not re.fullmatch(r"[0-9]+", version_num):
        raise ValueError(f"unexpected server_version_num {version_num!r}")
    number = int(version_num)
    if number < 100000:
        raise ValueError(f"unsupported PostgreSQL server version_num {number}")
    return number // 10000, f"{number // 10000}.{number % 10000}"


def tool_version(tool: str, output: str) -> tuple[int, str]:
    """Return the major version and the version text of `<tool> --version`."""
    match = re.match(rf"{re.escape(tool)} \(PostgreSQL\) (([0-9]+)\S*)", output.strip())
    if match is None:
        raise ValueError(f"unexpected {tool} --version output {output.strip()!r}")
    return int(match.group(2)), match.group(1)


def problems(version_num: str, versions: dict[str, str]) -> list[str]:
    """Name every tool older than the server, with both versions."""
    server_major, server_text = server_version(version_num)
    found = []
    for tool in TOOLS:
        major, text = tool_version(tool, versions[tool])
        if major < server_major:
            found.append(
                f"{tool} {text} is older than the PostgreSQL server {server_text}; "
                f"use {tool} {server_major} or newer"
            )
    return found


def run(command: list[str], secret: str | None) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise SystemExit(f"{command[0]} is required on PATH") from None
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        for value in filter(None, (secret, unquote(secret or ""))):
            message = message.replace(value, "<redacted>")
        raise SystemExit(f"{command[0]} failed: {message}")
    return result.stdout


def main() -> int:
    url = os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "")
    if not url:
        raise SystemExit("CONTRACTOR_TEST_DATABASE_URL is required")
    secret = urlsplit(url).password
    versions = {tool: run([tool, "--version"], None) for tool in TOOLS}
    query = ["psql", url, "-X", "-v", "ON_ERROR_STOP=1", "-Atc", "SHOW server_version_num"]
    version_num = run(query, secret).strip()
    try:
        found = problems(version_num, versions)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if found:
        for problem in found:
            print(problem, file=sys.stderr)
        return 1
    _, server_text = server_version(version_num)
    tools = ", ".join(f"{tool} {tool_version(tool, versions[tool])[1]}" for tool in TOOLS)
    print(f"{tools}; PostgreSQL server {server_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
