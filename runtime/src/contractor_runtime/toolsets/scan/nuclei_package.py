"""Bounded content-addressed private Nuclei packages and selected-rule provenance.

Only regular files are admitted. Source-relative opens refuse links and private
bytes are the sole scanner input after manifest verification. The operator can
compute an installation pin with this module's CLI; no scanner is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import jcs

MAX_FILES = 50_000
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PATH_BYTES = 1024
MAX_DEPTH = 64
MANIFEST_MEDIA_TYPE = "application/vnd.contractor.nuclei-manifest+json"
_ID = re.compile(rb"(?m)^id:[ \t]*([\"']?)([A-Za-z0-9_-]{1,256})\1[ \t]*(?:#[^\r\n]*)?\r?$")


class PackageError(Exception):
    """A fixed scanner preparation diagnostic without file paths or contents."""

    def __init__(self, code: str):
        super().__init__(code)


@dataclass(frozen=True)
class PackageSnapshot:
    root: Path
    digest: str
    files: list[dict]


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise PackageError("scan_timeout")


def snapshot_package(source: Path, destination: Path, deadline: float) -> PackageSnapshot:
    """Copy and hash every package file, off the event loop, under fixed bounds."""
    files: list[dict] = []
    count, total, manifest_bytes = 0, 0, 32
    flags = os.O_RDONLY | os.O_NOFOLLOW

    def copy_directory(fd: int, target: Path, prefix: str, depth: int) -> None:
        nonlocal count, total, manifest_bytes
        _check_deadline(deadline)
        if depth > MAX_DEPTH:
            raise PackageError("nuclei_package_invalid")
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                _check_deadline(deadline)
                count += 1
                if count > MAX_FILES * 2:
                    raise PackageError("nuclei_package_invalid")
                names.append(entry.name)
        for name in sorted(names, key=lambda name: name.encode("utf-8", errors="strict")):
            _check_deadline(deadline)
            relative = prefix + name
            if len(relative.encode("utf-8")) > MAX_PATH_BYTES or any(
                ord(c) < 32 or ord(c) == 127 for c in relative
            ):
                raise PackageError("nuclei_package_invalid")
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                (target / name).mkdir(mode=0o700)
                child = os.open(name, flags | os.O_DIRECTORY, dir_fd=fd)
                try:
                    copy_directory(child, target / name, relative + "/", depth + 1)
                finally:
                    os.close(child)
                continue
            if (
                not stat.S_ISREG(info.st_mode)
                or len(files) >= MAX_FILES
                or info.st_size > MAX_FILE_BYTES
            ):
                raise PackageError("nuclei_package_invalid")
            size = 0
            digest = hashlib.sha256()
            # O_NONBLOCK prevents a replacement FIFO from hanging preparation;
            # fstat rechecks type after open, not just the earlier directory entry.
            child = os.open(name, flags | os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(child, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise PackageError("nuclei_package_invalid")
                with (target / name).open("xb") as output:
                    os.fchmod(output.fileno(), 0o400)
                    while chunk := stream.read(64 * 1024):
                        _check_deadline(deadline)
                        size += len(chunk)
                        total += len(chunk)
                        if size > MAX_FILE_BYTES or total > MAX_PACKAGE_BYTES:
                            raise PackageError("nuclei_package_invalid")
                        output.write(chunk)
                        digest.update(chunk)
                after = os.fstat(stream.fileno())
                if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) or size != before.st_size:
                    raise PackageError("nuclei_package_invalid")
            record = {"path": relative, "digest": "sha256:" + digest.hexdigest(), "size": size}
            manifest_bytes += len(jcs.canonicalize(record)) + 1
            if manifest_bytes > MAX_MANIFEST_BYTES:
                raise PackageError("nuclei_package_invalid")
            files.append(record)

    try:
        destination.mkdir(mode=0o700, exist_ok=True)
        root = os.open(source, flags | os.O_DIRECTORY)
        try:
            copy_directory(root, destination, "", 0)
        finally:
            os.close(root)
        if not files:
            raise PackageError("nuclei_templates_unavailable")
        files.sort(key=lambda item: item["path"].encode("utf-8"))
        canonical = jcs.canonicalize({"schemaVersion": 1, "files": files})
        _check_deadline(deadline)
        return PackageSnapshot(
            destination, "sha256:" + hashlib.sha256(canonical).hexdigest(), files
        )
    except (OSError, UnicodeError):
        raise PackageError("nuclei_package_invalid") from None


def selected_manifest(
    snapshot: PackageSnapshot,
    output: bytes,
    engine_version: str,
    deadline: float,
    identifiers: dict[str, str] | None = None,
    source: dict | None = None,
) -> tuple[bytes, int]:
    """Resolve the engine's -tl selection exclusively within verified package bytes."""
    index = {record["path"]: record for record in snapshot.files}
    selected = {}
    try:
        for line in output.decode("utf-8", errors="strict").splitlines():
            _check_deadline(deadline)
            line = line.strip()
            if not line or line.startswith("Listing available  nuclei templates for "):
                continue
            path = Path(line)
            if path.is_absolute():
                line = path.relative_to(snapshot.root).as_posix()
            if line not in index or line in selected or not line.endswith((".yaml", ".yml")):
                raise ValueError
            data = (snapshot.root / line).read_bytes()
            if identifiers is None:
                identifier = _ID.search(data)
                if identifier is None:
                    raise ValueError
                name = identifier[2].decode("ascii")
            else:
                name = identifiers[line]
            selected[line] = {**index[line], "id": name}
        if not selected:
            raise ValueError
        manifest = {
            "schemaVersion": 1,
            "packageDigest": snapshot.digest,
            "files": snapshot.files,
            "selectedTemplates": [selected[name] for name in sorted(selected)],
            "engineVersion": engine_version,
        }
        if source is not None:
            manifest["source"] = source
        data = jcs.canonicalize(manifest)
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError
        _check_deadline(deadline)
        return data, len(selected)
    except (OSError, UnicodeError, ValueError, KeyError):
        raise PackageError("nuclei_selection_invalid") from None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the canonical installed Nuclei package pin"
    )
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="contractor-nuclei-pin-") as temporary:
        snapshot = snapshot_package(
            args.directory.resolve(), Path(temporary) / "package", time.monotonic() + 300
        )
        print(snapshot.digest)


if __name__ == "__main__":
    main()
