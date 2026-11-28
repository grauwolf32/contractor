"""Shared bounded source ZIP interpretation for source readers and static scanners."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from contractor_runtime.contracts.settings import SourceArchiveLimits, default_source_archive_limits

_DEFAULT_LIMITS = default_source_archive_limits()
MAX_ARCHIVE_ENTRIES = _DEFAULT_LIMITS.max_entries
MAX_TOTAL_UNCOMPRESSED_BYTES = _DEFAULT_LIMITS.max_expanded_bytes
MAX_FILE_UNCOMPRESSED_BYTES = _DEFAULT_LIMITS.max_file_bytes
MAX_PATH_UTF8_BYTES = 512

IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "vendor",
    }
)

BINARY_EXTENSIONS = frozenset(
    {
        ".7z",
        ".a",
        ".avi",
        ".bin",
        ".bmp",
        ".class",
        ".db",
        ".dll",
        ".dylib",
        ".eot",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".lockb",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".otf",
        ".pdf",
        ".png",
        ".pyc",
        ".so",
        ".sqlite",
        ".tar",
        ".tiff",
        ".ttf",
        ".wav",
        ".webp",
        ".woff",
        ".woff2",
        ".xz",
        ".zip",
    }
)


@dataclass(frozen=True, slots=True)
class _SourceFile:
    path: str
    size: int
    digest: str


def _extract_archive(
    data: bytes, staging: Path, limits: SourceArchiveLimits | None = None
) -> tuple[list[_SourceFile], int, int]:
    limits = limits or default_source_archive_limits()
    staging.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            entries = archive.infolist()
            _validate_entries(entries, limits)
            files: list[_SourceFile] = []
            ignored_count = 0
            total_bytes = 0
            for info in entries:
                normalized = _validated_member_path(info.filename)
                if info.is_dir():
                    if _is_ignored_path(normalized, is_dir=True):
                        ignored_count += 1
                    continue
                if (
                    _is_ignored_path(normalized)
                    or PurePosixPath(normalized).suffix.lower() in BINARY_EXTENSIONS
                ):
                    ignored_count += 1
                    continue
                try:
                    payload = _read_member_bounded(archive, info, limits)
                except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                    raise ValueError("source ZIP member could not be read safely") from error
                try:
                    payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    ignored_count += 1
                    continue
                total_bytes += len(payload)
                target = staging.joinpath(*PurePosixPath(normalized).parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                target.write_bytes(payload)
                files.append(
                    _SourceFile(
                        path=normalized,
                        size=len(payload),
                        digest=hashlib.sha256(payload).hexdigest(),
                    )
                )
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError("source artifact is not a valid bounded ZIP") from error
    if not files:
        raise ValueError("source ZIP contains no readable UTF-8 files")
    files.sort(key=lambda item: item.path)
    return files, total_bytes, ignored_count


def _validate_entries(
    entries: list[zipfile.ZipInfo], limits: SourceArchiveLimits | None = None
) -> None:
    limits = limits or default_source_archive_limits()
    if len(entries) > limits.max_entries:
        raise ValueError(f"source ZIP exceeds the {limits.max_entries} entry limit")
    seen: set[str] = set()
    regular_files: set[str] = set()
    total_declared = 0
    normalized_entries: list[tuple[zipfile.ZipInfo, str]] = []
    for info in entries:
        normalized = _validated_member_path(info.filename)
        if normalized in seen:
            raise ValueError("source ZIP contains duplicate normalized paths")
        seen.add(normalized)
        if info.flag_bits & 0x1:
            raise ValueError("encrypted source ZIP members are not supported")
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        allowed_kind = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
        if kind not in {0, allowed_kind}:
            raise ValueError("source ZIP contains a link or special file")
        if info.file_size < 0 or info.file_size > limits.max_file_bytes:
            raise ValueError(
                f"source ZIP member exceeds the {limits.max_file_bytes} byte file limit"
            )
        total_declared += info.file_size
        if total_declared > limits.max_expanded_bytes:
            raise ValueError(
                f"source ZIP exceeds the {limits.max_expanded_bytes} byte uncompressed limit"
            )
        normalized_entries.append((info, normalized))
        if not info.is_dir():
            regular_files.add(normalized)
    for _info, normalized in normalized_entries:
        parts = PurePosixPath(normalized).parts
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in regular_files:
                raise ValueError("source ZIP nests an entry below a regular file")


def _validated_member_path(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise ValueError("source ZIP contains an unsafe member path")
    if len(raw.encode("utf-8")) > MAX_PATH_UTF8_BYTES:
        raise ValueError("source ZIP member path exceeds the 512-byte limit")
    candidate = raw[:-1] if raw.endswith("/") else raw
    raw_parts = candidate.split("/")
    if not candidate or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("source ZIP contains an unsafe member path")
    path = PurePosixPath(candidate)
    if path.is_absolute():
        raise ValueError("source ZIP contains an unsafe member path")
    if any(":" in part for part in path.parts):
        raise ValueError("source ZIP contains an unsafe member path")
    normalized = path.as_posix()
    if not normalized:
        raise ValueError("source ZIP contains an unsafe member path")
    return normalized


def _read_member_bounded(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, limits: SourceArchiveLimits
) -> bytes:
    with archive.open(info, mode="r") as source:
        payload = source.read(limits.max_file_bytes + 1)
        if len(payload) > limits.max_file_bytes or source.read(1):
            raise ValueError(
                f"source ZIP member exceeds the {limits.max_file_bytes} byte streamed limit"
            )
    if len(payload) != info.file_size:
        raise ValueError("source ZIP member size differs from its declaration")
    return payload


def _is_ignored_path(path: str, *, is_dir: bool = False) -> bool:
    parts = PurePosixPath(path).parts
    return any(part in IGNORED_DIRECTORY_NAMES for part in (parts if is_dir else parts[:-1]))
