"""Complete bounded custom HTTP-rulepack parsing, with no installed-rule fallback."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import time
import unicodedata
import zipfile
import zlib
from pathlib import Path

import jcs

from contractor_runtime.toolsets.scan.nuclei_package import PackageError, PackageSnapshot
from contractor_runtime.toolsets.scan.rule_formats import (
    MAX_FILE_BYTES,
    InvalidRuleYAML,
    load_rule_yaml,
    preflight_rule_zip,
)

RULEPACK_MEDIA_TYPE = "application/vnd.contractor.nuclei-rulepack+zip"
MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_FILES = 1000
MAX_NAME_BYTES = 128
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,256}")
# Other protocols and Workflow templates cannot be hidden beside valid HTTP rules.
_PROTOCOLS = {
    "dns",
    "file",
    "headless",
    "tcp",
    "network",
    "ssl",
    "websocket",
    "whois",
    "code",
    "javascript",
    "workflows",
    "requests",
}


def _invalid() -> PackageError:
    return PackageError("nuclei_rulepack_invalid")


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise PackageError("scan_timeout")


def parse_template(data: bytes, deadline: float) -> dict:
    """Read bounded plain YAML without evaluating Nuclei variables or helpers."""
    try:
        return load_rule_yaml(data, deadline)
    except InvalidRuleYAML:
        raise _invalid() from None
    except TimeoutError:
        raise PackageError("scan_timeout") from None


def _template_id(data: bytes, deadline: float) -> str:
    """Validate structure before native Nuclei syntax validation of every file."""
    value = parse_template(data, deadline)
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("id"), str)
        or _IDENTIFIER.fullmatch(value["id"]) is None
        or not isinstance(value.get("info"), dict)
        or not isinstance(value.get("http"), list)
        or not value["http"]
        or any(not isinstance(request, dict) or not request for request in value["http"])
        or set(value) & _PROTOCOLS
    ):
        raise _invalid()
    return value["id"]


def _preflight_zip(data: bytes, deadline: float) -> None:
    try:
        preflight_rule_zip(data, deadline, MAX_FILES)
    except ValueError:
        raise _invalid() from None
    except TimeoutError:
        raise PackageError("scan_timeout") from None


def valid_rulepack_metadata(media_type: str, size: int) -> bool:
    """Check canonical media and encoded size; empty content is checked separately."""
    if size < 0:
        return False
    if media_type == "application/yaml":
        return size <= MAX_FILE_BYTES
    if media_type == RULEPACK_MEDIA_TYPE:
        return size <= MAX_ZIP_BYTES
    return False


def materialize_rulepack(
    data: bytes, media_type: str, root: Path, deadline: float
) -> tuple[PackageSnapshot, dict[str, str], str]:
    """Off-loop streaming extraction and full structure validation under fixed budgets."""
    _deadline(deadline)
    if not data or not valid_rulepack_metadata(media_type, len(data)):
        raise _invalid()
    files, identifiers = [], {}
    ids, names = set(), set()
    expanded = 0
    root.mkdir(mode=0o700, exist_ok=True)

    def add(name: str, stream) -> None:
        nonlocal expanded
        _deadline(deadline)
        normalized = unicodedata.normalize("NFC", name).casefold()
        if (
            len(files) >= MAX_FILES
            or not name.endswith((".yaml", ".yml"))
            or len(name.encode("utf-8")) > MAX_NAME_BYTES
            or name != name.strip()
            or any(char in name for char in "/\\:")
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or name.startswith(".")
            or normalized in names
        ):
            raise _invalid()
        names.add(normalized)
        content = bytearray()
        while chunk := stream.read(64 * 1024):
            _deadline(deadline)
            expanded += len(chunk)
            if len(content) + len(chunk) > MAX_FILE_BYTES or expanded > MAX_EXPANDED_BYTES:
                raise _invalid()
            content.extend(chunk)
        payload = bytes(content)
        identifier = _template_id(payload, deadline)
        if identifier in ids:
            raise _invalid()
        ids.add(identifier)
        with (root / name).open("xb") as output:
            os.fchmod(output.fileno(), 0o400)
            output.write(payload)
        files.append(
            {
                "path": name,
                "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
        identifiers[name] = identifier

    try:
        if media_type == "application/yaml":
            add("rulepack.yaml", io.BytesIO(data))
        else:
            _preflight_zip(data, deadline)
            with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
                entries = archive.infolist()
                if not 1 <= len(entries) <= MAX_FILES:
                    raise _invalid()
                for entry in entries:
                    mode = stat.S_IFMT(entry.external_attr >> 16)
                    if (
                        entry.filename != entry.orig_filename
                        or entry.is_dir()
                        or entry.external_attr & 0x10
                        or mode not in {0, stat.S_IFREG}
                        or entry.flag_bits & 1
                        or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or entry.file_size > MAX_FILE_BYTES
                    ):
                        raise _invalid()
                    with archive.open(entry, "r") as source:
                        add(entry.filename, source)
        files.sort(key=lambda record: record["path"].encode("utf-8"))
        canonical = jcs.canonicalize({"schemaVersion": 1, "files": files})
        _deadline(deadline)
        digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
        source_digest = "sha256:" + hashlib.sha256(data).hexdigest()
        return PackageSnapshot(root, digest, files), identifiers, source_digest
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile, RuntimeError, zlib.error):
        raise _invalid() from None
