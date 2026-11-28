"""Semgrep-owned local rules and bounded JSON adaptation; no registry requests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER
from contractor_runtime.contracts.settings import SourceArchiveLimits
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.source_archive import (
    _validate_entries,
    _validated_member_path,
)
from contractor_runtime.toolsets.scan.findings import _text
from contractor_runtime.toolsets.scan.nuclei_package import PackageError
from contractor_runtime.toolsets.scan.process import ProcessResult
from contractor_runtime.toolsets.scan.rule_formats import (
    MAX_FILE_BYTES,
    InvalidRuleYAML,
    load_rule_yaml,
    preflight_rule_zip,
)

ENGINE_VERSION = "1.187.0"
RULEPACK_MEDIA_TYPE = "application/vnd.contractor.semgrep-rulepack+zip"
MAX_ZIP_BYTES = 16 << 20
MAX_EXPANDED_BYTES = 64 << 20
MAX_RULES = 10_000
MAX_RESULTS = 100
MAX_RESULT_BYTES = 128 << 10
_RULE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,511}\Z")
_LIMITS = SourceArchiveLimits(
    max_expanded_bytes=MAX_EXPANDED_BYTES, max_file_bytes=MAX_FILE_BYTES, max_entries=MAX_RULES
)


def _invalid() -> PackageError:
    return PackageError("semgrep_rulepack_invalid")


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise PackageError("scan_timeout")


def _parse(data: bytes, deadline: float) -> list[dict]:
    try:
        value = load_rule_yaml(data, deadline)
        if not isinstance(value, dict) or set(value) != {"rules"}:
            raise _invalid()
        rules = value["rules"]
        if not isinstance(rules, list) or not 1 <= len(rules) <= MAX_RULES:
            raise _invalid()
        seen = set()
        for rule in rules:
            if (
                not isinstance(rule, dict)
                or not isinstance(rule.get("id"), str)
                or not _RULE_ID.fullmatch(rule["id"])
                or rule["id"] in seen
                or rule.get("severity")
                not in {"INFO", "WARNING", "ERROR", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
                or not isinstance(rule.get("languages"), list)
                or not rule["languages"]
            ):
                raise _invalid()
            _text(rule.get("message"), 8192)
            seen.add(rule["id"])
        # JSON encoding also rejects YAML dates, nonfinite values and recursive objects.
        json.dumps(value, allow_nan=False)
        return rules
    except TimeoutError:
        raise PackageError("scan_timeout") from None
    except (InvalidRuleYAML, UnicodeError, ValueError, TypeError, RecursionError):
        raise _invalid() from None


def snapshot_rules(
    installed: Path,
    destination: Path,
    deadline: float,
    *,
    data: bytes | None = None,
    media_type: str | None = None,
) -> dict:
    """Validate every selected file before writing one native local config.

    Installed upstream repositories exclude hidden paths, root template.yaml and
    *.test.yaml fixtures. Custom packages contain only YAML rule files, without
    those exclusions. File paths qualify rule IDs so upstream duplicate short
    IDs retain their identities without relying on native config-path rewriting.
    """
    files: list[tuple[str, bytes]] = []
    total = 0

    def add(name: str, raw: bytes) -> None:
        nonlocal total
        _check_deadline(deadline)
        _validated_member_path(name)
        total += len(raw)
        if (
            not name.endswith((".yaml", ".yml"))
            or not raw
            or len(raw) > MAX_FILE_BYTES
            or total > MAX_EXPANDED_BYTES
            or len(files) >= MAX_RULES
        ):
            raise _invalid()
        files.append((name, raw))

    try:
        if data is not None:
            if media_type == "application/yaml":
                add("rules.yaml", data)
            elif media_type == RULEPACK_MEDIA_TYPE and 0 < len(data) <= MAX_ZIP_BYTES:
                preflight_rule_zip(data, deadline, MAX_RULES)
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    entries = archive.infolist()
                    _validate_entries(entries, _LIMITS)
                    for entry in entries:
                        if entry.is_dir():
                            continue
                        with archive.open(entry) as stream:
                            raw = stream.read(MAX_FILE_BYTES + 1)
                        if len(raw) != entry.file_size:
                            raise _invalid()
                        add(entry.filename, raw)
            else:
                raise _invalid()
        else:
            if installed.is_symlink() or not installed.is_dir():
                raise PackageError("semgrep_rules_unavailable")
            entries_seen = 0
            pending = [installed]
            while pending:
                root = pending.pop()
                with os.scandir(root) as entries:
                    for entry in entries:
                        _check_deadline(deadline)
                        entries_seen += 1
                        if entries_seen > 50_000:
                            raise _invalid()
                        name = entry.name
                        if name.startswith(".") or (
                            root == installed and name in {"stats", "apex", "elixir"}
                        ):
                            continue
                        if entry.is_dir(follow_symlinks=True):
                            if entry.is_symlink():
                                raise _invalid()
                            pending.append(Path(entry.path))
                            continue
                        if (
                            not name.endswith((".yaml", ".yml"))
                            or name.endswith((".test.yaml", ".test.yml"))
                            or (root == installed and name == "template.yaml")
                        ):
                            continue
                        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                            raise _invalid()
                        path = Path(entry.path)
                        with path.open("rb") as stream:
                            raw = stream.read(MAX_FILE_BYTES + 1)
                        add(path.relative_to(installed).as_posix(), raw)
        if not files:
            raise _invalid()
        rules, manifest, qualified_ids = [], [], set()
        for name, raw in sorted(files):
            parsed = _parse(raw, deadline)
            for rule in parsed:
                # A hash namespace avoids punctuation/path collisions and excessive IDs.
                qualified = hashlib.sha256(name.encode()).hexdigest()[:16] + "." + rule["id"]
                if qualified in qualified_ids:
                    raise _invalid()
                qualified_ids.add(qualified)
                rules.append({**rule, "id": qualified})
            if len(rules) > MAX_RULES:
                raise _invalid()
            manifest.append([name, "sha256:" + hashlib.sha256(raw).hexdigest()])
        payload = json.dumps({"rules": rules}, ensure_ascii=False, allow_nan=False).encode()
        if len(payload) > MAX_EXPANDED_BYTES:
            raise _invalid()
        destination.write_bytes(payload)
        destination.chmod(0o600)
        return {
            "rulesDigest": "sha256:"
            + hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest(),
            "ruleFiles": len(files),
            "ruleCount": len(rules),
            "ruleIds": {r["id"] for r in rules},
        }
    except TimeoutError:
        raise PackageError("scan_timeout") from None
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise _invalid() from None


def observation(result: ProcessResult, prepared: dict) -> dict:
    value = {
        **result.observation(),
        "scanner": "semgrep",
        "stdout": "",
        "stderr": "",
        "diagnosticsRedacted": True,
        "results": [],
        "resultsTruncated": False,
        "invalidResultRows": 0,
        "scanComplete": False,
        **{k: v for k, v in prepared.items() if k not in {"ruleIds", "sourceFiles"}},
    }
    if not result.stdout:
        if not value["errorCode"]:
            value.update(status="failed", errorCode="invalid_scanner_output")
        return value
    try:
        doc = strict_json_loads(result.stdout)
        if (
            not isinstance(doc, dict)
            or not isinstance(doc.get("results"), list)
            or not isinstance(doc.get("errors"), list)
            or doc.get("version") != ENGINE_VERSION
            or not isinstance(doc.get("paths"), dict)
            or not isinstance(doc["paths"].get("scanned"), list)
        ):
            raise ValueError
        results, used = [], 0
        for row in doc["results"]:
            try:
                if not isinstance(row, dict):
                    raise ValueError
                path = row["path"]
                if not isinstance(path, str) or not path.startswith("source/"):
                    raise ValueError
                path = path.removeprefix("source/")
                if path not in prepared.get("sourceFiles", set()):
                    raise ValueError
                rule, extra = row["check_id"], row["extra"]
                if rule not in prepared.get("ruleIds", set()):
                    raise ValueError
                start, end = row["start"]["line"], row["end"]["line"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 1 <= start <= end <= MAX_EXACT_INTEGER
                ):
                    raise ValueError
                message, severity = _text(extra["message"], 8192), extra["severity"]
                if (
                    not isinstance(message, str)
                    or not message
                    or len(message.encode()) > 8192
                    or severity
                    not in {"INFO", "WARNING", "ERROR", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
                ):
                    raise ValueError
                normalized = {
                    "rule": rule,
                    "file": path,
                    "startLine": start,
                    "endLine": end,
                    "message": message,
                    "severity": severity,
                }
                size = len(json.dumps(normalized, ensure_ascii=False).encode())
                if len(results) == MAX_RESULTS or used + size > MAX_RESULT_BYTES:
                    value["resultsTruncated"] = True
                    continue
                results.append(normalized)
                used += size
            except (KeyError, TypeError, ValueError):
                value["invalidResultRows"] += 1
        value["results"] = results
        value["engineErrors"] = len(doc["errors"])
        scanned = doc["paths"]["scanned"]
        # Source extraction and native language/ignore/size filters can omit files.
        # Compare identities, not just counts; error messages and paths stay private.
        expected = {"source/" + p for p in prepared.get("sourceFiles", set())}
        covered = set(scanned) if all(isinstance(p, str) for p in scanned) else set()
        invalid_coverage = len(covered) != len(scanned) or bool(covered - expected)
        value["scannedFiles"] = len(covered & expected)
        if invalid_coverage and not value["errorCode"]:
            value.update(status="failed", errorCode="invalid_scanner_output")
        value["unscannedFiles"] = len(expected - covered)
        value["scanComplete"] = (
            bool(expected)
            and covered == expected
            and not (
                result.error_code
                or doc["errors"]
                or value["invalidResultRows"]
                or value["resultsTruncated"]
                or prepared.get("ignoredFiles", 0)
            )
        )
        if (doc["errors"] or value["invalidResultRows"]) and not value["errorCode"]:
            value.update(status="failed", errorCode="semgrep_scan_incomplete")
        return value
    except (KeyError, TypeError, ValueError, UnicodeError):
        value.update(status="failed", errorCode=value["errorCode"] or "invalid_scanner_output")
        return value


def finding_candidates(observed: dict, source: bytes) -> dict:
    """Select bounded code windows for explicit Finding outputs, off the event loop.

    The exact source digest fences the second read used for evidence publication.
    Only matched files are read (4 MiB/file, 8 MiB cumulative); a large source
    cannot amplify one 100-row batch into an unbounded archive rescan.
    """
    from contractor_runtime.toolsets.scan.findings import _batch, digest

    if "sha256:" + hashlib.sha256(source).hexdigest() != observed.get("sourceDigest"):
        raise ValueError("Semgrep source evidence changed")
    candidates, evidence, skipped, seen = [], [], {}, set()
    cached: dict[str, bytes] = {}
    read_bytes = 0
    severities = {
        "INFO": None,
        "LOW": "low",
        "WARNING": "medium",
        "MEDIUM": "medium",
        "ERROR": "high",
        "HIGH": "high",
        "CRITICAL": "critical",
    }
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        for row in observed.get("results", []):
            severity = severities[row["severity"]]
            if severity is None:
                skipped["informational"] = skipped.get("informational", 0) + 1
                continue
            path, start, end = row["file"], row["startLine"], row["endLine"]
            try:
                if end - start >= 400:
                    raise ValueError
                if path not in cached:
                    info = archive.getinfo(path)
                    if info.file_size > 4 << 20 or read_bytes + info.file_size > 8 << 20:
                        raise ValueError
                    with archive.open(info) as stream:
                        data = stream.read((4 << 20) + 1)
                    if len(data) != info.file_size or len(data) > 4 << 20:
                        raise ValueError
                    cached[path] = data
                    read_bytes += len(data)
                raw = cached[path]
                lines = raw.decode("utf-8").splitlines(keepends=True)
                if end > len(lines):
                    raise ValueError
                snippet = "".join(lines[start - 1 : end])
                if not snippet or len(snippet.encode()) > 4096:
                    raise ValueError
            except (KeyError, ValueError, UnicodeError, OSError, zipfile.BadZipFile):
                skipped["source_window_unavailable"] = (
                    skipped.get("source_window_unavailable", 0) + 1
                )
                continue
            rule = row["rule"]
            rule_key = (
                rule
                if len(rule.encode()) <= 160
                else "rule-" + hashlib.sha256(rule.encode()).hexdigest()
            )
            identity = [
                "semgrep",
                observed["rulesDigest"],
                observed["sourceDigest"],
                rule_key,
                path,
                str(start),
                str(end),
            ]
            key = digest(identity)
            if key in seen:
                skipped["duplicate_row"] = skipped.get("duplicate_row", 0) + 1
                continue
            seen.add(key)
            title = (
                ("Semgrep: " + row["message"].splitlines()[0])
                .encode()[:512]
                .decode("utf-8", errors="ignore")
            )
            candidates.append(
                {
                    "key": key,
                    "producer": "semgrep",
                    "rule": rule_key,
                    "title": title,
                    "assertion": row["message"],
                    "location": {"file": path, "range": {"start_line": start, "end_line": end}},
                    "severity": severity,
                    "qualifiedKey": identity,
                }
            )
            evidence.append(
                {
                    "key": key,
                    "producer": "semgrep",
                    "rule": rule,
                    "engineVersion": observed.get("engineVersion"),
                    "rulesDigest": observed["rulesDigest"],
                    "rulepackArtifact": observed.get("rulepackArtifact"),
                    "sourceArtifact": observed["sourceArtifact"],
                    "sourceDigest": observed["sourceDigest"],
                    "file": path,
                    "fileDigest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                    "startLine": start,
                    "endLine": end,
                    "text": snippet,
                    "matched": True,
                }
            )
    batch = _batch(observed, candidates, evidence, skipped, [])
    batch["complete"] = (
        batch["complete"]
        and observed.get("scanComplete", False)
        and not skipped.get("source_window_unavailable")
    )
    return batch
