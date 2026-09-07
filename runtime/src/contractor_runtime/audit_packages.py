"""Pure legacy Audit package codecs shared by tools and trusted completion.

This module has no model, collector or artifact transport dependencies. The @1
entry points retain their original validation and byte encoding behavior.
"""

import hashlib
import io
import json
import re
import stat
import zipfile
from typing import Any

import jcs

PACKAGE_SCHEMA = "contractor.audit.package.v1"
TASK_SCHEMA = "contractor.audit.item-task.v1"
EXECUTION_SCHEMA = "contractor.audit.execution-manifest.v1"
RESULT_SCHEMA = "contractor.audit.check-results.v1"
EVIDENCE_SCHEMA = "contractor.audit.evidence.v1"
PACKAGE_MEDIA_TYPE = "application/zip"
JSON_MEDIA_TYPE = "application/json"
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_MEMBER_BYTES = 16 * 1024 * 1024
MAX_BATCH_ITEMS = 64
MAX_SUMMARY_BYTES = 16 * 1024
MAX_VALUES = 512
MAX_EVIDENCE = 256
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


def _validate_values(field: str, values: list[str]) -> None:
    if not isinstance(values, list) or len(values) > MAX_VALUES:
        raise ValueError(f"{field} exceeds its bound")
    if values != sorted(set(values)):
        raise ValueError(f"{field} must be sorted and unique")
    for value in values:
        if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
            raise ValueError(f"{field} contains an invalid value")


def _decode_task_package(payload: bytes, media_type: str) -> tuple[dict[str, Any], str]:
    if media_type != PACKAGE_MEDIA_TYPE or not 0 < len(payload) <= MAX_PACKAGE_BYTES:
        raise ValueError("task input is not an Audit package")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            if sorted(archive.namelist()) != ["manifest.json", "task.json"]:
                raise ValueError("task package members are invalid")
            manifest_bytes = _read_bounded(archive, "manifest.json")
            task_bytes = _read_bounded(archive, "task.json")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("task package is invalid") from error
    manifest = _canonical_object(manifest_bytes, "task package manifest")
    if (
        set(manifest) != {"schema", "package_id", "kind", "members"}
        or manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("kind") != "item-task"
        or not isinstance(manifest.get("package_id"), str)
        or IDENTIFIER.fullmatch(manifest["package_id"]) is None
    ):
        raise ValueError("task package manifest is invalid")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 1 or not isinstance(members[0], dict):
        raise ValueError("task package manifest is invalid")
    member = members[0]
    if (
        member.get("id") != "task-document"
        or member.get("path") != "task.json"
        or member.get("media_type") != JSON_MEDIA_TYPE
        or member.get("size") != len(task_bytes)
        or member.get("digest") != _digest(task_bytes)
    ):
        raise ValueError("task package member is invalid")
    task = _canonical_object(task_bytes, "task document")
    if task.get("schema") != TASK_SCHEMA:
        raise ValueError("task document schema is invalid")
    return task, manifest["package_id"]


def _decode_task_input(payload: bytes, media_type: str) -> list[tuple[dict[str, Any], str, bytes]]:
    try:
        task, package_id = _decode_task_package(payload, media_type)
        return [(task, package_id, payload)]
    except ValueError:
        try:
            return _decode_task_set(payload, media_type)
        except ValueError as batch_error:
            raise ValueError("task input is not a valid Audit task or task set") from batch_error


def _decode_task_set(payload: bytes, media_type: str) -> list[tuple[dict[str, Any], str, bytes]]:
    if media_type != PACKAGE_MEDIA_TYPE or not 0 < len(payload) <= MAX_PACKAGE_BYTES:
        raise ValueError("task set input is not an Audit package")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            manifest_bytes = _read_bounded(archive, "manifest.json")
            manifest = _canonical_object(manifest_bytes, "task set manifest")
            members = manifest.get("members")
            if (
                set(manifest) != {"schema", "package_id", "kind", "members"}
                or manifest.get("schema") != PACKAGE_SCHEMA
                or manifest.get("kind") != "item-task-set"
                or not isinstance(manifest.get("package_id"), str)
                or IDENTIFIER.fullmatch(manifest["package_id"]) is None
                or not isinstance(members, list)
                or not 2 <= len(members) <= MAX_BATCH_ITEMS
            ):
                raise ValueError("task set manifest is invalid")
            expected_paths = ["manifest.json"] + [
                f"tasks/{index:03d}.zip" for index in range(len(members))
            ]
            if sorted(archive.namelist()) != expected_paths:
                raise ValueError("task set package members are invalid")
            result: list[tuple[dict[str, Any], str, bytes]] = []
            for index, member in enumerate(members):
                path = f"tasks/{index:03d}.zip"
                if (
                    not isinstance(member, dict)
                    or set(member) != {"id", "path", "media_type", "size", "digest"}
                    or member.get("id") != f"task-{index:03d}"
                    or member.get("path") != path
                    or member.get("media_type") != PACKAGE_MEDIA_TYPE
                ):
                    raise ValueError("task set member is invalid")
                nested = _read_bounded(archive, path, MAX_PACKAGE_MEMBER_BYTES)
                if member.get("size") != len(nested) or member.get("digest") != _digest(nested):
                    raise ValueError("task set member digest is invalid")
                task, package_id = _decode_task_package(nested, PACKAGE_MEDIA_TYPE)
                result.append((task, package_id, nested))
            return result
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("task set package is invalid") from error


def _decode_execution_manifest(payload: bytes, media_type: str) -> dict[str, Any]:
    if media_type != JSON_MEDIA_TYPE:
        raise ValueError("execution manifest input is not JSON")
    manifest = _canonical_object(payload, "execution manifest")
    items = manifest.get("items")
    if (
        set(manifest) != {"schema", "items"}
        or manifest.get("schema") != EXECUTION_SCHEMA
        or not isinstance(items, list)
        or not 1 <= len(items) <= MAX_BATCH_ITEMS
    ):
        raise ValueError("execution manifest membership is invalid")
    seen_items: set[str] = set()
    seen_packages: set[str] = set()
    for index, item in enumerate(items):
        if (
            not isinstance(item, dict)
            or item.get("ordinal") != index
            or not isinstance(item.get("item_key"), str)
            or IDENTIFIER.fullmatch(item["item_key"]) is None
            or not isinstance(item.get("subject_key"), str)
            or IDENTIFIER.fullmatch(item["subject_key"]) is None
            or not isinstance(item.get("task_package_id"), str)
            or IDENTIFIER.fullmatch(item["task_package_id"]) is None
            or not isinstance(item.get("task_package_digest"), str)
            or not item["task_package_digest"].startswith("sha256:")
        ):
            raise ValueError("execution manifest item is invalid")
        if item["item_key"] in seen_items or item["task_package_id"] in seen_packages:
            raise ValueError("execution manifest identity is duplicated")
        seen_items.add(item["item_key"])
        seen_packages.add(item["task_package_id"])
    return manifest


def _match_trusted_inputs(
    tasks: list[tuple[dict[str, Any], str, bytes]],
    execution: dict[str, Any],
) -> None:
    if len(tasks) != len(execution["items"]):
        raise ValueError("task and execution manifest membership differs")
    for index, (task, task_package_id, task_package) in enumerate(tasks):
        item = execution["items"][index]
        if (
            item.get("item_key") != task.get("item_key")
            or item.get("subject_key") != task.get("subject_key")
            or item.get("task_package_id") != task_package_id
            or item.get("task_package_digest") != _digest(task_package)
        ):
            raise ValueError("task and execution manifest identities differ")
        if not isinstance(task.get("item_key"), str) or not isinstance(
            task.get("subject_key"), str
        ):
            raise ValueError("task identity is invalid")


def _build_result_package(
    *,
    tasks: list[dict[str, Any]],
    execution_bytes: bytes,
    results: list[dict[str, Any]],
    allow_empty: bool = False,
) -> bytes:
    if (not tasks and not allow_empty) or len(tasks) != len(results):
        raise ValueError("result membership differs from trusted tasks")
    evidence_values: list[dict[str, str]] = []
    evidence_members: list[tuple[str, str, str, bytes]] = []
    result_values: list[dict[str, Any]] = []
    for task_index, (task, result) in enumerate(zip(tasks, results, strict=True)):
        requested = _requested_coverage(task)
        completed = result["completed"]
        if any(value not in requested for value in completed):
            raise ValueError("completed coverage is outside the trusted request")
        evidence_ids: list[str] = []
        for evidence_index, evidence in enumerate(result["evidence"]):
            if len(tasks) == 1:
                evidence_id = f"ev-{evidence_index + 1}"
            else:
                evidence_id = f"ev-{task_index + 1}-{evidence_index + 1}"
            content_id = evidence_id.replace("ev-", "ev-content-", 1)
            evidence_ids.append(evidence_id)
            evidence_values.append(
                {
                    "id": evidence_id,
                    "kind": evidence["kind"],
                    "summary": evidence["summary"],
                    "content_member_id": content_id,
                }
            )
            evidence_members.append(
                (
                    content_id,
                    f"evidence/{evidence_id}.txt",
                    "text/plain",
                    evidence["summary"].encode("utf-8"),
                )
            )
        result_values.append(
            {
                "item_key": task["item_key"],
                "subject_key": task["subject_key"],
                "assessment": result["assessment"],
                "summary": result["summary"],
                "evidence_ids": evidence_ids,
                "coverage": {
                    "requested": requested,
                    "completed": completed,
                    "gaps": result["gaps"],
                },
                "proposals": result["proposals"],
            }
        )
    result_document = {
        "schema": RESULT_SCHEMA,
        "execution_manifest_digest": _digest(execution_bytes),
        "results": result_values,
    }
    members: list[tuple[str, str, str, bytes]] = [
        ("check-results", "check-results.json", JSON_MEDIA_TYPE, jcs.canonicalize(result_document))
    ]
    if evidence_values:
        members.append(
            (
                "evidence",
                "evidence.json",
                JSON_MEDIA_TYPE,
                jcs.canonicalize({"schema": EVIDENCE_SCHEMA, "evidence": evidence_values}),
            )
        )
        members.extend(evidence_members)
    package_identity = hashlib.sha256(b"".join(member[3] for member in members)).hexdigest()[:32]
    package_manifest = {
        "schema": PACKAGE_SCHEMA,
        "package_id": "result-" + package_identity,
        "kind": "check-results",
        "members": [
            {
                "id": member_id,
                "path": path,
                "media_type": media_type,
                "size": len(data),
                "digest": _digest(data),
            }
            for member_id, path, media_type, data in sorted(members, key=lambda item: item[1])
        ],
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        _write_member(archive, "manifest.json", jcs.canonicalize(package_manifest))
        for _, path, _, data in sorted(members, key=lambda item: item[1]):
            _write_member(archive, path, data)
    payload = output.getvalue()
    if len(payload) > MAX_PACKAGE_BYTES:
        raise ValueError("result package exceeds its aggregate bound")
    return payload


def _requested_coverage(task: dict[str, Any]) -> list[str]:
    checklist = task.get("checklist")
    operation = task.get("operation")
    finding = task.get("finding")
    if isinstance(checklist, dict) and operation is None:
        requested = checklist.get("required_evidence")
        if not isinstance(requested, list):
            raise ValueError("checklist requested coverage is invalid")
        _validate_values("requested coverage", requested)
        return requested
    if isinstance(operation, dict) and checklist is None:
        return ["operation-resolution"]
    if isinstance(finding, dict) and checklist is None and operation is None:
        method = finding.get("method")
        if not isinstance(method, str) or IDENTIFIER.fullmatch(method) is None:
            raise ValueError("finding requested coverage is invalid")
        return [method]
    raise ValueError("task kind is invalid")


def _canonical_object(payload: bytes, name: str) -> dict[str, Any]:
    if not payload or len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"{name} exceeds its bound")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict) or jcs.canonicalize(value) != payload:
        raise ValueError(f"{name} is not a canonical object")
    return value


def _read_bounded(archive: zipfile.ZipFile, name: str, limit: int = MAX_DOCUMENT_BYTES) -> bytes:
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size > limit:
        raise ValueError("Audit package member exceeds its bound")
    with archive.open(info, mode="r") as source:
        result = source.read(limit + 1)
    if len(result) > limit or len(result) != info.file_size:
        raise ValueError("Audit package member exceeds its bound")
    return result


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_STORED
    archive.writestr(info, data)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()
