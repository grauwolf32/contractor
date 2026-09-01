from __future__ import annotations

import hashlib
import io
import re
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode
from yaml.tokens import AliasToken, AnchorToken, ScalarToken, TagToken

MEDIA_TYPE = "application/vnd.contractor.agent-skill+zip"

CODE_ARCHIVE_INVALID = "skill_archive_invalid"
CODE_PATH_INVALID = "skill_path_invalid"
CODE_MEMBER_FORBIDDEN = "skill_member_forbidden"
CODE_MANIFEST_INVALID = "skill_manifest_invalid"
CODE_NAME_MISMATCH = "skill_name_mismatch"
CODE_LIMIT_EXCEEDED = "skill_limit_exceeded"

MAXIMUM_ARCHIVE_BYTES = 16 << 20
MAXIMUM_ENTRIES = 2_000
MAXIMUM_EXPANDED_BYTES = 32 << 20
MAXIMUM_MANIFEST_BYTES = 256 << 10
MAXIMUM_FRONTMATTER_BYTES = 32 << 10
MAXIMUM_RESOURCE_BYTES = 1 << 20
MAXIMUM_PATH_BYTES = 512
MAXIMUM_PATH_COMPONENTS = 8
MAXIMUM_YAML_NODES = 128
MAXIMUM_YAML_DEPTH = 3
MAXIMUM_METADATA_ENTRIES = 32
MAXIMUM_DESCRIPTION_BYTES = 1_024
MAXIMUM_LICENSE_BYTES = 512
MAXIMUM_COMPATIBILITY_BYTES = 500
MAXIMUM_METADATA_VALUE_BYTES = 1_024

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PORTABLE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_METADATA_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_STRING_TAG = "tag:yaml.org,2002:str"
_MAP_TAG = "tag:yaml.org,2002:map"


class SkillPackageError(ValueError):
    """Stable validation failure that contains no attacker-controlled content."""

    def __init__(self, code: str, member: str = "") -> None:
        self.code = code
        self.member = member
        message = f'{code}: member "{member}"' if member else code
        super().__init__(message)


@dataclass(frozen=True)
class Manifest:
    name: str
    description: str
    license: str = ""
    compatibility: str = ""
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Resource:
    path: str
    size: int


@dataclass(frozen=True)
class ValidatedMember:
    path: str
    _data: bytes = field(repr=False)

    @property
    def size(self) -> int:
        return len(self._data)

    def data(self) -> bytes:
        return bytes(self._data)


@dataclass(frozen=True)
class SkillPackage:
    manifest: Manifest
    digest: str
    resources: tuple[Resource, ...]
    stored_bytes: int
    expanded_bytes: int
    members: tuple[ValidatedMember, ...] = field(repr=False)

    def member(self, path: str) -> ValidatedMember | None:
        return next((member for member in self.members if member.path == path), None)


def validate_package(payload: bytes, expected_name: str = "") -> SkillPackage:
    """Validate exact ZIP bytes without extracting them or producing side effects."""
    if len(payload) > MAXIMUM_ARCHIVE_BYTES:
        raise SkillPackageError(CODE_LIMIT_EXCEEDED)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), mode="r")
        entries = archive.infolist()
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise SkillPackageError(CODE_ARCHIVE_INVALID) from None

    with archive:
        if len(entries) > MAXIMUM_ENTRIES:
            raise SkillPackageError(CODE_LIMIT_EXCEEDED)
        if archive.comment and len(archive.comment) > 65_535:
            raise SkillPackageError(CODE_ARCHIVE_INVALID)

        paths: dict[str, bool] = {}
        members: list[ValidatedMember] = []
        expanded = 0
        declared_expanded = 0
        manifest_count = 0
        for entry in entries:
            path, is_directory = _validate_member_path(entry.filename)
            if path in paths:
                raise SkillPackageError(CODE_PATH_INVALID, path)
            _reject_path_collision(paths, path, is_directory)
            paths[path] = is_directory
            if entry.flag_bits & 1 or entry.compress_type not in {
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            }:
                raise SkillPackageError(CODE_ARCHIVE_INVALID, path)

            mode = (entry.external_attr >> 16) & 0xFFFF
            if is_directory:
                if not _allowed_directory(path) or (mode and not stat.S_ISDIR(mode)):
                    raise SkillPackageError(CODE_MEMBER_FORBIDDEN, path)
                if entry.file_size != 0:
                    raise SkillPackageError(CODE_ARCHIVE_INVALID, path)
                continue
            if (mode and not stat.S_ISREG(mode)) or not _allowed_file(path):
                raise SkillPackageError(CODE_MEMBER_FORBIDDEN, path)

            limit = MAXIMUM_RESOURCE_BYTES
            if path == "SKILL.md":
                manifest_count += 1
                limit = MAXIMUM_MANIFEST_BYTES
            if entry.file_size > limit or entry.file_size > MAXIMUM_EXPANDED_BYTES - expanded:
                raise SkillPackageError(CODE_LIMIT_EXCEEDED, path)
            if entry.file_size > MAXIMUM_EXPANDED_BYTES - declared_expanded:
                raise SkillPackageError(CODE_LIMIT_EXCEEDED, path)
            declared_expanded += entry.file_size
            data = _read_bounded(archive, entry, min(limit, MAXIMUM_EXPANDED_BYTES - expanded))
            expanded += len(data)
            if path == "SKILL.md" or path.startswith("references/"):
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    raise SkillPackageError(CODE_MANIFEST_INVALID, path) from None
                if b"\x00" in data:
                    raise SkillPackageError(CODE_MANIFEST_INVALID, path)
            members.append(ValidatedMember(path=path, _data=data))

    if manifest_count != 1:
        raise SkillPackageError(CODE_MANIFEST_INVALID)
    members.sort(key=lambda member: member.path)
    manifest_member = next(member for member in members if member.path == "SKILL.md")
    manifest = _parse_manifest(manifest_member._data, expected_name)
    resources = tuple(
        Resource(path=member.path, size=member.size)
        for member in members
        if member.path != "SKILL.md"
    )
    return SkillPackage(
        manifest=manifest,
        digest=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        resources=resources,
        stored_bytes=len(payload),
        expanded_bytes=expanded,
        members=tuple(members),
    )


def _validate_member_path(raw: str) -> tuple[str, bool]:
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError:
        raise SkillPackageError(CODE_PATH_INVALID) from None
    if not raw or len(encoded) > MAXIMUM_PATH_BYTES or "\\" in raw or raw.startswith("/"):
        raise SkillPackageError(CODE_PATH_INVALID)
    is_directory = raw.endswith("/")
    path = raw[:-1] if is_directory else raw
    components = path.split("/")
    if (
        not path
        or len(components) > MAXIMUM_PATH_COMPONENTS
        or any(component in {"", ".", ".."} for component in components)
    ):
        raise SkillPackageError(CODE_PATH_INVALID)
    for index, component in enumerate(components):
        if index == 0 and len(components) == 1 and component == "SKILL.md":
            continue
        if not _PORTABLE_COMPONENT.fullmatch(component):
            raise SkillPackageError(CODE_PATH_INVALID)
    return path, is_directory


def _reject_path_collision(paths: Mapping[str, bool], path: str, is_directory: bool) -> None:
    components = path.split("/")
    for index in range(1, len(components)):
        ancestor = "/".join(components[:index])
        if ancestor in paths and not paths[ancestor]:
            raise SkillPackageError(CODE_PATH_INVALID, path)
    prefix = f"{path}/"
    if not is_directory and any(existing.startswith(prefix) for existing in paths):
        raise SkillPackageError(CODE_PATH_INVALID, path)


def _read_bounded(archive: zipfile.ZipFile, entry: zipfile.ZipInfo, limit: int) -> bytes:
    try:
        with archive.open(entry, mode="r") as stream:
            data = stream.read(limit + 1)
            if len(data) > limit or stream.read(1):
                raise SkillPackageError(CODE_LIMIT_EXCEEDED, entry.filename)
    except SkillPackageError:
        raise
    except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise SkillPackageError(CODE_ARCHIVE_INVALID, entry.filename) from None
    return data


def _parse_manifest(data: bytes, expected_name: str) -> Manifest:
    if len(data) > MAXIMUM_MANIFEST_BYTES:
        raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md") from None
    if "\x00" in text or not data.startswith(b"---\n"):
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
    closing = data.find(b"\n---\n", 4)
    if closing < 0:
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
    frontmatter = data[4:closing]
    body = data[closing + 5 :]
    if len(frontmatter) > MAXIMUM_FRONTMATTER_BYTES:
        raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
    if not body.strip():
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")

    try:
        token_nodes = 0
        for token in yaml.scan(frontmatter):
            if isinstance(token, (AliasToken, AnchorToken, TagToken)):
                raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
            if isinstance(token, ScalarToken) or type(token).__name__ in {
                "BlockMappingStartToken",
                "BlockSequenceStartToken",
                "FlowMappingStartToken",
                "FlowSequenceStartToken",
            }:
                token_nodes += 1
                if token_nodes > MAXIMUM_YAML_NODES:
                    raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
        documents = list(yaml.compose_all(frontmatter))
    except SkillPackageError:
        raise
    except yaml.YAMLError:
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md") from None
    if len(documents) != 1 or documents[0] is None:
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
    root = documents[0]
    _validate_yaml_tree(root)
    manifest = _construct_manifest(root)
    if expected_name and manifest.name != expected_name:
        raise SkillPackageError(CODE_NAME_MISMATCH, "SKILL.md")
    return manifest


def _validate_yaml_tree(root: Node) -> None:
    nodes = 0

    def walk(node: Node, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAXIMUM_YAML_NODES or depth > MAXIMUM_YAML_DEPTH:
            raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
        if isinstance(node, MappingNode):
            if node.tag != _MAP_TAG:
                raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
            children = (child for pair in node.value for child in pair)
        elif isinstance(node, ScalarNode):
            if node.tag != _STRING_TAG:
                raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
            children = ()
        else:
            raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
        for child in children:
            walk(child, depth + 1)

    walk(root, 1)
    if not isinstance(root, MappingNode):
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")


def _construct_manifest(root: MappingNode) -> Manifest:
    values: dict[str, Node] = {}
    allowed = {"name", "description", "license", "compatibility", "metadata"}
    for key_node, value_node in root.value:
        key = _scalar_string(key_node)
        if key is None or key not in allowed or key in values:
            raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
        values[key] = value_node

    name = _scalar_string(values.get("name"))
    description = _scalar_string(values.get("description"))
    if name is None:
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
    if len(name.encode()) > 64:
        raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
    if not _SKILL_NAME.fullmatch(name):
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
    if description is None or not description.strip():
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
    if len(description.encode()) > MAXIMUM_DESCRIPTION_BYTES:
        raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
    license_value = _optional_bounded_string(values, "license", MAXIMUM_LICENSE_BYTES)
    compatibility = _optional_bounded_string(values, "compatibility", MAXIMUM_COMPATIBILITY_BYTES)
    metadata_node = values.get("metadata")
    metadata: dict[str, str] = {}
    if metadata_node is not None:
        if not isinstance(metadata_node, MappingNode) or metadata_node.tag != _MAP_TAG:
            raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
        if len(metadata_node.value) > MAXIMUM_METADATA_ENTRIES:
            raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
        for key_node, value_node in metadata_node.value:
            key = _scalar_string(key_node)
            value = _scalar_string(value_node)
            if key is None or value is None or key.startswith("adk_") or key in metadata:
                raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
            if len(key.encode()) > 64 or len(value.encode()) > MAXIMUM_METADATA_VALUE_BYTES:
                raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
            if not _METADATA_KEY.fullmatch(key):
                raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
            metadata[key] = value
    return Manifest(
        name=name,
        description=description,
        license=license_value,
        compatibility=compatibility,
        metadata=MappingProxyType(metadata),
    )


def _scalar_string(node: Node | None) -> str | None:
    if not isinstance(node, ScalarNode) or node.tag != _STRING_TAG:
        return None
    return node.value


def _optional_bounded_string(values: Mapping[str, Node], key: str, limit: int) -> str:
    if key not in values:
        return ""
    value = _scalar_string(values[key])
    if value is None:
        raise SkillPackageError(CODE_MANIFEST_INVALID, "SKILL.md")
    if len(value.encode()) > limit:
        raise SkillPackageError(CODE_LIMIT_EXCEEDED, "SKILL.md")
    return value


def _allowed_directory(path: str) -> bool:
    return path in {"references", "assets"} or path.startswith(("references/", "assets/"))


def _allowed_file(path: str) -> bool:
    return path == "SKILL.md" or path.startswith(("references/", "assets/"))
