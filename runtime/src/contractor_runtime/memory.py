"""Language-neutral Memory note codec; no ADK or storage policy lives here."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import jcs

SCHEMA_VERSION: Final = "contractor.memory-note/v1"
MEDIA_TYPE: Final = "application/vnd.contractor.memory-note+json"
ARTIFACT_NAME_PREFIX: Final = "memory."
MAXIMUM_ARTIFACT_NAME_BYTES: Final = 128
MAXIMUM_PAYLOAD_BYTES: Final = 32 * 1024
MAXIMUM_NAME_BYTES: Final = MAXIMUM_ARTIFACT_NAME_BYTES - len(ARTIFACT_NAME_PREFIX.encode("ascii"))
MAXIMUM_DESCRIPTION_BYTES: Final = 512
MAXIMUM_TAGS: Final = 3
MAXIMUM_TAG_BYTES: Final = 64
MAXIMUM_EXACT_ORDINAL: Final = (1 << 53) - 1

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
TAG_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")

REASON_NAME: Final = "name"
REASON_CONTENT: Final = "content"
REASON_DESCRIPTION: Final = "description"
REASON_TAGS: Final = "tags"
REASON_ORDINAL: Final = "ordinal"
REASON_PAYLOAD_TOO_LARGE: Final = "payload_too_large"
REASON_MALFORMED: Final = "malformed"
REASON_SCHEMA: Final = "schema"
REASON_BINDING_MISMATCH: Final = "binding_mismatch"
REASON_NONCANONICAL: Final = "noncanonical"

ERROR_REASONS: Final = frozenset(
    {
        REASON_NAME,
        REASON_CONTENT,
        REASON_DESCRIPTION,
        REASON_TAGS,
        REASON_ORDINAL,
        REASON_PAYLOAD_TOO_LARGE,
        REASON_MALFORMED,
        REASON_SCHEMA,
        REASON_BINDING_MISMATCH,
        REASON_NONCANONICAL,
    }
)


class MemoryCodecError(ValueError):
    """Bounded validation error that never retains supplied note text."""

    def __init__(self, reason: str) -> None:
        if reason not in ERROR_REASONS:
            raise ValueError("unknown Memory codec reason")
        self.reason = reason
        super().__init__(f"invalid memory note ({reason})")


@dataclass(frozen=True, slots=True)
class StoredMemoryNote:
    schema_version: str
    name: str
    content: str
    description: str
    tags: tuple[str, ...]
    ordinal: int


@dataclass(frozen=True, slots=True)
class MemoryNote:
    name: str
    content: str
    description: str
    tags: list[str]
    ordinal: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryPreview:
    name: str
    description: str
    tags: list[str]
    ordinal: int
    created_at: datetime
    updated_at: datetime


def full_projection(
    note: StoredMemoryNote, created_at: datetime, updated_at: datetime
) -> MemoryNote:
    return MemoryNote(
        name=note.name,
        content=note.content,
        description=note.description,
        tags=list(note.tags),
        ordinal=note.ordinal,
        created_at=created_at,
        updated_at=updated_at,
    )


def preview_projection(
    note: StoredMemoryNote, created_at: datetime, updated_at: datetime
) -> MemoryPreview:
    return MemoryPreview(
        name=note.name,
        description=note.description,
        tags=list(note.tags),
        ordinal=note.ordinal,
        created_at=created_at,
        updated_at=updated_at,
    )


def artifact_name(name: str) -> str:
    _validate_name(name)
    return ARTIFACT_NAME_PREFIX + name


def name_from_artifact(value: str) -> str:
    if not value.startswith(ARTIFACT_NAME_PREFIX):
        raise MemoryCodecError(REASON_NAME)
    name = value.removeprefix(ARTIFACT_NAME_PREFIX)
    _validate_name(name)
    return name


def normalize_note(note: StoredMemoryNote) -> StoredMemoryNote:
    if note.schema_version != SCHEMA_VERSION:
        raise MemoryCodecError(REASON_SCHEMA)
    _validate_name(note.name)
    _validate_content(note.content)
    _validate_description(note.description)
    _validate_ordinal(note.ordinal)
    tags = _normalize_tags(note.tags)
    return StoredMemoryNote(
        schema_version=SCHEMA_VERSION,
        name=note.name,
        content=note.content,
        description=note.description,
        tags=tags,
        ordinal=note.ordinal,
    )


def encode_note(note: StoredMemoryNote) -> bytes:
    normalized = normalize_note(note)
    try:
        encoded = jcs.canonicalize(_stored_mapping(normalized))
    except Exception:
        raise MemoryCodecError(REASON_MALFORMED) from None
    if len(encoded) > MAXIMUM_PAYLOAD_BYTES:
        raise MemoryCodecError(REASON_PAYLOAD_TOO_LARGE)
    return encoded


def decode_note(artifact_binding_name: str, payload: bytes) -> StoredMemoryNote:
    if not isinstance(payload, bytes):
        raise TypeError("Memory payload must be bytes")
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise MemoryCodecError(REASON_PAYLOAD_TOO_LARGE)
    try:
        text = payload.decode("utf-8")
        raw = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_JSONInteger,
            parse_float=_JSONFloat,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey):
        raise MemoryCodecError(REASON_MALFORMED) from None
    if not isinstance(raw, dict):
        raise MemoryCodecError(REASON_MALFORMED)
    if set(raw) != {
        "schemaVersion",
        "name",
        "content",
        "description",
        "tags",
        "ordinal",
    }:
        raise MemoryCodecError(REASON_SCHEMA)
    if (
        type(raw["schemaVersion"]) is not str
        or type(raw["name"]) is not str
        or type(raw["content"]) is not str
        or type(raw["description"]) is not str
        or type(raw["tags"]) is not list
        or any(type(tag) is not str for tag in raw["tags"])
        or type(raw["ordinal"]) is not _JSONInteger
    ):
        raise MemoryCodecError(REASON_MALFORMED)
    if raw["schemaVersion"] != SCHEMA_VERSION:
        raise MemoryCodecError(REASON_SCHEMA)
    ordinal_text = str(raw["ordinal"])
    if ordinal_text.startswith("-") or len(ordinal_text) > len(str(MAXIMUM_EXACT_ORDINAL)):
        raise MemoryCodecError(REASON_ORDINAL)
    ordinal = int(ordinal_text)
    note = StoredMemoryNote(
        schema_version=raw["schemaVersion"],
        name=raw["name"],
        content=raw["content"],
        description=raw["description"],
        tags=tuple(raw["tags"]),
        ordinal=ordinal,
    )
    _validate_name(note.name)
    try:
        note.content.encode("utf-8")
        note.description.encode("utf-8")
    except UnicodeEncodeError:
        raise MemoryCodecError(REASON_NONCANONICAL) from None
    _validate_content(note.content)
    _validate_description(note.description)
    _validate_ordinal(note.ordinal)
    normalized_tags = _normalize_tags(note.tags)
    if note.tags != normalized_tags:
        raise MemoryCodecError(REASON_TAGS)
    if name_from_artifact(artifact_binding_name) != note.name:
        raise MemoryCodecError(REASON_BINDING_MISMATCH)
    try:
        canonical = jcs.canonicalize(_stored_mapping(note))
    except Exception:
        raise MemoryCodecError(REASON_MALFORMED) from None
    if payload != canonical:
        raise MemoryCodecError(REASON_NONCANONICAL)
    return note


def _validate_name(name: object) -> None:
    if type(name) is not str:
        raise MemoryCodecError(REASON_NAME)
    try:
        encoded = name.encode("utf-8")
    except UnicodeEncodeError:
        raise MemoryCodecError(REASON_NAME) from None
    if not encoded or len(encoded) > MAXIMUM_NAME_BYTES or NAME_PATTERN.fullmatch(name) is None:
        raise MemoryCodecError(REASON_NAME)


def _validate_content(content: object) -> None:
    if type(content) is not str or not content:
        raise MemoryCodecError(REASON_CONTENT)
    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        raise MemoryCodecError(REASON_CONTENT) from None


def _validate_description(description: object) -> None:
    if type(description) is not str:
        raise MemoryCodecError(REASON_DESCRIPTION)
    try:
        encoded = description.encode("utf-8")
    except UnicodeEncodeError:
        raise MemoryCodecError(REASON_DESCRIPTION) from None
    if len(encoded) > MAXIMUM_DESCRIPTION_BYTES:
        raise MemoryCodecError(REASON_DESCRIPTION)


def _validate_ordinal(ordinal: object) -> None:
    if type(ordinal) is not int or not 0 <= ordinal <= MAXIMUM_EXACT_ORDINAL:
        raise MemoryCodecError(REASON_ORDINAL)


def _normalize_tags(tags: object) -> tuple[str, ...]:
    if not isinstance(tags, (tuple, list)):
        raise MemoryCodecError(REASON_TAGS)
    unique: set[str] = set()
    for tag in tags:
        if type(tag) is not str:
            raise MemoryCodecError(REASON_TAGS)
        try:
            encoded = tag.encode("utf-8")
        except UnicodeEncodeError:
            raise MemoryCodecError(REASON_TAGS) from None
        if len(encoded) > MAXIMUM_TAG_BYTES or TAG_PATTERN.fullmatch(tag) is None:
            raise MemoryCodecError(REASON_TAGS)
        unique.add(tag)
    if len(unique) > MAXIMUM_TAGS:
        raise MemoryCodecError(REASON_TAGS)
    return tuple(sorted(unique))


def _stored_mapping(note: StoredMemoryNote) -> dict[str, object]:
    return {
        "schemaVersion": note.schema_version,
        "name": note.name,
        "content": note.content,
        "description": note.description,
        "tags": list(note.tags),
        "ordinal": note.ordinal,
    }


class _DuplicateKey(Exception):
    pass


class _JSONInteger(str):
    pass


class _JSONFloat(str):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result
