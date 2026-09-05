from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contractor_runtime.memory import (
    ARTIFACT_NAME_PREFIX,
    MAXIMUM_ARTIFACT_NAME_BYTES,
    MAXIMUM_DESCRIPTION_BYTES,
    MAXIMUM_EXACT_ORDINAL,
    MAXIMUM_NAME_BYTES,
    MAXIMUM_PAYLOAD_BYTES,
    MAXIMUM_TAG_BYTES,
    REASON_MALFORMED,
    REASON_PAYLOAD_TOO_LARGE,
    SCHEMA_VERSION,
    MemoryCodecError,
    StoredMemoryNote,
    artifact_name,
    decode_note,
    encode_note,
    full_projection,
    name_from_artifact,
    normalize_note,
    preview_projection,
)

FIXTURES = Path(__file__).parents[2] / "testdata" / "memory" / "cases.json"


def test_shared_codec_fixtures() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    assert fixtures["schemaVersion"] == "contractor.memory-codec-fixtures/v1"

    for fixture in fixtures["valid"]:
        note = _note_from_input(fixture["input"])
        encoded = encode_note(note)
        assert encoded.decode() == fixture["canonical"], fixture["id"]
        assert decode_note(artifact_name(note.name), encoded) == normalize_note(note)

    for fixture in fixtures["generatedValid"]:
        note = _generated_note(fixture["input"])
        encoded = encode_note(note)
        assert len(encoded) == fixture["canonicalBytes"], fixture["id"]
        assert hashlib.sha256(encoded).hexdigest() == fixture["canonicalSha256"]
        assert decode_note(ARTIFACT_NAME_PREFIX + note.name, encoded) == note

    for fixture in fixtures["invalidInput"]:
        note = _mutated_note(fixture["mutation"])
        with pytest.raises(MemoryCodecError) as raised:
            encode_note(note)
        _assert_safe_reason(raised.value, fixture["reason"], note)

    for fixture in fixtures["invalidStored"]:
        with pytest.raises(MemoryCodecError) as raised:
            decode_note(fixture["bindingName"], fixture["payload"].encode())
        _assert_safe_reason(raised.value, fixture["reason"])


def test_codec_rejects_oversized_and_invalid_utf8_stored_payloads() -> None:
    with pytest.raises(MemoryCodecError) as oversized:
        decode_note("memory.note", b"x" * (MAXIMUM_PAYLOAD_BYTES + 1))
    assert oversized.value.reason == REASON_PAYLOAD_TOO_LARGE

    with pytest.raises(MemoryCodecError) as invalid_utf8:
        decode_note("memory.note", b"\xff")
    assert invalid_utf8.value.reason == REASON_MALFORMED


def test_artifact_name_mapping_and_model_projections() -> None:
    assert artifact_name("repo_overview") == "memory.repo_overview"
    assert name_from_artifact("memory.repo_overview") == "repo_overview"
    with pytest.raises(MemoryCodecError):
        name_from_artifact("report.repo_overview")
    maximum_name = "a" + "b" * (MAXIMUM_NAME_BYTES - 1)
    assert len(artifact_name(maximum_name)) == MAXIMUM_ARTIFACT_NAME_BYTES
    with pytest.raises(MemoryCodecError):
        artifact_name(maximum_name + "b")

    note = normalize_note(_base_note())
    created = datetime(2026, 9, 1, 10, tzinfo=UTC)
    updated = created + timedelta(seconds=1)
    full = asdict(full_projection(note, created, updated))
    preview = asdict(preview_projection(note, created, updated))
    assert set(full) == {
        "name",
        "content",
        "description",
        "tags",
        "ordinal",
        "created_at",
        "updated_at",
    }
    assert set(preview) == set(full) - {"content"}
    assert {"artifact", "revision", "namespace"}.isdisjoint(full)
    assert {"artifact", "revision", "namespace"}.isdisjoint(preview)


def _note_from_input(raw: dict[str, object]) -> StoredMemoryNote:
    return StoredMemoryNote(
        schema_version=str(raw["schemaVersion"]),
        name=str(raw["name"]),
        content=str(raw["content"]),
        description=str(raw["description"]),
        tags=tuple(str(tag) for tag in raw["tags"]),  # type: ignore[union-attr]
        ordinal=int(raw["ordinal"]),
    )


def _generated_note(raw: dict[str, object]) -> StoredMemoryNote:
    def repeated(field: str, repeat_field: str) -> str:
        value = raw.get(field, "")
        assert isinstance(value, str)
        repeat = raw.get(repeat_field)
        if repeat is None:
            return value
        assert isinstance(repeat, dict)
        return str(repeat["value"]) * int(repeat["count"])

    tags = [str(tag) for tag in raw.get("tags", [])]  # type: ignore[union-attr]
    for repeat in raw.get("tagRepeats", []):  # type: ignore[union-attr]
        assert isinstance(repeat, dict)
        tags.append(str(repeat["value"]) * int(repeat["count"]))
    return StoredMemoryNote(
        schema_version=str(raw["schemaVersion"]),
        name=repeated("name", "nameRepeat"),
        content=repeated("content", "contentRepeat"),
        description=repeated("description", "descriptionRepeat"),
        tags=tuple(tags),
        ordinal=int(raw["ordinal"]),
    )


def _base_note() -> StoredMemoryNote:
    return StoredMemoryNote(
        schema_version=SCHEMA_VERSION,
        name="safe_note",
        content="recognizable-content-canary",
        description="recognizable-description-canary",
        tags=("safe-tag",),
        ordinal=1,
    )


def _mutated_note(mutation: str) -> StoredMemoryNote:
    values: dict[str, object] = asdict(_base_note())
    values["tags"] = tuple(values["tags"])  # type: ignore[arg-type]
    if mutation == "uppercase_name":
        values["name"] = "Uppercase"
    elif mutation == "long_name":
        values["name"] = "a" * (MAXIMUM_NAME_BYTES + 1)
    elif mutation == "invalid_name_utf8":
        values["name"] = "\ud800"
    elif mutation == "empty_content":
        values["content"] = ""
    elif mutation == "invalid_content_utf8":
        values["content"] = "\ud800"
    elif mutation == "long_description_utf8":
        values["description"] = "é" * (MAXIMUM_DESCRIPTION_BYTES // 2 + 1)
    elif mutation == "invalid_description_utf8":
        values["description"] = "\ud800"
    elif mutation == "too_many_unique_tags":
        values["tags"] = ("a", "b", "c", "d")
    elif mutation == "invalid_tag":
        values["tags"] = ("Uppercase",)
    elif mutation == "long_tag":
        values["tags"] = ("a" * (MAXIMUM_TAG_BYTES + 1),)
    elif mutation == "ordinal_above_exact_range":
        values["ordinal"] = MAXIMUM_EXACT_ORDINAL + 1
    elif mutation == "payload_above_limit":
        values["content"] = "x" * 32651
    elif mutation == "wrong_schema":
        values["schema_version"] = "contractor.memory-note/v0"
    else:
        raise AssertionError(f"unknown fixture mutation {mutation}")
    return StoredMemoryNote(**values)  # type: ignore[arg-type]


def _assert_safe_reason(
    error: MemoryCodecError, expected: str, note: StoredMemoryNote | None = None
) -> None:
    assert error.reason == expected
    assert repr(error) == f"MemoryCodecError('invalid memory note ({expected})')"
    if note is not None:
        for canary in (note.content, note.description, *note.tags):
            if len(canary) >= 8:
                assert canary not in str(error)
