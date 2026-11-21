"""Attach one original exported memory through ordinary exact Artifact writes."""

import hashlib

import jcs

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactResponseLimitError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.artifacts import PURPOSE_RESERVED_NAMESPACES
from contractor_runtime.contracts.base import NAME_PATTERN
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.artifact_write import (
    ArtifactReconciliationFailed,
    definite_write_rejection,
    write_artifact_exact,
)
from contractor_runtime.toolsets.common.memory_codec import (
    SCHEMA_VERSION,
    StoredMemoryNote,
    artifact_name,
    encode_note,
)

SNAPSHOT_SCHEMA = "contractor.memory-snapshot/v1"
MAX_SNAPSHOT_BYTES = 512 * 1024
MAX_SNAPSHOT_MEMORIES = 16
MAX_MEMORY_FACT_CALLS = 8


def select_memory(data: bytes, name: str) -> bytes:
    artifact_name(name)  # Reuse Memory's name grammar, without reading a hidden binding.
    if len(data) > MAX_SNAPSHOT_BYTES:
        raise ValueError("memory snapshot exceeds its byte bound")
    snapshot = strict_json_loads(data.decode("utf-8"))
    if not isinstance(snapshot, dict) or set(snapshot) != {"schemaVersion", "namespace", "notes"}:
        raise ValueError("invalid memory snapshot")
    namespace = snapshot["namespace"]
    notes = snapshot["notes"]
    if (
        snapshot["schemaVersion"] != SNAPSHOT_SCHEMA
        or not isinstance(namespace, str)
        or NAME_PATTERN.fullmatch(namespace) is None
        or namespace in PURPOSE_RESERVED_NAMESPACES
        or not isinstance(notes, list)
        or not 1 <= len(notes) <= MAX_SNAPSHOT_MEMORIES
    ):
        raise ValueError("invalid memory snapshot")
    names = []
    for note in notes:
        if not isinstance(note, dict) or set(note) != {"name", "content", "description", "tags"}:
            raise ValueError("invalid exported memory")
        # The Server's snapshot validator checks the same stored-note encoding
        # with ordinal zero. No ordinal is exposed or invented in exported evidence.
        encode_note(StoredMemoryNote(SCHEMA_VERSION, **note, ordinal=0))
        names.append(note["name"])
    if names != sorted(set(names)) or jcs.canonicalize(snapshot) != data:
        raise ValueError("memory snapshot must be canonical and unambiguous")
    if name not in names:
        raise ValueError("memory name is absent from the selected snapshot")
    return jcs.canonicalize({**snapshot, "notes": [notes[names.index(name)]]})


async def memory_evidence(
    client: ArtifactClient, snapshot: ArtifactRef, name: str, namespace: str
) -> ArtifactRef:
    try:
        artifact_name(name)
        snapshot.require_exact()
    except ValueError:
        raise ToolInputError(
            "memory_name must name a memory in the assigned exact snapshot"
        ) from None
    try:
        value = await client.read_artifact(snapshot, max_bytes=MAX_SNAPSHOT_BYTES)
    except ArtifactResponseLimitError:
        raise ToolInputError("The assigned memories snapshot exceeds 512 KiB") from None
    except ArtifactAPIError as error:
        if error.status_code == 404:
            raise ToolInputError("The assigned memories snapshot is unavailable") from None
        raise
    if value.media_type != "application/json":
        raise ToolInputError("The assigned memories snapshot must be application/json")
    try:
        data = await threads.to_thread_until_done(
            select_memory, value.data, name, name="finding-original-memory"
        )
    except (ValueError, TypeError, RecursionError):
        raise ToolInputError(
            "memory_name must exist in the valid assigned memories snapshot; no fact was saved"
        ) from None
    target = ArtifactRef(
        namespace=namespace, name="fact-memory-" + hashlib.sha256(data).hexdigest()
    )
    try:
        saved = await write_artifact_exact(
            client,
            target,
            data=data,
            media_type="application/json",
            expected_revision=None,
            max_bytes=MAX_SNAPSHOT_BYTES,
            reject_first=lambda error: (
                definite_write_rejection(error) and error.code != "artifact_conflict"
            ),
            reject_replay=lambda error: (
                definite_write_rejection(error) and error.code != "artifact_conflict"
            ),
        )
    except ArtifactReconciliationFailed as error:
        if error.read_error is not None:
            raise error.read_error from None
        raise ToolInputError("Original memory evidence could not be reconciled") from None
    return saved.artifact.require_exact()
