"""Shared model-facing visibility rules for Artifact-backed Toolsets."""

from __future__ import annotations

from collections.abc import Iterable

from contractor_runtime.contracts import ArtifactRef

MEMORY_ARTIFACT_PREFIX = "memory."
PURPOSE_RESERVED_NAMESPACES = frozenset({"inputs", "outputs", "skills"})


class ModelArtifactAccessError(ValueError):
    code = "artifact_access_denied"
    retryable = False

    def __init__(self) -> None:
        super().__init__("artifact is reserved for a purpose-specific Toolset")


def is_reserved_memory_binding(namespace: object, name: object) -> bool:
    """Return whether a Run binding belongs exclusively to MemoryTools."""

    return (
        isinstance(namespace, str)
        and namespace not in PURPOSE_RESERVED_NAMESPACES
        and isinstance(name, str)
        and name.startswith(MEMORY_ARTIFACT_PREFIX)
    )


def is_model_hidden_binding(namespace: object, name: object) -> bool:
    """Return whether generic model tools must not disclose this binding."""

    return namespace == "skills" or is_reserved_memory_binding(namespace, name)


def require_model_visible_binding(namespace: object, name: object) -> None:
    if is_model_hidden_binding(namespace, name):
        raise ModelArtifactAccessError


def model_visible_exact_refs(refs: Iterable[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    return tuple(ref for ref in refs if not is_model_hidden_binding(ref.namespace, ref.name))
