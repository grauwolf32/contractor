"""Shared model-facing visibility rules for Artifact-backed Toolsets."""

from __future__ import annotations

from collections.abc import Iterable

from contractor_runtime.contracts import ArtifactRef

MEMORY_ARTIFACT_PREFIX = "memory."
HTTP_BODY_ARTIFACT_PREFIX = "http.body."
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

    return (
        namespace == "skills"
        or is_reserved_memory_binding(namespace, name)
        or (
            isinstance(namespace, str)
            and namespace not in PURPOSE_RESERVED_NAMESPACES
            and isinstance(name, str)
            and name.startswith(HTTP_BODY_ARTIFACT_PREFIX)
        )
    )


def require_model_visible_binding(namespace: object, name: object) -> None:
    if is_model_hidden_binding(namespace, name):
        raise ModelArtifactAccessError


def model_visible_exact_refs(refs: Iterable[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    return tuple(ref for ref in refs if not is_model_hidden_binding(ref.namespace, ref.name))


def artifact_observation_cursor(client: object) -> int:
    value = getattr(client, "observation_cursor", 0)
    if type(value) is not int or value < 0:
        raise TypeError("ArtifactClient observation cursor must be a non-negative integer")
    return value


def model_visible_observations_since(client: object, cursor: int) -> tuple[ArtifactRef, ...]:
    observed = getattr(client, "observed_exact_refs_since", None)
    if not callable(observed):
        return ()
    return model_visible_exact_refs(observed(cursor))


def clear_artifact_observations(client: object) -> None:
    clear = getattr(client, "clear_observations", None)
    if callable(clear):
        clear()
