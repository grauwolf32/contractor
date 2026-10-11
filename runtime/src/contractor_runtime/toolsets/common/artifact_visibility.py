"""Shared model-facing visibility rules for Artifact-backed Toolsets."""

from __future__ import annotations

from collections.abc import Iterable

from contractor_runtime.contracts import ArtifactRef

MEMORY_ARTIFACT_PREFIX = "memory."
HTTP_BODY_ARTIFACT_PREFIX = "http.body."
PURPOSE_RESERVED_NAMESPACES = frozenset({"inputs", "outputs", "skills"})
# Model visibility (ART-R17, ART-R19, ART-R24, ART-R109), the same rule as the
# Server's artifactpolicy.IsModelHiddenBinding, which also excludes these
# bindings from the private paged listing; both are checked against
# api/testdata/v1alpha1/artifact-list-cases.json.
MODEL_HIDDEN_NAMESPACES = frozenset({"skills", "contractor-system"})
MODEL_HIDDEN_NAME_PREFIXES = (MEMORY_ARTIFACT_PREFIX, HTTP_BODY_ARTIFACT_PREFIX)
MODEL_NAME_PREFIX_EXEMPT_NAMESPACES = frozenset({"inputs", "outputs"})


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

    if not isinstance(namespace, str):
        return False
    if namespace in MODEL_HIDDEN_NAMESPACES:
        return True
    if namespace in MODEL_NAME_PREFIX_EXEMPT_NAMESPACES:
        return False
    return isinstance(name, str) and name.startswith(MODEL_HIDDEN_NAME_PREFIXES)


def require_model_visible_binding(namespace: object, name: object) -> None:
    if is_model_hidden_binding(namespace, name):
        raise ModelArtifactAccessError


def require_model_visible_namespace(namespace: object) -> None:
    """Reject a Namespace whose every binding is hidden from model tools."""

    if isinstance(namespace, str) and namespace in MODEL_HIDDEN_NAMESPACES:
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


class ArtifactObservingTool:
    """Expose model-visible Artifact observations of the tool's ArtifactClient."""

    _client: object

    def _observed_artifact_client(self) -> object:
        return self._client

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        client = self._observed_artifact_client()
        return model_visible_exact_refs(getattr(client, "known_exact_refs", ()))

    @property
    def artifact_observation_cursor(self) -> int:
        return artifact_observation_cursor(self._observed_artifact_client())

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        return model_visible_observations_since(self._observed_artifact_client(), cursor)

    def clear_artifact_observations(self) -> None:
        clear_artifact_observations(self._observed_artifact_client())
