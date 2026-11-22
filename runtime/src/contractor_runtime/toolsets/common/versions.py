"""Private, bounded read observations and model-response write bases."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding

MAX_SELECTED_VERSIONS = 4096
MAX_SELECTED_VERSION_BYTES = 4 * 1024 * 1024


class MaterialSelector(BaseModel):
    """Model selection, deliberately distinct from an exact stored ArtifactRef."""

    model_config = ConfigDict(extra="forbid", strict=True)
    namespace: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)


class SelectedVersions:
    def __init__(self) -> None:
        self._current: dict[tuple[str, str], ArtifactRef] = {}
        self._response: dict[tuple[str, str], ArtifactRef] | None = None
        self._size = 0
        self._inputs: tuple[ArtifactRef, ...] = ()
        self.findings: dict[str, int] = {}

    def clear(self) -> None:
        self._current.clear()
        self.findings.clear()
        self._inputs = ()
        self._response = None
        self._size = 0

    def assign(self, refs) -> None:
        self._inputs = tuple(ref.require_exact().model_copy(deep=True) for ref in refs)

    def selected(self, namespace: str, name: str) -> ArtifactRef | None:
        inputs = {
            ref.revision: ref
            for ref in self._inputs
            if (ref.namespace, ref.name) == (namespace, name)
        }
        if len(inputs) > 1:
            raise ToolInputError("Ambiguous input material; select an unambiguous task input")
        if inputs:
            return next(iter(inputs.values()))
        ref = self._current.get((namespace, name))
        return ref if ref is not None and ref.revision is not None else None

    def resolve(self, value: Any) -> ArtifactRef:
        try:
            selector = (
                value
                if isinstance(value, MaterialSelector)
                else MaterialSelector.model_validate(value)
            )
        except (ValueError, TypeError):
            raise ToolInputError(
                "Material selector requires namespace/name only; Runtime supplies revision"
            ) from None
        ref = ArtifactRef(namespace=selector.namespace, name=selector.name)
        require_model_visible_binding(ref.namespace, ref.name)
        selected = self.selected(ref.namespace, ref.name)
        if selected is None:
            raise ToolInputError("Material is not selected; read or select it before using it")
        return selected

    def discover_findings(self, items) -> None:
        if (
            len(self._current) + len(set(self.findings) | {item["findingId"] for item in items})
            > MAX_SELECTED_VERSIONS
        ):
            raise ToolInputError("Finding discovery limit reached; use fewer findings")
        if (
            self._size + 160 * len(set(self.findings) | {item["findingId"] for item in items})
            > MAX_SELECTED_VERSION_BYTES
        ):
            raise ToolInputError("Selected material metadata limit reached")
        for item in items:
            self.findings[item["findingId"]] = item["revision"]

    def finding_revision(self, identity: str) -> int:
        if identity not in self.findings:
            raise ToolInputError("Finding is not discovered; call list_project_findings first")
        return self.findings[identity]

    def begin_response(self) -> None:
        # Sibling calls cannot borrow a version another sibling just committed.
        self._response = self._current.copy()

    def base(self, namespace: str, name: str) -> str | None:
        refs = self._current if self._response is None else self._response
        ref = refs.get((namespace, name))
        return ref.revision if ref else None

    def admit_write(self, ref: ArtifactRef) -> None:
        key = (ref.namespace, ref.name)
        if key not in self._current:
            if len(self._current) + len(self.findings) >= MAX_SELECTED_VERSIONS:
                raise ToolInputError("Selected material limit reached; use fewer materials")
            # Reserve the binding before I/O; parallel new writes share the bound.
            # No revision means no update authority, even if the write later fails.
            self._store(ref.model_copy(update={"revision": None}), 1024)

    def _store(self, ref: ArtifactRef, size: int) -> None:
        key = (ref.namespace, ref.name)
        old = self._current.get(key)
        old_size = (
            (1024 if old.revision is None else len(old.model_dump_json().encode())) if old else 0
        )
        total = self._size + size - old_size
        if total + 160 * len(self.findings) > MAX_SELECTED_VERSION_BYTES or (
            old is None and len(self._current) + len(self.findings) >= MAX_SELECTED_VERSIONS
        ):
            raise ToolInputError("Selected material limit reached; use fewer materials")
        self._current[key], self._size = ref, total

    def observe(self, ref: ArtifactRef) -> None:
        exact = ref.require_exact().model_copy(deep=True)
        self._store(exact, len(exact.model_dump_json().encode()))


class ModelArtifacts:
    """Observe content reads/writes while keeping the underlying exact API strict.

    Only opted-in model tool owners use this adapter. Metadata/list operations
    are forwarded without granting a write base. Internal exact writes still
    supply their precondition explicitly.
    """

    def __init__(self, client: Any, versions: SelectedVersions) -> None:
        self.client, self.versions = client, versions

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    async def select(self, namespace: str, name: str) -> ArtifactRef:
        require_model_visible_binding(namespace, name)
        ref = self.versions.selected(namespace, name)
        if ref is not None:
            return ref
        # An explicit initial source selection resolves current metadata once.
        metadata = await self.client.read_artifact_metadata(
            ArtifactRef(namespace=namespace, name=name)
        )
        return metadata.artifact.require_exact()

    async def read_artifact(self, ref: ArtifactRef, **kwargs: Any) -> Any:
        value = await self.client.read_artifact(ref, **kwargs)
        self.versions.observe(value.artifact)
        return value

    def observe_cached_read(self, ref: ArtifactRef) -> None:
        observe = getattr(self.client, "observe_cached_read", None)
        if callable(observe):
            observe(ref)
        self.versions.observe(ref)
