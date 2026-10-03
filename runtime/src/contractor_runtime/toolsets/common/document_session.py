"""Lock-guarded sessions over one document written to a CAS Artifact binding."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from contractor_runtime.artifacts import ArtifactClient, ArtifactResponseLimitError
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding
from contractor_runtime.toolsets.common.document_write import write_document_exact
from contractor_runtime.toolsets.common.input_errors import ToolInputError


class DocumentSession[D]:
    """Hold the current document of one Worker-namespace binding.

    Subclasses decode seeds, serialize documents and project the document
    state; loading, CAS writes, validation-task tracking and close are shared.
    """

    media_type: str
    seed_media_types: frozenset[str]
    max_bytes: int
    reload_tool: str
    read_tool: str
    seed_revision_message: str
    seed_too_large_message: str
    seed_media_type_message: str
    not_loaded_message: str
    not_loaded_code: str | None = None

    def __init__(self, client: ArtifactClient, namespace: str) -> None:
        self._client = client
        self._namespace = namespace
        self._lock = asyncio.Lock()
        self._validation_task: asyncio.Task | None = None
        self._document: D | None = None
        self._target_name: str | None = None
        self._revision: str | None = None

    def _decode_seed(self, data: bytes) -> D:
        raise NotImplementedError

    def _serialize(self, document: D) -> bytes:
        raise NotImplementedError

    def _document_state(
        self, artifact: ArtifactRef, target_name: str, document: D, *, changed: bool, copied: bool
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def load(
        self,
        *,
        namespace: str,
        name: str,
        revision: str | None,
        target_name: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        validate_target_name(target_name)
        require_model_visible_binding(namespace, name)
        require_model_visible_binding(self._namespace, target_name)
        if namespace != self._namespace and revision is None:
            raise ToolInputError(self.seed_revision_message)
        source_ref = ArtifactRef(namespace=namespace, name=name, revision=revision)
        async with self._lock:
            try:
                value = await self._client.read_artifact(source_ref, max_bytes=self.max_bytes)
            except ArtifactResponseLimitError:
                raise ToolInputError(self.seed_too_large_message) from None
            if revision is not None and value.artifact.revision != revision:
                raise ValueError("Artifact API did not preserve the requested exact revision")
            if value.media_type not in self.seed_media_types:
                raise ToolInputError(self.seed_media_type_message)
            document = self._decode_seed(value.data)
            same_binding = namespace == self._namespace and name == target_name
            if same_binding and value.media_type == self.media_type:
                self._remember(document, target_name, value.artifact)
                return self._document_state(
                    value.artifact, target_name, document, changed=False, copied=False
                )
            written = await self._write(
                document,
                target_name=target_name,
                expected_revision=(value.artifact.revision if same_binding else expected_revision),
            )
            self._remember(document, target_name, written.artifact)
            return self._document_state(
                written.artifact, target_name, document, changed=True, copied=True
            )

    async def close(self) -> None:
        task = self._validation_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        async with self._lock:
            self._document = None
            self._target_name = None
            self._revision = None

    async def _validating[T](self, operation: Awaitable[T]) -> T:
        """Await a validator so close() can cancel it from another task."""

        self._validation_task = asyncio.current_task()
        try:
            return await operation
        finally:
            self._validation_task = None

    def _remember(self, document: D, target_name: str, artifact: ArtifactRef) -> None:
        self._document = document
        self._target_name = target_name
        self._revision = artifact.require_exact().revision

    def _require_document(self) -> tuple[D, ArtifactRef]:
        if self._document is None or self._target_name is None or self._revision is None:
            raise ToolInputError(self.not_loaded_message, code=self.not_loaded_code)
        return self._document, ArtifactRef(
            namespace=self._namespace,
            name=self._target_name,
            revision=self._revision,
        )

    def _require_target(self) -> tuple[str, str]:
        if self._target_name is None or self._revision is None:
            raise RuntimeError("document session has no current target")
        return self._target_name, self._revision

    async def _write(
        self,
        document: D,
        *,
        target_name: str,
        expected_revision: str | None,
    ) -> Any:
        data = self._serialize(document)
        require_model_visible_binding(self._namespace, target_name)
        return await write_document_exact(
            self._client,
            ArtifactRef(namespace=self._namespace, name=target_name),
            data=data,
            media_type=self.media_type,
            expected_revision=expected_revision,
            max_bytes=self.max_bytes,
            reload_tool=self.reload_tool,
            read_tool=self.read_tool,
        )


def validate_target_name(value: str) -> None:
    # The namespace only satisfies the model; the name grammar is what is checked.
    ArtifactRef(namespace="documents", name=value)
