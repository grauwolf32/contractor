"""Runtime-owned overlay persistence before graceful Worker results."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass

from contractor_runtime.artifacts import (
    ARTIFACT_HISTORY_MAINTENANCE_PENDING,
    ArtifactAPIError,
    ArtifactClient,
    ArtifactTransportError,
)
from contractor_runtime.contracts import (
    AllocationWorkspaceExport,
    ArtifactReadResult,
    ArtifactRef,
    WorkerResult,
)
from contractor_runtime.projectfs.overlay import (
    MAX_WORKSPACE_EXPORT_BYTES,
    WORKSPACE_OVERLAY_MEDIA_TYPE,
    OverlayWorkspaceSession,
    WorkspaceStateError,
)
from contractor_runtime.projectfs.storage import WorkspaceStorageError
from contractor_runtime.shared_config import artifact_payload_max_bytes
from contractor_runtime.threads import to_thread_until_done

WORKSPACE_DIFF_MEDIA_TYPE = "text/x-diff"
MAX_CAS_ATTEMPTS = 3
MAX_EXPORTED_RESULT_ARTIFACTS = 128
MAX_EXPORTED_RESULT_JSON_BYTES = 256 * 1024
SAFE_EXPORT_CAUSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class WorkspaceExportError(RuntimeError):
    """Stable, content-free failure at the pre-terminal export barrier."""

    code = "workspace_export_failed"

    def __init__(self, cause: str, *, retryable: bool) -> None:
        self.cause = cause if SAFE_EXPORT_CAUSE.fullmatch(cause) is not None else "export_error"
        self.retryable = retryable
        super().__init__(f"workspace export failed ({self.cause})")


@dataclass(frozen=True, slots=True)
class _ConfirmedRevision:
    """An exact export binding revision whose payload this exporter confirmed."""

    revision: str
    media_type: str
    size: int
    sha256: bytes


@dataclass(frozen=True, slots=True)
class WorkspaceExportResult:
    result: WorkerResult
    state_bytes: int
    diff_bytes: int
    result_workspace_digest: str


class WorkspaceAutoExporter:
    """Write one cumulative state/diff pair and then advance the checkpoint."""

    def __init__(
        self,
        *,
        workspace: OverlayWorkspaceSession,
        client: ArtifactClient,
        namespace: str,
        slots: AllocationWorkspaceExport,
    ) -> None:
        self._workspace = workspace
        self._client = client
        self._namespace = namespace
        self._state_slot = slots.state
        self._diff_slot = slots.diff
        self._reserved_slots = frozenset({slots.state, slots.diff})
        # At most one entry per reserved binding: the last exact revision whose
        # payload digest a write or an exact comparison confirmed (WSP-R178).
        self._confirmed: dict[str, _ConfirmedRevision] = {}

    @property
    def reserved_slots(self) -> frozenset[str]:
        return self._reserved_slots

    def release(self) -> None:
        """Forget confirmed revisions when the allocation releases its Worker."""

        self._confirmed.clear()

    async def export(self, result: WorkerResult) -> WorkspaceExportResult:
        if self._reserved_slots & result.artifacts.keys():
            raise WorkspaceExportError("reserved_result_slot", retryable=False)
        try:
            bundle = await self._workspace.prepare_export(
                max_payload_bytes=min(artifact_payload_max_bytes(), MAX_WORKSPACE_EXPORT_BYTES)
            )
            state_ref = await self._write_binding(
                self._state_slot,
                bundle.state,
                WORKSPACE_OVERLAY_MEDIA_TYPE,
            )
            diff_ref = await self._write_binding(
                self._diff_slot,
                bundle.diff,
                WORKSPACE_DIFF_MEDIA_TYPE,
            )
            final = self._inject_and_validate(result, state_ref, diff_ref)
            await self._workspace.commit_export(bundle)
        except WorkspaceExportError:
            raise
        except asyncio.CancelledError:
            raise
        except ArtifactAPIError as error:
            raise WorkspaceExportError(error.code, retryable=error.retryable) from None
        except ArtifactTransportError:
            raise WorkspaceExportError("artifact_transport", retryable=True) from None
        except WorkspaceStateError:
            raise WorkspaceExportError("workspace_state_invalid", retryable=False) from None
        except WorkspaceStorageError as error:
            cause = str(error.args[0]) if error.args else "workspace_unavailable"
            raise WorkspaceExportError(cause, retryable=False) from None
        except (TypeError, ValueError):
            raise WorkspaceExportError("invalid_export_result", retryable=False) from None
        except (OSError, TimeoutError):
            raise WorkspaceExportError("workspace_unavailable", retryable=True) from None
        except Exception:
            raise WorkspaceExportError("unexpected_export_failure", retryable=False) from None
        return WorkspaceExportResult(
            result=final,
            state_bytes=len(bundle.state),
            diff_bytes=len(bundle.diff),
            result_workspace_digest=bundle.result_workspace_digest,
        )

    async def _write_binding(self, name: str, data: bytes, media_type: str) -> ArtifactRef:
        target = ArtifactRef(namespace=self._namespace, name=name)
        sha256 = await to_thread_until_done(_sha256, data, name="workspace-export-digest")
        for attempt in range(MAX_CAS_ATTEMPTS):
            expected_revision: str | None
            try:
                current = await self._client.read_artifact_metadata(target)
            except ArtifactAPIError as error:
                if error.code != "artifact_not_found":
                    raise
                expected_revision = None
            else:
                exact = current.artifact.require_exact()
                if await self._holds(name, current, data, media_type, sha256):
                    self._confirm(name, exact, media_type, len(data), sha256)
                    return exact
                expected_revision = exact.revision
            try:
                written = await self._client.write_artifact(
                    target,
                    data=data,
                    media_type=media_type,
                    expected_revision=expected_revision,
                )
            except ArtifactAPIError as error:
                # A stale revision rereads; pending history maintenance stored
                # nothing and committed a pass, so the next attempt continues.
                if (
                    error.code not in ("artifact_conflict", ARTIFACT_HISTORY_MAINTENANCE_PENDING)
                    or attempt + 1 == MAX_CAS_ATTEMPTS
                ):
                    raise
            else:
                exact = written.artifact.require_exact()
                self._confirm(name, exact, media_type, len(data), sha256)
                return exact
        raise WorkspaceExportError("artifact_conflict", retryable=True)

    async def _holds(
        self,
        name: str,
        current: ArtifactReadResult,
        data: bytes,
        media_type: str,
        sha256: bytes,
    ) -> bool:
        """Return whether the observed exact revision already holds ``data``.

        A differing media type or size proves different content. Otherwise an
        exact revision this exporter confirmed is decided by its recorded
        digest, because revision content is immutable (ART-R49); any other
        revision is downloaded and compared byte for byte. The opaque revision
        itself is never taken as a content digest.
        """

        if current.media_type != media_type or current.size != len(data):
            return False
        exact = current.artifact
        confirmed = self._confirmed.get(name)
        if (
            confirmed is not None
            and confirmed.revision == exact.revision
            and confirmed.media_type == current.media_type
            and confirmed.size == current.size
        ):
            return confirmed.sha256 == sha256
        payload = await self._client.read_artifact(exact, max_bytes=max(1, current.size))
        return payload.media_type == media_type and payload.data == data

    def _confirm(
        self, name: str, exact: ArtifactRef, media_type: str, size: int, sha256: bytes
    ) -> None:
        revision = exact.revision
        if revision is None:
            raise RuntimeError("a confirmed export revision must be exact")
        self._confirmed[name] = _ConfirmedRevision(revision, media_type, size, sha256)

    def _inject_and_validate(
        self,
        result: WorkerResult,
        state_ref: ArtifactRef,
        diff_ref: ArtifactRef,
    ) -> WorkerResult:
        artifacts = dict(result.artifacts)
        artifacts[self._state_slot] = state_ref
        artifacts[self._diff_slot] = diff_ref
        if len(artifacts) > MAX_EXPORTED_RESULT_ARTIFACTS:
            raise WorkspaceExportError("invalid_export_result", retryable=False)
        final = WorkerResult.model_validate(
            {
                **result.model_dump(mode="python", by_alias=True, exclude_none=True),
                "artifacts": artifacts,
            }
        )
        if (
            len(final.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))
            > MAX_EXPORTED_RESULT_JSON_BYTES
        ):
            raise WorkspaceExportError("invalid_export_result", retryable=False)
        return final


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()
