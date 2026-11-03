"""Trusted Audit preparation and invocation-local completion strategy."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from functools import partial

from contractor_runtime import threads
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    AUDIT_COMPLETION_TOOLS,
    ArtifactRef,
    StageContentRequest,
    WorkerCompletionContract,
    WorkerCompletionDiagnostics,
    WorkerFailure,
    WorkerObservations,
    WorkerResult,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.worker_completion import (
    CompleteCompletion,
    ContinueCompletion,
    FailCompletion,
    PreparedWorkerCompletion,
    WorkerCompletionError,
)
from contractor_runtime.digests import sha256_digest
from contractor_runtime.toolsets.audit_results.collector import (
    AuditCollectionError,
    InvocationAuditCollector,
)
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_DOCUMENT_BYTES,
    MAX_REMINDERS,
    AuditInvocationOwner,
    AuditTrustedInputs,
)
from contractor_runtime.toolsets.audit_results.encoding import (
    AuditResultError,
    CanonicalAuditResultEncoder,
)
from contractor_runtime.toolsets.audit_results.publication import (
    DeterministicAuditResultPublisher,
    PublicationArtifactClient,
)


class PreparedAuditCompletion(PreparedWorkerCompletion):
    """Allocation-owned exact assignment; only the active invocation owns drafts."""

    required_tools = AUDIT_COMPLETION_TOOLS

    def __init__(
        self,
        contract: WorkerCompletionContract,
        inputs: AuditTrustedInputs,
        encoder: CanonicalAuditResultEncoder,
        client: PublicationArtifactClient,
        request_timeout: float,
    ):
        self.contract = contract.model_copy(deep=True)
        self._inputs = inputs
        # Holds the inputs decoded once for every invocation and publication.
        self._encoder = encoder
        self._client = client
        self._request_timeout = request_timeout
        self._collector: InvocationAuditCollector | None = None
        self._reminders = 0
        self.diagnostics = None
        self.diagnostics_sink = None
        self.phase_sink = None

    def reset_diagnostics(self):
        self._reminders = 0
        self.diagnostics = WorkerCompletionDiagnostics(
            kind="audit-check-results@1",
            phase="collecting",
            acceptedCount=0,
            totalCount=len(self._inputs.owner.item_keys),
            reminderCount=0,
        )
        self._record_diagnostics()

    def _record_diagnostics(self):
        if self.diagnostics_sink is not None:
            self.diagnostics_sink(self.diagnostics)

    def record_progress(self, accepted_count):
        self.diagnostics = self.diagnostics.model_copy(update={"accepted_count": accepted_count})
        self._record_diagnostics()

    async def record_phase(self, phase, failure_code=None):
        value = WorkerCompletionDiagnostics.model_validate(
            {
                **self.diagnostics.model_dump(by_alias=True),
                "phase": phase,
                **({"failureCode": failure_code} if failure_code is not None else {}),
                "reminderCount": self._reminders,
            }
        )
        if value == self.diagnostics:
            return
        self.diagnostics = value
        self._record_diagnostics()
        if self.phase_sink is not None:
            await self.phase_sink(self.diagnostics)

    @classmethod
    async def prepare(cls, *, contract, allocation_id, client, timeout):
        if client.allocation_id != allocation_id:
            raise AuditResultError("audit_result_invalid")
        async with asyncio.timeout(timeout):
            context = await client.read_artifact(contract.context, max_bytes=MAX_DOCUMENT_BYTES)
        if (
            not isinstance(context, ArtifactValue)
            or context.artifact != contract.context
            or context.media_type != "application/json"
        ):
            raise AuditResultError("audit_result_invalid")
        # CPU-bound strict decoding runs once off the event loop. Cancellation
        # joins the work before allocation cleanup can release its resources.
        encoder = CanonicalAuditResultEncoder()
        records = await threads.to_thread_until_done(
            partial(encoder.context_records, context.data), name="audit-input-decode"
        )
        owner = AuditInvocationOwner(
            allocation_id,
            "audit-preparation",
            sha256_digest(context.data),
            records.item_keys,
        )
        inputs = AuditTrustedInputs(owner, context.data)
        return cls(contract, inputs, encoder, client, timeout)

    def current(self) -> InvocationAuditCollector:
        if self._collector is None:
            raise AuditCollectionError("invocation", "No active Audit invocation.")
        return self._collector

    def begin(self, invocation_id: str) -> None:
        if self._collector is not None:
            raise AuditResultError("audit_result_invalid")
        inputs = replace(
            self._inputs, owner=replace(self._inputs.owner, invocation_id=invocation_id)
        )
        self._collector = InvocationAuditCollector(inputs, self._encoder)
        self._reminders = 0
        self.reset_diagnostics()

    async def end(self) -> None:
        collector, self._collector = self._collector, None
        if collector is not None:
            # Keep invocation ownership until discard settles, including a
            # cancellation arriving during otherwise successful cleanup.
            cleanup = asyncio.create_task(collector.discard(), name="audit-collector-discard")
            interrupted = await join_owned_task(cleanup)
            cleanup.result()
            if interrupted:
                raise asyncio.CancelledError

    async def finish(
        self,
        *,
        request: StageContentRequest,
        deadline: float,
        check_active: Callable[[], None],
    ) -> ContinueCompletion | CompleteCompletion | FailCompletion:
        check_active()
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError
        collector = self.current()
        snapshot = await collector.snapshot()
        accepted = {item.value.item_key for item in snapshot.items}
        self.record_progress(len(accepted))
        missing = [key for key in collector.owner.item_keys if key not in accepted]
        if missing:
            if self._reminders >= MAX_REMINDERS:
                return FailCompletion(
                    WorkerFailure(
                        code="audit_result_incomplete",
                        message=(
                            f"Audit results incomplete: {len(accepted)}/"
                            f"{len(collector.owner.item_keys)} recorded."
                        ),
                        retryable=True,
                    )
                )
            self._reminders += 1
            await self.record_phase("collecting")
            return ContinueCompletion(
                "Runtime completion reminder: use read_audit_task and submit_check_result "
                "to record "
                "the missing assigned items: " + ", ".join(missing) + ". "
                "Use only verdicts and evidence permitted by each task; "
                "terminal prose does not record results."
            )
        publisher = DeterministicAuditResultPublisher(
            owner=collector.owner,
            result_artifact=self.contract.result_artifact,
            client=self._client,
            check_active=check_active,
            encoder=self._encoder,
        )
        sealed = await collector.seal()
        await self.record_phase("sealed")
        await self.record_phase("publishing")
        # Publication permits at most two writes and two read-backs. Each IO
        # receives its own outbound-request budget while the Stage deadline
        # and a finite publication cap remain authoritative. The Runtime's
        # outer Stage timeout stays separate so Stage expiry reports
        # worker_timeout even during an artifact request.
        publish_deadline = asyncio.get_running_loop().time() + 4 * self._request_timeout + 1
        receipt = await publisher.publish(
            sealed,
            inputs=collector.inputs,
            deadline=publish_deadline,
            request_timeout=self._request_timeout,
        )
        check_active()
        await self.record_phase("published")
        exact = ArtifactRef(
            namespace=receipt.namespace, name=receipt.name, revision=receipt.revision
        )
        # The binding is contract-owned; the model cannot contribute an artifact ref.
        artifacts = {
            slot: exact.model_copy(deep=True)
            for slot, binding in request.result_artifacts.items()
            if binding == self.contract.result_artifact
        }
        return CompleteCompletion(
            WorkerResult(
                subtaskId=request.subtask_id,
                result=f"Recorded and published results for {len(accepted)} assigned Audit items.",
                artifacts=artifacts,
                summarized=False,
                observations=WorkerObservations(profile="lean@1", tools={}, truncated=False),
            )
        )

    def validate_request(self, request: StageContentRequest) -> WorkerFailure | None:
        if self.contract.result_artifact not in request.result_artifacts.values():
            return WorkerFailure(
                code="invalid_worker_result_binding",
                message="Audit result binding is missing",
                retryable=False,
            )
        return None

    def failure(self, error: Exception) -> WorkerFailure:
        if isinstance(error, TimeoutError):
            return WorkerFailure(
                code="worker_timeout", message="Audit invocation deadline exceeded", retryable=True
            )
        if not isinstance(error, WorkerCompletionError):
            error = AuditResultError("audit_result_publication_failed", retryable=True)
        return WorkerFailure(
            code=error.code,
            message="Audit completion failed (" + error.code + ")",
            retryable=error.retryable,
        )
