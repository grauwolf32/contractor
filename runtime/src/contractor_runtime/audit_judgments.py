"""Durable normalized judgments through the allocation-bound Audit owner."""

import json
from typing import ClassVar, Literal
from uuid import uuid4

import jcs
from jsonschema.exceptions import ValidationError as SchemaValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactTransportError,
    allocation_path,
    raise_for_status,
    require_json,
)
from contractor_runtime.contracts import API_VERSION
from contractor_runtime.contracts.audit_documents import validate_audit_material
from contractor_runtime.contracts.base import MAX_EXACT_INTEGER, RESOURCE_ID_PATTERN
from contractor_runtime.core.tool_errors import ToolResultDiagnostic
from contractor_runtime.strict_json import strict_json_loads


class JudgmentGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    itemKey: str
    taskId: str
    generation: int = Field(ge=0, le=MAX_EXACT_INTEGER)


class JudgmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    apiVersion: Literal["contractor/v1alpha1"]
    material: dict
    generations: list[JudgmentGeneration]
    replayed: bool


class JudgmentConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    apiVersion: Literal["contractor/v1alpha1"]
    code: Literal["audit_result_stale"]
    message: str
    retryable: Literal[False]
    requestId: str
    currentGenerations: list[JudgmentGeneration]


class JudgmentStaleError(Exception):
    def __init__(self, generations):
        super().__init__("Audit result changed before acceptance")
        self.generations = generations


class JudgmentPersistenceUnknown(ArtifactTransportError, ToolResultDiagnostic):
    code = "audit_result_persistence_unknown"
    retryable = False
    diagnostic_message = (
        "Judgment acceptance is unconfirmed after bounded receipt reconciliation. "
        "Earlier accepted results remain saved; do not repeat target traffic."
    )
    model_details: ClassVar[dict] = {"persistence": "unknown"}


class AuditJudgmentClient:
    def __init__(self, client, encoder, inputs):
        self.client, self.encoder, self.inputs = client, encoder, inputs
        self.audit_id = encoder.trusted_inputs(inputs).audit_id

    async def __call__(self, snapshot, bases, mode):
        def prepare():
            document = strict_json_loads(self.encoder.encode(snapshot, inputs=self.inputs).data)
            return jcs.canonicalize(
                {
                    "apiVersion": API_VERSION,
                    "invocationId": snapshot.owner.invocation_id,
                    "submissionId": uuid4().hex,
                    "mode": mode,
                    "items": [
                        {"baseGeneration": bases[item["item_key"]] or 0, "conclusion": item}
                        for item in document["items"]
                    ],
                }
            )

        body = await threads.to_thread_until_done(prepare, name="audit-judgment-request")
        keys = tuple(item.value.item_key for item in snapshot.items)
        try:
            return await self._submit(body, keys)
        except ArtifactTransportError:
            raise JudgmentPersistenceUnknown() from None

    async def _submit(self, body, keys):
        # Both technical sends retain the exact body, base and mutation identity.
        # A new target request or new semantic generation is never recovery.
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("Audit judgment request exceeds its size limit")
        for attempt in range(2):
            try:
                response = await self.client.transport.request(
                    "POST",
                    allocation_path(self.client.allocation_id, "audit-results"),
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    body=body,
                    max_response_bytes=1 << 20,
                )
            except ArtifactTransportError:
                if attempt:
                    raise
                continue
            break
        require_json(response.headers)
        try:
            value = strict_json_loads(response.body)
            if not isinstance(value, dict):
                raise ValueError("Expected object")
        except ValueError:
            raise ArtifactTransportError("Audit result API returned an invalid response") from None
        if response.status_code == 409 and value.get("code") == "audit_result_stale":
            try:
                conflict = JudgmentConflict.model_validate(value)
                generations = self._generations(conflict.currentGenerations, keys)
            except (ValidationError, ValueError, TypeError):
                raise ArtifactTransportError(
                    "Audit result API returned invalid generations"
                ) from None
            raise JudgmentStaleError(generations)
        raise_for_status(response)
        if response.status_code not in {200, 201}:
            raise ArtifactTransportError("Audit result API returned an invalid status")
        try:
            receipt = JudgmentReceipt.model_validate(value)
            generations = self._generations(receipt.generations, keys)
            if any(value < 1 for value in generations.values()):
                raise ValueError("Nonpositive accepted generation")
            await threads.to_thread_until_done(
                validate_audit_material, receipt.material, name="audit-judgment-receipt"
            )
            if (
                receipt.material["scope_kind"] != "audit"
                or receipt.material["scope_id"] != self.audit_id
                or receipt.material["media_type"] != "application/json"
            ):
                raise ValueError("Another Audit or media type")
        except (ValidationError, ValueError, TypeError, KeyError, SchemaValidationError):
            raise ArtifactTransportError("Audit result API returned an invalid receipt") from None
        return {
            "generations": generations,
            "material": json.loads(receipt.model_dump_json())["material"],
        }

    @staticmethod
    def _generations(values, keys):
        generations = {value.itemKey: value.generation for value in values}
        if len(values) != len(generations) or set(generations) != set(keys):
            raise ValueError("Another judgment membership")
        if any(RESOURCE_ID_PATTERN.fullmatch(value.taskId) is None for value in values):
            raise ValueError("Invalid Task identity")
        return generations
