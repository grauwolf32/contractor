import type { components } from "./generated/public";

type Audit = components["schemas"]["Audit"];

// The public Audit schema bounds these collections, matching config's
// MaxAuditProfileWorkflows, MaxAuditWorkflowMappings and MaxAuditRunAttempts.
const MAX_PREPARE_ROLES = 16;
const MAX_ROLE_OUTPUTS = 128;
const MAX_RUN_ATTEMPTS = 10;
// AuditExactArtifact maximum retained descriptor size.
const MAX_ARTIFACT_BYTES = 64 * 1024 * 1024;
const NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const DIGEST = /^sha256:[0-9a-f]{64}$/;

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function keys(value: Record<string, unknown>, allowed: string[]): boolean {
  return Object.keys(value).every((key) => allowed.includes(key));
}

function name(value: unknown): value is string {
  return typeof value === "string" && NAME.test(value);
}

function validSource(value: unknown): boolean {
  if (
    !object(value) ||
    !keys(value, ["source", "name", "role"]) ||
    !name(value.name)
  )
    return false;
  return value.source === "audit-input"
    ? value.role === undefined
    : value.source === "prepare-output" && name(value.role);
}

export function validInventory(value: unknown): boolean {
  if (
    !object(value) ||
    !keys(value, [
      "implementation",
      "source",
      "settings",
      "itemWorkflowRole",
      "standardSelection",
    ]) ||
    !name(value.itemWorkflowRole)
  )
    return false;
  if (value.implementation === "standard-mappings@1")
    return value.source === undefined && value.settings === undefined;
  if (
    ![
      "checklist@1",
      "openapi-operations@1",
      "openapi-scans@1",
      "finding-candidates@1",
    ].includes(String(value.implementation)) ||
    !validSource(value.source) ||
    value.standardSelection !== undefined
  )
    return false;
  return value.implementation === "openapi-scans@1"
    ? validSource(value.settings)
    : value.settings === undefined;
}

function validOutput(
  value: unknown,
  executionId: unknown,
  runId: unknown,
): boolean {
  if (
    !object(value) ||
    !keys(value, ["artifact", "executionId", "runId", "workflowOutput"]) ||
    value.executionId !== executionId ||
    value.runId !== runId ||
    !name(value.workflowOutput)
  )
    return false;
  const artifact = value.artifact;
  if (
    !object(artifact) ||
    !keys(artifact, ["ref", "digest", "mediaType", "sizeBytes"]) ||
    typeof artifact.digest !== "string" ||
    !DIGEST.test(artifact.digest) ||
    typeof artifact.mediaType !== "string" ||
    !Number.isSafeInteger(artifact.sizeBytes) ||
    Number(artifact.sizeBytes) < 0 ||
    Number(artifact.sizeBytes) > MAX_ARTIFACT_BYTES
  )
    return false;
  return (
    object(artifact.ref) &&
    keys(artifact.ref, ["namespace", "name", "revision"]) &&
    name(artifact.ref.namespace) &&
    name(artifact.ref.name) &&
    typeof artifact.ref.revision === "string" &&
    artifact.ref.revision.length > 0
  );
}

function validRole(value: unknown): boolean {
  if (
    !object(value) ||
    !keys(value, [
      "status",
      "attempts",
      "maxAttempts",
      "executionId",
      "runId",
      "outputs",
    ]) ||
    !Number.isSafeInteger(value.attempts) ||
    !Number.isSafeInteger(value.maxAttempts)
  )
    return false;
  const attempts = Number(value.attempts);
  const limit = Number(value.maxAttempts);
  if (
    limit < 1 ||
    limit > MAX_RUN_ATTEMPTS ||
    attempts < 0 ||
    attempts > limit ||
    !object(value.outputs) ||
    Object.keys(value.outputs).length > MAX_ROLE_OUTPUTS
  )
    return false;
  if (value.status === "pending")
    return (
      attempts === 0 &&
      value.executionId === undefined &&
      value.runId === undefined &&
      Object.keys(value.outputs).length === 0
    );
  if (
    attempts === 0 ||
    !name(value.executionId) ||
    (value.runId !== undefined && !name(value.runId))
  )
    return false;
  if (value.status === "accepted")
    return (
      name(value.runId) &&
      Object.keys(value.outputs).length > 0 &&
      Object.entries(value.outputs).every(
        ([key, output]) =>
          name(key) && validOutput(output, value.executionId, value.runId),
      )
    );
  return (
    ["running", "failed"].includes(String(value.status)) &&
    Object.keys(value.outputs).length === 0
  );
}

export function validAuditComposition(audit: Audit): boolean {
  if (
    !["not-started", "preparing", "inventory", "rounds"].includes(audit.phase)
  )
    return false;
  if (audit.phase === "not-started")
    return (
      audit.currentRoundId === undefined && audit.preparation === undefined
    );
  if (
    audit.phase === "rounds"
      ? !name(audit.currentRoundId)
      : audit.currentRoundId !== undefined ||
        audit.baseline === undefined ||
        audit.preparation === undefined
  )
    return false;
  if (audit.preparation === undefined) return true;
  return validPreparation(audit.preparation, audit.phase !== "preparing");
}

export function validPreparation(
  value: unknown,
  requireAccepted = false,
): boolean {
  if (!object(value) || !keys(value, ["roles"]) || !object(value.roles))
    return false;
  const roles = Object.entries(value.roles);
  return (
    roles.length > 0 &&
    roles.length <= MAX_PREPARE_ROLES &&
    roles.every(
      ([key, role]) =>
        name(key) &&
        validRole(role) &&
        (!requireAccepted || (object(role) && role.status === "accepted")),
    )
  );
}
