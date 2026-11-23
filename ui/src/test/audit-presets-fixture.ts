import type { AuditProfile } from "../api/audits";
import type { components } from "../api/generated/public";
import { conforming } from "./builders";
type AuditStandard = components["schemas"]["AuditStandardPackage"];

export const presetFixture: AuditProfile = conforming("AuditProfile", {
  ref: {
    name: "source-review",
    version: "1",
    digest: `sha256:${"1".repeat(64)}`,
  },
  inputs: { source: { required: true, media_types: ["application/zip"] } },
  limits: {
    max_tasks: 10,
    max_iterations: 1,
    max_tasks_per_iteration: 10,
    batch_size: 1,
    max_submitted_runs: 30,
    max_attempts: 3,
    deadline_seconds: 600,
    max_evidence_bytes: 1048576,
    max_hypotheses: 10,
    max_hypothesis_revisions: 10,
    max_hypothesis_bytes: 32768,
  },
  policy: { active_operations: "prohibited" },
  program: {
    schema: "contractor.audit.program.v1",
    inputs: { source: { required: true, media_types: ["application/zip"] } },
    workflows: {
      check: {
        ref: "source-check@3",
        inputs: { source: { source: "audit-input", name: "source" } },
        parameters: {},
        outputs: { result: "result" },
      },
    },
    hypotheses: {},
    initial_work: [
      { key: "inspect", workflow: "check", objective: "Inspect source" },
    ],
    limits: {
      max_tasks: 10,
      max_iterations: 1,
      max_tasks_per_iteration: 10,
      batch_size: 1,
      max_submitted_runs: 30,
      max_attempts: 3,
      deadline_seconds: 600,
      max_evidence_bytes: 1048576,
      max_hypotheses: 10,
      max_hypothesis_revisions: 10,
      max_hypothesis_bytes: 32768,
    },
    policy: { active_operations: "prohibited" },
  },
});
export const newerPresetFixture: AuditProfile = conforming("AuditProfile", {
  ...presetFixture,
  ref: {
    ...presetFixture.ref,
    version: "10",
    digest: `sha256:${"2".repeat(64)}`,
  },
});
export const dynamicPresetFixture: AuditProfile = conforming("AuditProfile", {
  ...presetFixture,
  ref: { ...presetFixture.ref, name: "openapi-operation-trace" },
  inputs: { openapi: { required: true, media_types: ["application/yaml"] } },
});

export const standardFixture: AuditStandard = conforming(
  "AuditStandardPackage",
  {
    reference: { scheme: "review-standard", version: "2026" },
    title: "Source review standard",
    description: "Requirements for a source review.",
    source: {
      name: "Review standard source",
      url: "https://example.test/standard",
    },
    license: {
      id: "CC-BY-4.0",
      url: "https://example.test/license",
      attribution: "Example review authors",
      disclosure: "full",
    },
    digest: `sha256:${"3".repeat(64)}`,
    artifact: {
      namespace: "audit-standards",
      name: "review",
      revision: "revision-1",
    },
    entryCount: 3,
    mappingCount: 3,
    evidenceContractCount: 1,
    entries: [
      {
        id: "REQ-1",
        kind: "requirement",
        title: "Authorization",
        statement: "Verify authorization before accessing a private record.",
        level: "1",
      },
      {
        id: "REQ-2",
        kind: "requirement",
        title: "Input validation",
        statement: "Reject invalid payloads at the trust boundary.",
        level: "1",
      },
      {
        id: "REQ-3",
        kind: "requirement",
        title: "Transport security",
        statement: "Verify transport encryption.",
        level: "1",
      },
    ],
    mappings: [
      {
        key: "REQ-1",
        entryIds: ["REQ-1"],
        workflowRole: "check",
        method: "source-analysis",
        evidenceContract: { id: "source-evidence", version: "1" },
        title: "Check authorization",
        objective: "Trace permission checks from the request handler.",
      },
      {
        key: "REQ-2",
        entryIds: ["REQ-2"],
        workflowRole: "check",
        method: "source-analysis",
        evidenceContract: { id: "source-evidence", version: "1" },
        title: "Check input validation",
        objective: "Inspect parsing and validation paths.",
      },
      {
        key: "REQ-3",
        entryIds: ["REQ-3"],
        workflowRole: "check",
        method: "configuration-review",
        evidenceContract: { id: "source-evidence", version: "1" },
        title: "Check transport security",
        objective: "Inspect TLS configuration.",
      },
    ],
    evidenceContracts: [
      {
        id: "source-evidence",
        version: "1",
        assessments: ["satisfied", "violated", "inconclusive"],
        evidenceKinds: ["artifact"],
        minimumEvidence: 1,
        maximumEvidence: 5,
        humanReview: "required",
        rationaleRequired: true,
      },
    ],
  },
);

/** The Server's answer to a read of `standard`. */
export function standardResponse(standard: AuditStandard = standardFixture) {
  return conforming("AuditStandardPackageResponse", {
    apiVersion: "contractor/v1alpha1",
    standard,
  });
}
