import type { AuditStandard } from "../api/audit-presets";
import type { AuditProfile } from "../api/audits";

export const presetFixture: AuditProfile = {
  ref: {
    name: "source-review",
    version: "1",
    digest: `sha256:${"1".repeat(64)}`,
  },
  mode: "requirements-verification",
  standards: [{ scheme: "review-standard", version: "2026" }],
  inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
  inventory: {
    implementation: "standard-mappings@1",
    itemWorkflowRole: "check",
    standardSelection: {
      scope: "Selected source requirements",
      levels: ["1"],
      entryIds: ["REQ-1", "REQ-2"],
    },
  },
  workflows: {
    check: {
      kind: "check",
      workflow: { name: "source-check", version: "3" },
      inputs: {},
      parameters: {},
      outputs: {},
    },
  },
  execution: {
    roundMode: "fixed-barrier",
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: 10,
    maxItemsTotal: 10,
    maxSubmittedRuns: 30,
    maxItemRunAttempts: 3,
    deadlineSeconds: 600,
    maxEvidenceBytes: 1048576,
    incompleteRound: "assess-with-gaps",
  },
  interaction: {
    activeChecks: "prohibited",
    findingConfirmation: "human-required",
    notApplicable: "human-required",
    reportAcceptance: "automatic",
  },
  serverCompatible: true,
  requiresInputValidation: true,
  compatibilityReasons: [],
};

export const newerPresetFixture: AuditProfile = {
  ...presetFixture,
  ref: {
    ...presetFixture.ref,
    version: "10",
    digest: `sha256:${"2".repeat(64)}`,
  },
  inventory: {
    implementation: "standard-mappings@1",
    itemWorkflowRole: "check",
  },
};

export const dynamicPresetFixture: AuditProfile = {
  ...presetFixture,
  ref: { ...presetFixture.ref, name: "openapi-operation-trace" },
  mode: "operation-tracing",
  standards: [],
  inputs: { openapi: { required: true, mediaTypes: ["application/yaml"] } },
  inventory: {
    implementation: "openapi-operations@1",
    source: { source: "audit-input", name: "openapi" },
    itemWorkflowRole: "check",
  },
};

export const standardFixture: AuditStandard = {
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
};
