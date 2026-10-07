// Check types, materials and projects shaped like the bundled catalog
// (configs/audit-profiles), for the Start page tests.
import type { ArtifactMetadata } from "../../../api/artifacts";
import type { Audit, AuditProfile } from "../../../api/audits";
import type { Project } from "../../../api/projects";

const DIGEST = `sha256:${"2".repeat(64)}`;

type Inputs = AuditProfile["inputs"];

export function profileFixture(
  name: string,
  overrides: Partial<Omit<AuditProfile, "ref">> & { version?: string } = {},
): AuditProfile {
  const { version = "1", ...rest } = overrides;
  return {
    ref: { name, version, digest: DIGEST },
    mode: "risk-assessment",
    standards: [],
    inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
    inventory: {
      implementation: "standard-mappings@1",
      itemWorkflowRole: "check",
    },
    execution: {
      roundMode: "fixed-barrier",
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 10,
      maxItemsTotal: 10,
      maxSubmittedRuns: 30,
      maxItemRunAttempts: 3,
      deadlineSeconds: 86400,
      maxEvidenceBytes: 67_108_864,
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
    ...rest,
  };
}

const SOURCE: Inputs = {
  source: { required: true, mediaTypes: ["application/zip"] },
};
const OPENAPI_AND_SOURCE: Inputs = {
  source: { required: true, mediaTypes: ["application/zip"] },
  openapi: {
    required: true,
    mediaTypes: ["application/json", "application/yaml", "application/zip"],
  },
};

function limits(total: number, attempts = 3) {
  return {
    roundMode: "fixed-barrier" as const,
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: total,
    maxItemsTotal: total,
    maxSubmittedRuns: total * attempts,
    maxItemRunAttempts: attempts,
    deadlineSeconds: 86400,
    maxEvidenceBytes: 67_108_864,
    incompleteRound: "assess-with-gaps" as const,
  };
}

export const trace = profileFixture("openapi-operation-trace", {
  mode: "operation-tracing",
  inputs: OPENAPI_AND_SOURCE,
  inventory: {
    implementation: "openapi-operations@1",
    source: { source: "audit-input", name: "openapi" },
    itemWorkflowRole: "trace",
  },
  execution: limits(256),
});

export const observe = profileFixture("openapi-operation-observe", {
  mode: "operation-tracing",
  inputs: OPENAPI_AND_SOURCE,
  inventory: {
    implementation: "openapi-operations@1",
    source: { source: "audit-input", name: "openapi" },
    itemWorkflowRole: "trace",
  },
  execution: limits(256),
  interaction: {
    activeChecks: "prohibited",
    findingConfirmation: "disabled",
    notApplicable: "profile-rule",
    reportAcceptance: "automatic",
  },
});

export const top10 = profileFixture("owasp-top10-2025-source-risk", {
  standards: [{ scheme: "owasp-web-top10", version: "2025" }],
  inputs: SOURCE,
  execution: limits(10),
});

export const asvsReview = profileFixture("owasp-asvs-5-0-l1-source-review", {
  mode: "requirements-verification",
  standards: [{ scheme: "owasp-asvs", version: "5.0.0-l1-source.1" }],
  inputs: SOURCE,
  inventory: {
    implementation: "standard-mappings@1",
    itemWorkflowRole: "check",
    standardSelection: {
      scope:
        "ASVS 5.0 Level 1 source and documentation review (70 requirements)",
      levels: ["1"],
      entryIds: Array.from({ length: 70 }, (_, index) => `v5.0.0-${index}`),
    },
  },
  execution: limits(70),
});

export const asvsPilot = profileFixture("owasp-asvs-5-0-l1-source-pilot", {
  mode: "requirements-verification",
  standards: [{ scheme: "owasp-asvs", version: "5.0.0" }],
  inputs: SOURCE,
  inventory: {
    implementation: "standard-mappings@1",
    itemWorkflowRole: "check",
    standardSelection: {
      scope: "ASVS 5.0 Level 1 source and documentation pilot (5 requirements)",
      levels: ["1"],
      entryIds: ["a", "b", "c", "d", "e"],
    },
  },
  execution: limits(5, 2),
});

const LIVE_INTERACTION = {
  activeChecks: "approval-required",
  findingConfirmation: "human-required",
  notApplicable: "human-required",
  reportAcceptance: "automatic",
} as const;

function liveWorkflows() {
  return {
    check: {
      kind: "check" as const,
      workflow: { name: "audit-wstg-active-http", version: "1" },
      inputs: {},
      parameters: {
        target: { source: "scope-field" as const, name: "target" },
        authorization_scope: {
          source: "scope-field" as const,
          name: "authorizationScope",
        },
      },
      outputs: { result: "result" },
    },
  };
}

/** The list response leaves workflows out; the detail has them. */
export const wstgLive = profileFixture("owasp-wstg-4-2-active-http", {
  mode: "requirements-verification",
  standards: [{ scheme: "owasp-wstg", version: "4.2-http.1" }],
  inputs: {
    context: { required: true, mediaTypes: ["text/plain", "text/markdown"] },
  },
  execution: limits(94, 1),
  interaction: LIVE_INTERACTION,
});

export const wstgLiveDetail: AuditProfile = {
  ...wstgLive,
  workflows: liveWorkflows(),
};

export const nuclei = profileFixture("openapi-nuclei-scan", {
  inputs: {
    openapi: {
      required: true,
      mediaTypes: ["application/json", "application/yaml"],
    },
    settings: { required: true, mediaTypes: ["application/json"] },
  },
  inventory: {
    implementation: "openapi-scans@1",
    source: { source: "audit-input", name: "openapi" },
    settings: { source: "audit-input", name: "settings" },
    itemWorkflowRole: "scan",
  },
  execution: limits(16),
  interaction: {
    activeChecks: "approval-required",
    findingConfirmation: "disabled",
    notApplicable: "profile-rule",
    reportAcceptance: "automatic",
  },
});

export const checklist = profileFixture("source-checklist", {
  mode: "custom-checklist",
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
    checklist: {
      required: true,
      mediaTypes: ["application/json", "application/yaml"],
    },
  },
  inventory: {
    implementation: "checklist@1",
    source: { source: "audit-input", name: "checklist" },
    itemWorkflowRole: "check",
  },
  execution: limits(128),
});

export const unsupported = profileFixture("legacy-multi-round", {
  serverCompatible: false,
  compatibilityReasons: ["multiple_rounds_unsupported"],
});

export function materialFixture(
  name: string,
  mediaType: string,
  overrides: Partial<ArtifactMetadata> = {},
): ArtifactMetadata {
  return {
    artifact: { namespace: "sources", name, revision: `${name}-r1` },
    mediaType,
    size: 2048,
    current: true,
    frozen: false,
    createdAt: "2026-10-01T10:00:00Z",
    ...overrides,
  };
}

export const sourceZip = materialFixture("shop-source", "application/zip");
export const openapiYaml = materialFixture("shop-openapi", "application/yaml");

export function projectFixture(overrides: Partial<Project> = {}): Project {
  return {
    projectId: "project_shop",
    kind: "project",
    name: "Shop service",
    description: "Workshop API",
    lifecycle: "active",
    revision: "1",
    createdAt: "2026-10-01T10:00:00Z",
    updatedAt: "2026-10-01T10:00:00Z",
    ...overrides,
  };
}

export function draftAudit(
  profile: AuditProfile,
  overrides: Partial<Audit> = {},
): Audit {
  return {
    auditId: "audit_new",
    projectId: "project_shop",
    profile: { ...profile.ref },
    inputs: {},
    scope: {},
    runtimeLabels: [],
    state: "draft",
    phase: "not-started",
    revision: 1,
    dispatchState: "closed",
    holdState: "pending",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 10,
      maxItemsTotal: 10,
      maxSubmittedRuns: 30,
      maxItemRunAttempts: 3,
      maxEvidenceBytes: 67_108_864,
    },
    reservedRunCount: 0,
    submittedRunCount: 0,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 0,
    eventSequence: 1,
    createdAt: "2026-10-07T10:00:00Z",
    updatedAt: "2026-10-07T10:00:00Z",
    ...overrides,
  } as Audit;
}

/** The started check in its first round. */
export function startedAudit(draft: Audit): Audit {
  return {
    ...draft,
    state: "active",
    phase: "rounds",
    currentRoundId: "round_1",
    revision: draft.revision + 1,
    dispatchState: "open",
    holdState: "held",
  } as Audit;
}

/** POST /v1/audits/{auditId}/start response body. */
export function startResponse(draft: Audit) {
  const audit = startedAudit(draft);
  return {
    audit,
    round: {
      roundId: "round_1",
      ordinal: 1,
      state: "executing",
      expectedItemCount: 1,
      revision: 1,
      createdAt: audit.createdAt,
      updatedAt: audit.updatedAt,
    },
    items: [],
  };
}
