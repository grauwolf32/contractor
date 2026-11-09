// Isolated common-contract profiles and materials for generic Start page tests.
import type { ArtifactMetadata } from "../../../api/artifacts";
import type { Audit, AuditProfile } from "../../../api/audits";
import type { Project } from "../../../api/projects";
import * as build from "../../../test/builders";

const DIGEST = `sha256:${"2".repeat(64)}`;

type Inputs = AuditProfile["inputs"];

const SOURCE: Inputs = {
  source: { required: true, media_types: ["application/zip"] },
};
const OPENAPI_AND_SOURCE: Inputs = {
  ...SOURCE,
  openapi: {
    required: true,
    media_types: ["application/json", "application/yaml", "application/zip"],
  },
};
function limits(total: number, attempts = 3) {
  return {
    max_iterations: 1,
    batch_size: 1,
    max_tasks_per_iteration: total,
    max_tasks: total,
    max_submitted_runs: total * attempts,
    max_attempts: attempts,
    deadline_seconds: 86400,
    max_evidence_bytes: 67108864,
    max_hypotheses: 128,
    max_hypothesis_revisions: 128,
    max_hypothesis_bytes: 32768,
  };
}
export function profileFixture(
  name: string,
  overrides: Partial<Omit<AuditProfile, "ref">> & { version?: string } = {},
): AuditProfile {
  const { version = "1", ...rest } = overrides;
  return build.conforming("AuditProfile", {
    ref: { name, version, digest: DIGEST },
    inputs: SOURCE,
    limits: limits(10),
    policy: { active_operations: "prohibited" },
    ...rest,
  });
}
export const trace = profileFixture("openapi-operation-trace", {
  inputs: OPENAPI_AND_SOURCE,
  limits: limits(256),
});
export const observe = profileFixture("openapi-operation-observe", {
  inputs: OPENAPI_AND_SOURCE,
  limits: limits(256),
});
export const top10 = profileFixture("source-review", { limits: limits(10) });
export const asvsReview = profileFixture("broad-review", {
  limits: limits(70),
});
export const asvsPilot = profileFixture("small-review", {
  limits: limits(5, 2),
});
export const wstgLive = profileFixture("http-observe", {
  inputs: {
    context: { required: true, media_types: ["text/plain", "text/markdown"] },
  },
  limits: limits(94, 1),
  policy: { active_operations: "allowed" },
});
export const wstgLiveDetail: AuditProfile = wstgLive;
export const nuclei = profileFixture("openapi-nuclei-scan", {
  inputs: {
    openapi: {
      required: true,
      media_types: ["application/json", "application/yaml"],
    },
    request_values: { required: true, media_types: ["application/json"] },
  },
  limits: limits(16),
  policy: { active_operations: "allowed" },
});
export const checklist = profileFixture("selected-materials", {
  inputs: {
    ...SOURCE,
    checklist: {
      required: true,
      media_types: ["application/json", "application/yaml"],
    },
  },
  limits: limits(128),
});
export const unsupported = profileFixture("custom-inspection");
export const webScanProfile = withExecutionParameters(
  profileFixture("web-scan", {
    inputs: {
      target: { required: true, media_types: ["application/json"] },
      engagement: {
        required: true,
        media_types: ["application/vnd.contractor.engagement+json"],
      },
      rulepack: {
        required: false,
        media_types: [
          "application/yaml",
          "application/vnd.contractor.nuclei-rulepack+zip",
        ],
      },
    },
    policy: { active_operations: "allowed" },
  }),
  { credential_id: { required: false } },
);

export function materialFixture(
  name: string,
  mediaType: string,
  overrides: Partial<ArtifactMetadata> = {},
): ArtifactMetadata {
  return build.conforming("ArtifactMetadata", {
    canUploadNewVersion: true,
    storageTimestamps: build.ARTIFACT_STORAGE_TIMESTAMPS,
    artifact: { namespace: "sources", name, revision: `${name}-r1` },
    mediaType,
    size: 2048,
    current: true,
    frozen: false,
    createdAt: "2026-10-01T10:00:00Z",
    ...overrides,
  });
}

export const sourceZip = materialFixture("shop-source", "application/zip");
export const openapiYaml = materialFixture("shop-openapi", "application/yaml");

export function projectFixture(overrides: Partial<Project> = {}): Project {
  return build.project("project_shop", {
    name: "Shop service",
    description: "Workshop API",
    createdAt: "2026-10-01T10:00:00Z",
    updatedAt: "2026-10-01T10:00:00Z",
    ...overrides,
  });
}

/** A draft of `profile` with one pinned input per required profile input. */
export function draftAudit(
  profile: AuditProfile,
  overrides: Partial<Audit> = {},
): Audit {
  return build.audit("project_shop", "audit_new", "draft", {
    profile: { ...profile.ref },
    inputs: Object.fromEntries(
      Object.entries(profile.inputs).map(([name, input]) => [
        name,
        build.auditInput(`shop-${name}`, { mediaType: input.media_types[0] }),
      ]),
    ),
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 10,
      maxItemsTotal: 10,
      maxSubmittedRuns: 30,
      maxItemRunAttempts: 3,
      maxEvidenceBytes: 67_108_864,
    },
    createdAt: "2026-10-07T10:00:00Z",
    updatedAt: "2026-10-07T10:00:00Z",
    ...overrides,
  });
}

/** The started check in its first round. */
export function startedAudit(draft: Audit): Audit {
  return build.conforming("Audit", {
    ...draft,
    state: "active",
    taskCount: 0,
    iteration: 0,
    revision: draft.revision + 1,
    dispatchState: "open",
    holdState: "held",
  } as Audit);
}

/** POST /v1/audits/{auditId}/start response body. */
export function startResponse(draft: Audit) {
  const audit = startedAudit(draft);
  return build.conforming("AuditStartResponse", { audit });
}

/** A current common assertion program, independent of the shipped catalog. */
export function assertionProfileFixture(): AuditProfile {
  const profile = profileFixture("source-checklist", {
    inputs: checklist.inputs,
    limits: limits(128),
  });
  return build.conforming("AuditProfile", {
    ...profile,
    program: {
      schema: "contractor.audit.program.v1",
      inputs: profile.inputs,
      workflows: {
        check: {
          ref: "source-check@1",
          inputs: {
            source: { source: "audit-input", name: "source" },
            context: { source: "task-context" },
          },
          parameters: {},
          outputs: { result: "result" },
        },
      },
      hypotheses: {},
      initial_work: [],
      seed_source: { workflow: "check", input: "checklist" },
      limits: profile.limits,
      policy: profile.policy,
    },
  });
}

export const engagementMaterial = materialFixture(
  "engagement",
  "application/vnd.contractor.engagement+json",
);

export function withExecutionParameters(
  profile: AuditProfile,
  parameters: Record<string, { required: boolean }>,
): AuditProfile {
  return build.conforming("AuditProfile", {
    ...profile,
    program: {
      schema: "contractor.audit.program.v1",
      inputs: profile.inputs,
      parameters,
      workflows: {
        scan: {
          ref: "fixture-scan@1",
          inputs: Object.fromEntries(
            Object.keys(profile.inputs).map((name) => [
              name,
              { source: "audit-input", name },
            ]),
          ),
          parameters: {},
          parameter_inputs: Object.fromEntries(
            Object.keys(parameters).map((name) => [name, name]),
          ),
          outputs: {},
        },
      },
      hypotheses: {},
      initial_work: [
        { key: "scan", workflow: "scan", objective: "Run fixture scan" },
      ],
      limits: profile.limits,
      policy: profile.policy,
    },
  });
}
