import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";
import { requireProjectID } from "./projects";
import { CONFIG_ID_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

export const AUDIT_PAGE_SIZE = 50;
export const AUDIT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
export const AUDIT_REVISION_PATTERN = /^[1-9][0-9]{0,18}$/;

export type AuditProfile = components["schemas"]["AuditProfile"];
export type AuditProfilePage = components["schemas"]["AuditProfilePage"];
export type Audit = components["schemas"]["Audit"];
export type AuditPage = components["schemas"]["AuditPage"];
export type AuditState = components["schemas"]["AuditState"];
export type AuditItem = components["schemas"]["AuditItem"];
export type AuditItemPage = components["schemas"]["AuditItemPage"];
export type AuditCoverageRow = components["schemas"]["AuditCoverageRow"];
export type AuditCoveragePage = components["schemas"]["AuditCoveragePage"];
export type AuditReport = components["schemas"]["AuditReport"];
export type AuditStartResponse = components["schemas"]["AuditStartResponse"];
export type CreateAuditRequest = components["schemas"]["CreateAuditRequest"];
export type ExactArtifactRef = components["schemas"]["ExactArtifactRef"];
export type AuditFinding = components["schemas"]["AuditFinding"];
export type AuditFindingPage = components["schemas"]["AuditFindingPage"];
export type AuditFindingState = components["schemas"]["AuditFindingState"];
export type AuditFindingSeverity =
  components["schemas"]["AuditFindingSeverity"];
export type AuditAnalystVerdict = components["schemas"]["AuditAnalystVerdict"];
export type AuditReviewRequest = components["schemas"]["AuditReviewRequest"];
export type AuditReviewPage = components["schemas"]["AuditReviewPage"];
export type AuditReviewAction = components["schemas"]["AuditReviewAction"];
export type AuditActionDecisionResult =
  components["schemas"]["AuditActionDecisionResult"];
export type AuditFindingProvenancePage =
  components["schemas"]["AuditFindingProvenancePage"];
export type DecideAuditFindingRequest =
  components["schemas"]["DecideAuditFindingRequest"];

export interface PageRequest {
  cursor?: string;
}

export interface ProjectAuditPageRequest extends PageRequest {
  projectId: string;
  state?: AuditState;
  profile?: string;
}

export interface AuditCreateOptions {
  projectId: string;
  request: CreateAuditRequest;
  idempotencyKey: string;
}

export type AuditMutationAction =
  "start" | "pause" | "resume" | "cancel" | "delete";

export interface AuditMutationOptions {
  auditId: string;
  expectedRevision: number;
  idempotencyKey: string;
}

function requireData<T>(result: {
  data?: T;
  error?: unknown;
  response: Response;
}): T {
  if (result.data === undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
  return result.data;
}

function invalidAuditResponse(status: number): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message: "Server returned an invalid Audit response",
  });
}

function requireProfileIdentity(name: string, version: string): void {
  if (!CONFIG_ID_PATTERN.test(name) || !CONFIG_VERSION_PATTERN.test(version)) {
    throw new TypeError("AuditProfile identity is invalid");
  }
}

function requireAuditID(auditId: string): void {
  if (!AUDIT_ID_PATTERN.test(auditId)) {
    throw new TypeError("Audit ID is invalid");
  }
}

function requireAuditRevision(revision: number): void {
  if (!Number.isSafeInteger(revision) || revision < 1) {
    throw new TypeError("Audit revision is invalid");
  }
}

function safeProfile(profile: AuditProfile, status: number): AuditProfile {
  if (
    profile === null ||
    typeof profile !== "object" ||
    !CONFIG_ID_PATTERN.test(profile.ref.name) ||
    !CONFIG_VERSION_PATTERN.test(profile.ref.version) ||
    typeof profile.ref.digest !== "string" ||
    !profile.ref.digest.startsWith("sha256:") ||
    profile.inputs === null ||
    typeof profile.inputs !== "object" ||
    !Array.isArray(profile.standards) ||
    !Array.isArray(profile.compatibilityReasons) ||
    typeof profile.serverCompatible !== "boolean" ||
    typeof profile.requiresInputValidation !== "boolean"
  ) {
    throw invalidAuditResponse(status);
  }
  return {
    ref: { ...profile.ref },
    mode: profile.mode,
    standards: profile.standards.map((standard) => ({ ...standard })),
    inputs: Object.fromEntries(
      Object.entries(profile.inputs).map(([name, input]) => [
        name,
        { ...input, mediaTypes: [...input.mediaTypes] },
      ]),
    ),
    inventory: { ...profile.inventory },
    ...(profile.workflows === undefined
      ? {}
      : {
          workflows: Object.fromEntries(
            Object.entries(profile.workflows).map(([role, workflow]) => [
              role,
              {
                ...workflow,
                workflow: { ...workflow.workflow },
                inputs: Object.fromEntries(
                  Object.entries(workflow.inputs).map(([name, value]) => [
                    name,
                    { ...value },
                  ]),
                ),
                parameters: Object.fromEntries(
                  Object.entries(workflow.parameters).map(([name, value]) => [
                    name,
                    { ...value },
                  ]),
                ),
                outputs: { ...workflow.outputs },
              },
            ]),
          ),
        }),
    execution: { ...profile.execution },
    interaction: { ...profile.interaction },
    serverCompatible: profile.serverCompatible,
    requiresInputValidation: profile.requiresInputValidation,
    compatibilityReasons: [...profile.compatibilityReasons],
  };
}

function safeAudit(audit: Audit, status: number): Audit {
  if (
    audit === null ||
    typeof audit !== "object" ||
    !AUDIT_ID_PATTERN.test(audit.auditId) ||
    !AUDIT_ID_PATTERN.test(audit.projectId) ||
    !Number.isSafeInteger(audit.revision) ||
    audit.revision < 1 ||
    typeof audit.state !== "string" ||
    !Array.isArray(audit.runtimeLabels) ||
    audit.inputs === null ||
    typeof audit.inputs !== "object" ||
    typeof audit.createdAt !== "string" ||
    typeof audit.updatedAt !== "string"
  ) {
    throw invalidAuditResponse(status);
  }
  // Audit endpoints are deliberately safe projections. structuredClone keeps
  // query state detached without requesting or interpreting package bytes.
  return structuredClone(audit);
}

function requireRevisionETag(response: Response, revision: number): void {
  if (response.headers.get("ETag") !== `"${revision}"`) {
    throw invalidAuditResponse(response.status);
  }
}

function requireProfileETag(response: Response, digest: string): void {
  if (response.headers.get("ETag") !== `"${digest}"`) {
    throw invalidAuditResponse(response.status);
  }
}

function safePage<T>(
  value: { items: T[]; page: components["schemas"]["PageInfo"] },
  status: number,
): { items: T[]; page: components["schemas"]["PageInfo"] } {
  if (!Array.isArray(value.items) || value.page === undefined) {
    throw invalidAuditResponse(status);
  }
  return { items: [...value.items], page: { ...value.page } };
}

export async function listAuditProfiles(
  api: PublicAPI,
  request: PageRequest = {},
): Promise<AuditProfilePage> {
  const result = await api.request((client) =>
    client.GET("/v1/audit-profiles", {
      params: {
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = safePage(requireData(result), result.response.status);
  return {
    items: page.items.map((profile) =>
      safeProfile(profile, result.response.status),
    ),
    page: page.page,
  };
}

export async function getAuditProfile(
  api: PublicAPI,
  name: string,
  version: string,
): Promise<AuditProfile> {
  requireProfileIdentity(name, version);
  const result = await api.request((client) =>
    client.GET("/v1/audit-profiles/{name}/versions/{version}", {
      params: { path: { name, version } },
    }),
  );
  const profile = safeProfile(requireData(result), result.response.status);
  if (profile.ref.name !== name || profile.ref.version !== version) {
    throw invalidAuditResponse(result.response.status);
  }
  requireProfileETag(result.response, profile.ref.digest);
  return profile;
}

export async function createAudit(
  api: PublicAPI,
  options: AuditCreateOptions,
): Promise<Audit> {
  requireProjectID(options.projectId);
  requireProfileIdentity(
    options.request.profile.name,
    options.request.profile.version,
  );
  const result = await api.request((client) =>
    client.POST("/v1/projects/{projectId}/audits", {
      params: {
        path: { projectId: options.projectId },
        header: { "Idempotency-Key": options.idempotencyKey },
      },
      body: options.request,
    }),
  );
  const audit = safeAudit(requireData(result), result.response.status);
  if (audit.projectId !== options.projectId || audit.state !== "draft") {
    throw invalidAuditResponse(result.response.status);
  }
  requireRevisionETag(result.response, audit.revision);
  return audit;
}

export async function listProjectAudits(
  api: PublicAPI,
  request: ProjectAuditPageRequest,
): Promise<AuditPage> {
  requireProjectID(request.projectId);
  const result = await api.request((client) =>
    client.GET("/v1/projects/{projectId}/audits", {
      params: {
        path: { projectId: request.projectId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
          ...(request.state === undefined ? {} : { state: request.state }),
          ...(request.profile === undefined
            ? {}
            : { profile: request.profile }),
        },
      },
    }),
  );
  const page = safePage(requireData(result), result.response.status);
  return {
    items: page.items.map((audit) => safeAudit(audit, result.response.status)),
    page: page.page,
  };
}

export async function getAudit(
  api: PublicAPI,
  auditId: string,
): Promise<Audit> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}", {
      params: { path: { auditId } },
    }),
  );
  const audit = safeAudit(requireData(result), result.response.status);
  if (audit.auditId !== auditId) {
    throw invalidAuditResponse(result.response.status);
  }
  requireRevisionETag(result.response, audit.revision);
  return audit;
}

export async function mutateAudit(
  api: PublicAPI,
  action: AuditMutationAction,
  options: AuditMutationOptions,
): Promise<Audit | AuditStartResponse> {
  requireAuditID(options.auditId);
  requireAuditRevision(options.expectedRevision);
  const common = {
    params: {
      path: { auditId: options.auditId },
      header: {
        "Idempotency-Key": options.idempotencyKey,
        "If-Match": `"${options.expectedRevision}"`,
      },
    },
  } as const;
  if (action === "start") {
    const result = await api.request((client) =>
      client.POST("/v1/audits/{auditId}/start", common),
    );
    const started = requireData(result);
    const audit = safeAudit(started.audit, result.response.status);
    if (
      audit.auditId !== options.auditId ||
      !Array.isArray(started.items) ||
      started.round === undefined
    ) {
      throw invalidAuditResponse(result.response.status);
    }
    requireRevisionETag(result.response, audit.revision);
    return {
      audit,
      round: structuredClone(started.round),
      items: structuredClone(started.items),
    };
  }
  if (action === "pause") {
    const result = await api.request((client) =>
      client.POST("/v1/audits/{auditId}/pause", common),
    );
    return finishAuditMutation(result, options.auditId);
  }
  if (action === "resume") {
    const result = await api.request((client) =>
      client.POST("/v1/audits/{auditId}/resume", common),
    );
    return finishAuditMutation(result, options.auditId);
  }
  if (action === "cancel") {
    const result = await api.request((client) =>
      client.POST("/v1/audits/{auditId}/cancel", common),
    );
    return finishAuditMutation(result, options.auditId);
  }
  const result = await api.request((client) =>
    client.DELETE("/v1/audits/{auditId}", common),
  );
  return finishAuditMutation(result, options.auditId);
}

function finishAuditMutation(
  result: { data?: Audit; error?: unknown; response: Response },
  expectedAuditId: string,
): Audit {
  const audit = safeAudit(requireData(result), result.response.status);
  if (audit.auditId !== expectedAuditId) {
    throw invalidAuditResponse(result.response.status);
  }
  requireRevisionETag(result.response, audit.revision);
  return audit;
}

export async function listAuditItems(
  api: PublicAPI,
  auditId: string,
  request: PageRequest = {},
): Promise<AuditItemPage> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/items", {
      params: {
        path: { auditId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = safePage(requireData(result), result.response.status);
  return { items: structuredClone(page.items), page: page.page };
}

export async function listAuditCoverage(
  api: PublicAPI,
  auditId: string,
  request: PageRequest = {},
): Promise<AuditCoveragePage> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/coverage", {
      params: {
        path: { auditId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = safePage(requireData(result), result.response.status);
  return { items: structuredClone(page.items), page: page.page };
}

export async function getAuditReport(
  api: PublicAPI,
  auditId: string,
): Promise<AuditReport> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/report", {
      params: { path: { auditId } },
    }),
  );
  const report = requireData(result);
  if (
    report === null ||
    typeof report !== "object" ||
    (report.status !== "pending" &&
      report.status !== "proposed" &&
      report.status !== "ready" &&
      report.status !== "unavailable")
  ) {
    throw invalidAuditResponse(result.response.status);
  }
  return structuredClone(report);
}

export async function listAuditFindings(
  api: PublicAPI,
  auditId: string,
  request: PageRequest = {},
): Promise<AuditFindingPage> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/findings", {
      params: {
        path: { auditId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = safePage(requireData(result), result.response.status);
  return { items: structuredClone(page.items), page: page.page };
}

export async function getAuditFinding(
  api: PublicAPI,
  auditId: string,
  findingId: string,
): Promise<AuditFinding> {
  requireAuditID(auditId);
  requireAuditID(findingId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/findings/{findingId}", {
      params: { path: { auditId, findingId } },
    }),
  );
  const finding = requireData(result);
  if (finding.auditId !== auditId || finding.findingId !== findingId) {
    throw invalidAuditResponse(result.response.status);
  }
  requireRevisionETag(result.response, finding.revision);
  return structuredClone(finding);
}

export async function listAuditReviews(
  api: PublicAPI,
  auditId: string,
  request: PageRequest & { finding?: string } = {},
): Promise<AuditReviewPage> {
  requireAuditID(auditId);
  if (request.finding !== undefined) requireAuditID(request.finding);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/reviews", {
      params: {
        path: { auditId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
          ...(request.finding === undefined
            ? {}
            : { finding: request.finding }),
        },
      },
    }),
  );
  const page = safePage(requireData(result), result.response.status);
  return { items: structuredClone(page.items), page: page.page };
}

export async function createAuditFindingReview(
  api: PublicAPI,
  options: {
    auditId: string;
    findingId: string;
    expectedRevision: number;
    idempotencyKey: string;
  },
): Promise<AuditReviewRequest> {
  requireAuditID(options.auditId);
  requireAuditID(options.findingId);
  requireAuditRevision(options.expectedRevision);
  const result = await api.request((client) =>
    client.POST("/v1/audits/{auditId}/findings/{findingId}/reviews", {
      params: {
        path: { auditId: options.auditId, findingId: options.findingId },
        header: {
          "Idempotency-Key": options.idempotencyKey,
          "If-Match": `"${options.expectedRevision}"`,
        },
      },
      body: {},
    }),
  );
  return structuredClone(requireData(result));
}

export async function decideAuditAction(
  api: PublicAPI,
  options: {
    auditId: string;
    requestId: string;
    expectedRevision: number;
    idempotencyKey: string;
    action: AuditReviewAction;
    rationale: string;
  },
): Promise<AuditActionDecisionResult> {
  requireAuditID(options.auditId);
  requireAuditID(options.requestId);
  requireAuditRevision(options.expectedRevision);
  const result = await api.request((client) =>
    client.POST("/v1/audits/{auditId}/reviews/{requestId}/decisions", {
      params: {
        path: { auditId: options.auditId, requestId: options.requestId },
        header: {
          "Idempotency-Key": options.idempotencyKey,
          "If-Match": `"${options.expectedRevision}"`,
        },
      },
      body: { action: options.action, rationale: options.rationale },
    }),
  );
  const response = requireData(result);
  if ("finding" in response) {
    throw invalidAuditResponse(result.response.status);
  }
  return structuredClone(response);
}

export async function decideAuditFinding(
  api: PublicAPI,
  options: {
    auditId: string;
    requestId: string;
    expectedRevision: number;
    idempotencyKey: string;
    decision: DecideAuditFindingRequest;
  },
): Promise<components["schemas"]["AuditFindingDecisionResult"]> {
  requireAuditID(options.auditId);
  requireAuditID(options.requestId);
  requireAuditRevision(options.expectedRevision);
  const result = await api.request((client) =>
    client.POST("/v1/audits/{auditId}/reviews/{requestId}/decisions", {
      params: {
        path: { auditId: options.auditId, requestId: options.requestId },
        header: {
          "Idempotency-Key": options.idempotencyKey,
          "If-Match": `"${options.expectedRevision}"`,
        },
      },
      body: options.decision,
    }),
  );
  const response = requireData(result);
  if (!("finding" in response)) {
    throw invalidAuditResponse(result.response.status);
  }
  return structuredClone(response);
}

export async function listAuditFindingProvenance(
  api: PublicAPI,
  auditId: string,
  findingId: string,
  request: PageRequest & { auditRevision: number; findingRevision: number },
): Promise<AuditFindingProvenancePage> {
  requireAuditID(auditId);
  requireAuditID(findingId);
  requireAuditRevision(request.auditRevision);
  requireAuditRevision(request.findingRevision);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/findings/{findingId}/provenance", {
      params: {
        path: { auditId, findingId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          auditRevision: request.auditRevision,
          findingRevision: request.findingRevision,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return structuredClone(requireData(result));
}

export function auditMutationAudit(value: Audit | AuditStartResponse): Audit {
  return "audit" in value ? value.audit : value;
}

export function auditNeedsPolling(state: AuditState): boolean {
  return (
    state === "active" ||
    state === "waiting_review" ||
    state === "finalizing" ||
    state === "cancelling" ||
    state === "deleting"
  );
}
