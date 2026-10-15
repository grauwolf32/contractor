import { RESOURCE_ID } from "./contract";
import { requireAuditComposition } from "./audit-composition";
import type { PublicAPI } from "./client";
import { invalidAPIResponse, PublicAPIError, requireData } from "./error";
import type { components } from "./generated/public";
import { requireProjectID } from "./projects";
import { CONFIG_NAME_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

export const AUDIT_PAGE_SIZE = 50;
// Audit IDs are public ResourceIds.
export const AUDIT_ID_PATTERN = RESOURCE_ID;

export type AuditProfile = components["schemas"]["AuditProfile"];
export type AuditProfilePage = components["schemas"]["AuditProfilePage"];
export type Audit = components["schemas"]["Audit"];
export type AuditPage = components["schemas"]["AuditPage"];
export type AuditEvent = components["schemas"]["AuditEvent"];
export type AuditEventPage = components["schemas"]["AuditEventPage"];
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
export type OwnerFindingPage = components["schemas"]["OwnerFindingPage"];
export type OwnerReviewPage = components["schemas"]["OwnerReviewPage"];
export type AuditReviewAction = components["schemas"]["AuditReviewAction"];
export type AuditActionDecisionResult =
  components["schemas"]["AuditActionDecisionResult"];
export type AuditFindingProvenancePage =
  components["schemas"]["AuditFindingProvenancePage"];
export type DecideAuditFindingRequest =
  components["schemas"]["DecideAuditFindingRequest"];

export type AuditWorkspace = components["schemas"]["AuditWorkspace"];
export interface FindingPageRequest extends PageRequest {
  auditRevision?: number;
  state?: AuditFindingState;
  verdict?: "true_positive" | "false_positive" | "unreviewed";
  severity?: AuditFindingSeverity;
}
export interface ReviewPageRequest extends PageRequest {
  auditRevision?: number;
  finding?: string;
  state?: AuditReviewRequest["state"];
}

export interface PageRequest {
  cursor?: string;
}

export async function listAuditEvents(
  api: PublicAPI,
  auditId: string,
  request: PageRequest = {},
  signal?: AbortSignal,
): Promise<AuditEventPage> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/events", {
      ...(signal === undefined ? {} : { signal }),
      params: {
        path: { auditId },
        query: { limit: AUDIT_PAGE_SIZE, ...request },
      },
    }),
  );
  const value = requireData(result);
  // Cross-field: the events are this Audit's, newest first, within the
  // reported sequence and total, and a continued page has a cursor.
  if (
    value.total < value.items.length ||
    (value.page.hasMore &&
      (value.items.length === 0 || value.page.nextCursor === undefined)) ||
    value.items.some(
      (event, index) =>
        event.auditId !== auditId ||
        event.sequence > value.throughSequence ||
        (index > 0 && event.sequence >= value.items[index - 1]!.sequence),
    )
  )
    throw invalidAuditResponse(result.response.status);
  return value;
}

export interface ProjectAuditPageRequest extends PageRequest {
  limit?: number;
  projectId: string;
  state?: AuditState;
  profile?: string;
}

export async function listOwnerAudits(
  api: PublicAPI,
  request: PageRequest & { state?: string } = {},
  signal?: AbortSignal,
): Promise<AuditPage> {
  const result = await api.request((client) =>
    client.GET("/v1/audits", {
      ...(signal === undefined ? {} : { signal }),
      params: { query: { limit: 200, ...request } },
    }),
  );
  const page = requireData(result);
  for (const audit of page.items)
    requireAuditComposition(audit, result.response.status);
  return page;
}

export async function listOwnerFindings(
  api: PublicAPI,
  request: Omit<FindingPageRequest, "auditRevision"> & {
    auditState?: string;
  } = {},
  signal?: AbortSignal,
): Promise<OwnerFindingPage> {
  const result = await api.request((client) =>
    client.GET("/v1/findings", {
      ...(signal === undefined ? {} : { signal }),
      params: { query: { limit: 200, ...request } },
    }),
  );
  return requireData(result);
}

export async function listOwnerReviews(
  api: PublicAPI,
  request: PageRequest & {
    state?: AuditReviewRequest["state"];
    auditState?: string;
  } = {},
  signal?: AbortSignal,
): Promise<OwnerReviewPage> {
  const result = await api.request((client) =>
    client.GET("/v1/reviews", {
      ...(signal === undefined ? {} : { signal }),
      params: { query: { limit: 200, ...request } },
    }),
  );
  return requireData(result);
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
  deadlineSeconds?: number;
}

function invalidAuditResponse(status: number): PublicAPIError {
  return invalidAPIResponse(
    status,
    "Server returned an invalid Audit response",
  );
}

function requireProfileIdentity(name: string, version: string): void {
  if (
    !CONFIG_NAME_PATTERN.test(name) ||
    !CONFIG_VERSION_PATTERN.test(version)
  ) {
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

// Cross-field: the ETag header names the revision of the body.
function requireRevisionETag(response: Response, revision: number): void {
  if (response.headers.get("ETag") !== `"${revision}"`) {
    throw invalidAuditResponse(response.status);
  }
}

// Cross-field: the ETag header names the digest of the body.
function requireProfileETag(response: Response, digest: string): void {
  if (response.headers.get("ETag") !== `"${digest}"`) {
    throw invalidAuditResponse(response.status);
  }
}

export async function listAuditProfiles(
  api: PublicAPI,
  request: PageRequest & { signal?: AbortSignal } = {},
): Promise<AuditProfilePage> {
  const result = await api.request((client) =>
    client.GET("/v1/audit-profiles", {
      ...(request.signal === undefined ? {} : { signal: request.signal }),
      params: {
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}

export async function getAuditProfile(
  api: PublicAPI,
  name: string,
  version: string,
  signal?: AbortSignal,
): Promise<AuditProfile> {
  requireProfileIdentity(name, version);
  const result = await api.request((client) =>
    client.GET("/v1/audit-profiles/{name}/versions/{version}", {
      ...(signal === undefined ? {} : { signal }),
      params: { path: { name, version } },
    }),
  );
  const profile = requireData(result);
  // Cross-field: the Server answered for the requested version.
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
  const audit = requireAuditComposition(
    requireData(result),
    result.response.status,
  );
  // Cross-field: a new Audit is a draft of the requested Project.
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
  if (
    request.limit !== undefined &&
    (!Number.isInteger(request.limit) ||
      request.limit < 1 ||
      request.limit > 100)
  )
    throw new TypeError("Audit page size is invalid");
  const result = await api.request((client) =>
    client.GET("/v1/projects/{projectId}/audits", {
      params: {
        path: { projectId: request.projectId },
        query: {
          limit: request.limit ?? AUDIT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
          ...(request.state === undefined ? {} : { state: request.state }),
          ...(request.profile === undefined
            ? {}
            : { profile: request.profile }),
        },
      },
    }),
  );
  const page = requireData(result);
  for (const audit of page.items)
    requireAuditComposition(audit, result.response.status);
  return page;
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
  const audit = requireAuditComposition(
    requireData(result),
    result.response.status,
  );
  // Cross-field: the Server answered for the requested Audit.
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
      client.POST("/v1/audits/{auditId}/start", {
        ...common,
        ...(options.deadlineSeconds === undefined
          ? {}
          : { body: { deadlineSeconds: options.deadlineSeconds } }),
      }),
    );
    const started = requireData(result);
    const audit = requireAuditComposition(
      started.audit,
      result.response.status,
    );
    // Cross-field: the started Audit is the requested one, and its first
    // Round is the one it now points to.
    if (
      audit.auditId !== options.auditId ||
      (started.round !== undefined &&
        started.round.roundId !== audit.currentRoundId)
    ) {
      throw invalidAuditResponse(result.response.status);
    }
    requireRevisionETag(result.response, audit.revision);
    return started;
  }
  if (action === "pause") {
    const result = await api.request((client) =>
      client.POST("/v1/audits/{auditId}/pause", common),
    );
    return finishAuditMutation(result, options.auditId);
  }
  if (action === "resume") {
    const result = await api.request((client) =>
      client.POST("/v1/audits/{auditId}/resume", {
        ...common,
        ...(options.deadlineSeconds === undefined
          ? {}
          : { body: { deadlineSeconds: options.deadlineSeconds } }),
      }),
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
  const audit = requireAuditComposition(
    requireData(result),
    result.response.status,
  );
  // Cross-field: the Server answered for the requested Audit.
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
  return requireData(result);
}

/**
 * One coverage page, read at one Audit revision. With `auditRevision` the
 * Server reads the page at exactly that revision or answers 409.
 */
export async function listAuditCoverage(
  api: PublicAPI,
  auditId: string,
  request: PageRequest & { round?: string; auditRevision?: number } = {},
): Promise<AuditCoveragePage> {
  requireAuditID(auditId);
  if (request.round !== undefined) requireAuditID(request.round);
  if (request.auditRevision !== undefined)
    requireAuditRevision(request.auditRevision);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/coverage", {
      params: {
        path: { auditId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          ...(request.round === undefined ? {} : { round: request.round }),
          ...(request.auditRevision === undefined
            ? {}
            : { auditRevision: request.auditRevision }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}

interface ReportReadQueue {
  active: number;
  waiting: Array<() => void>;
}

// Inbox and Reports can read many retained reports at once. Each report
// reads file content on Server, whose transfer capacity is deliberately small.
// Share two slots across the views using the same API client.
const reportReadQueues = new WeakMap<PublicAPI, ReportReadQueue>();

async function readReportWithCapacity<T>(
  api: PublicAPI,
  read: () => Promise<T>,
): Promise<T> {
  let queue = reportReadQueues.get(api);
  if (queue === undefined) {
    queue = { active: 0, waiting: [] };
    reportReadQueues.set(api, queue);
  }
  if (queue.active < 2) queue.active++;
  else await new Promise<void>((resolve) => queue.waiting.push(resolve));
  try {
    return await read();
  } finally {
    // Transfer the slot to the oldest waiter before admitting another read.
    const next = queue.waiting.shift();
    if (next === undefined) queue.active--;
    else next();
  }
}

export async function getAuditReport(
  api: PublicAPI,
  auditId: string,
): Promise<AuditReport> {
  requireAuditID(auditId);
  const result = await readReportWithCapacity(api, () =>
    api.request((client) =>
      client.GET("/v1/audits/{auditId}/report", {
        params: { path: { auditId } },
      }),
    ),
  );
  const report = requireData(result);
  // Cross-field: a proposed or ready report carries its summary.
  if (
    (report.status === "ready" || report.status === "proposed") &&
    (report.summaryArtifact === undefined || report.summary === undefined)
  ) {
    throw invalidAuditResponse(result.response.status);
  }
  return report;
}

export async function listAuditFindings(
  api: PublicAPI,
  auditId: string,
  request: FindingPageRequest = {},
): Promise<AuditFindingPage> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/findings", {
      params: {
        path: { auditId },
        query: { limit: AUDIT_PAGE_SIZE, ...request },
      },
    }),
  );
  return requireData(result);
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
  // Cross-field: the Server answered for the requested possible issue.
  if (finding.auditId !== auditId || finding.findingId !== findingId) {
    throw invalidAuditResponse(result.response.status);
  }
  requireRevisionETag(result.response, finding.revision);
  return finding;
}

export async function listAuditReviews(
  api: PublicAPI,
  auditId: string,
  request: ReviewPageRequest = {},
): Promise<AuditReviewPage> {
  requireAuditID(auditId);
  if (request.finding !== undefined) requireAuditID(request.finding);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/reviews", {
      params: {
        path: { auditId },
        query: { limit: AUDIT_PAGE_SIZE, ...request },
      },
    }),
  );
  return requireData(result);
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
  return requireData(result);
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
  // Cross-field: an action decision does not answer with a finding result.
  if ("finding" in response) {
    throw invalidAuditResponse(result.response.status);
  }
  return response;
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
  // Cross-field: a finding decision answers with the decided finding.
  if (!("finding" in response)) {
    throw invalidAuditResponse(result.response.status);
  }
  return response;
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
  return requireData(result);
}

export function auditMutationAudit(value: Audit | AuditStartResponse): Audit {
  return "audit" in value ? value.audit : value;
}

/**
 * These states need interval polling on their own. A paused Audit can still
 * change while submitted Runs drain; callers should use auditPollInterval
 * when they have the outstanding Run count.
 */
export function auditNeedsPolling(state: AuditState): boolean {
  return (
    state === "active" ||
    state === "waiting_review" ||
    state === "finalizing" ||
    state === "cancelling" ||
    state === "deleting"
  );
}

/** Polling interval for a set of Audits: off when none of them can change. */
export function auditPollInterval(
  audits: Iterable<Pick<Audit, "state" | "outstandingRunCount">>,
  interval = 5_000,
): number | false {
  for (const audit of audits) {
    if (auditNeedsPolling(audit.state) || audit.outstandingRunCount > 0)
      return interval;
  }
  return false;
}

export async function getAuditWorkspace(
  api: PublicAPI,
  auditId: string,
): Promise<AuditWorkspace> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/workspace", {
      params: { path: { auditId } },
    }),
  );
  const value = requireData(result);
  // Cross-field: the Server answered for the requested Audit.
  if (value.auditId !== auditId)
    throw invalidAuditResponse(result.response.status);
  return value;
}

export async function getAuditReview(
  api: PublicAPI,
  auditId: string,
  requestId: string,
): Promise<AuditReviewRequest> {
  requireAuditID(auditId);
  requireAuditID(requestId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/reviews/{requestId}", {
      params: { path: { auditId, requestId } },
    }),
  );
  const value = requireData(result);
  // Cross-field: the Server answered for the requested review request.
  if (value.auditId !== auditId || value.requestId !== requestId)
    throw invalidAuditResponse(result.response.status);
  requireRevisionETag(result.response, value.revision);
  return value;
}
