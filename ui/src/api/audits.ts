import { requireAuditComposition } from "./audit-composition";
import type { PublicAPI } from "./client";
import { RESOURCE_ID } from "./contract";
import { invalidAPIResponse, PublicAPIError, requireData } from "./error";
import type { components } from "./generated/public";
import { requireProjectID } from "./projects";
import { CHANGING_AUDIT_STATES, isStateIn } from "./states";
import { CONFIG_NAME_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

export const AUDIT_PAGE_SIZE = 50;
// Audit IDs are public ResourceIds.
export const AUDIT_ID_PATTERN = RESOURCE_ID;

export type AuditProfile = components["schemas"]["AuditProfile"];
type AuditProfilePage = components["schemas"]["AuditProfilePage"];
export type Audit = components["schemas"]["Audit"];
type AuditPage = components["schemas"]["AuditPage"];
type AuditEventPage = components["schemas"]["AuditEventPage"];
export type AuditState = components["schemas"]["AuditState"];
export type AuditReport = components["schemas"]["AuditReport"];
type AuditStartResponse = components["schemas"]["AuditStartResponse"];
export type CreateAuditRequest = components["schemas"]["CreateAuditRequest"];
export type ExactArtifactRef = components["schemas"]["ExactArtifactRef"];
export type FindingProposalReceipt =
  components["schemas"]["FindingProposalReceipt"];
export type ProjectFinding = components["schemas"]["ProjectFinding"];
export type ProjectFindingSummary =
  components["schemas"]["ProjectFindingSummary"];
export type ProjectFindingEditResult =
  components["schemas"]["EditProjectFindingResult"];
export type EditProjectFindingRequest =
  components["schemas"]["EditProjectFindingRequest"];
export type FindingState = components["schemas"]["FindingState"];
export type FindingSeverity = components["schemas"]["FindingSeverity"];
export type FindingVerdict = components["schemas"]["FindingVerdict"];
type OwnerFindingPage = components["schemas"]["OwnerFindingPage"];

export type AuditWorkspace = components["schemas"]["AuditWorkspace"];
export interface FindingPageRequest extends PageRequest {
  auditRevision?: number;
  projectRevision?: number;
  state?: FindingState;
  verdict?: "true_positive" | "false_positive" | "unreviewed";
  severity?: FindingSeverity;
}

interface PageRequest {
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

interface ProjectAuditPageRequest extends PageRequest {
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
      params: { query: request },
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
      params: { query: request },
    }),
  );
  return requireData(result);
}

interface AuditCreateOptions {
  projectId: string;
  request: CreateAuditRequest;
  idempotencyKey: string;
}

export type AuditMutationAction =
  "start" | "pause" | "resume" | "cancel" | "delete";

interface AuditMutationOptions {
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
  signal?: AbortSignal,
): Promise<Audit> {
  requireAuditID(auditId);
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}", {
      ...(signal === undefined ? {} : { signal }),
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
    // A start replay still identifies the requested Audit.
    if (audit.auditId !== options.auditId) {
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
    report.status === "ready" &&
    (report.summaryArtifact === undefined || report.summary === undefined)
  ) {
    throw invalidAuditResponse(result.response.status);
  }
  return report;
}

export function auditMutationAudit(value: Audit | AuditStartResponse): Audit {
  return "audit" in value ? value.audit : value;
}

/**
 * These states need interval polling on their own. A paused Audit can still
 * change while submitted Runs drain and terminal outputs are collected;
 * callers should use auditPollInterval with the authoritative Audit projection.
 */
function auditNeedsPolling(state: AuditState): boolean {
  return isStateIn(CHANGING_AUDIT_STATES, state);
}

/** Polling interval for a set of Audits: off when none of them can change. */
export function auditPollInterval(
  audits: Iterable<
    Pick<Audit, "state" | "outstandingRunCount" | "collectionPending">
  >,
  interval = 5_000,
): number | false {
  for (const audit of audits) {
    if (
      auditNeedsPolling(audit.state) ||
      audit.outstandingRunCount > 0 ||
      audit.collectionPending
    )
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

export async function getProjectFinding(
  api: PublicAPI,
  projectId: string,
  findingId: string,
): Promise<ProjectFinding> {
  requireProjectID(projectId);
  requireAuditID(findingId);
  const result = await api.request((client) =>
    client.GET("/v1/projects/{projectId}/findings/{findingId}", {
      params: { path: { projectId, findingId } },
    }),
  );
  const finding = requireData(result);
  if (
    finding.projectId !== projectId ||
    finding.findingId !== findingId ||
    finding.claim === undefined
  )
    throw invalidAPIResponse(
      result.response.status,
      "Project Finding response does not match the requested identity",
    );
  requireRevisionETag(result.response, finding.revision);
  return finding;
}

export async function listProjectFindings(
  api: PublicAPI,
  projectId: string,
  request: { cursor?: string; projectRevision?: number } = {},
) {
  requireProjectID(projectId);
  return requireData(
    await api.request((client) =>
      client.GET("/v1/projects/{projectId}/findings", {
        params: {
          path: { projectId },
          query: { limit: AUDIT_PAGE_SIZE, ...request },
        },
      }),
    ),
  );
}

export async function editProjectFinding(
  api: PublicAPI,
  options: {
    projectId: string;
    findingId: string;
    expectedRevision: number;
    idempotencyKey: string;
    decision: EditProjectFindingRequest;
  },
): Promise<ProjectFindingEditResult> {
  requireProjectID(options.projectId);
  requireAuditID(options.findingId);
  requireAuditRevision(options.expectedRevision);
  const result = await api.request((client) =>
    client.PATCH("/v1/projects/{projectId}/findings/{findingId}", {
      params: {
        path: { projectId: options.projectId, findingId: options.findingId },
        header: {
          "If-Match": `"${options.expectedRevision}"`,
          "Idempotency-Key": options.idempotencyKey,
        },
      },
      body: options.decision,
    }),
  );
  const value = requireData(result);
  if (
    value.finding.projectId !== options.projectId ||
    value.finding.findingId !== options.findingId
  )
    throw invalidAPIResponse(
      result.response.status,
      "Project Finding response does not match the requested identity",
    );
  requireRevisionETag(result.response, value.finding.revision);
  return value;
}

export async function listProjectFindingSources(
  api: PublicAPI,
  projectId: string,
  findingId: string,
  request: { cursor?: string; projectRevision?: number } = {},
) {
  return requireData(
    await api.request((client) =>
      client.GET("/v1/projects/{projectId}/finding-sources", {
        params: {
          path: { projectId },
          query: { findingId, limit: AUDIT_PAGE_SIZE, ...request },
        },
      }),
    ),
  );
}
export async function getProjectFindingSource(
  api: PublicAPI,
  projectId: string,
  receiptId: string,
) {
  return requireData(
    await api.request((client) =>
      client.GET("/v1/projects/{projectId}/finding-sources/{receiptId}", {
        params: { path: { projectId, receiptId } },
      }),
    ),
  );
}
export async function listProjectFindingSourceHistory(
  api: PublicAPI,
  projectId: string,
  receiptId: string,
  request: { cursor?: string } = {},
) {
  return requireData(
    await api.request((client) =>
      client.GET(
        "/v1/projects/{projectId}/finding-sources/{receiptId}/history",
        {
          params: {
            path: { projectId, receiptId },
            query: { limit: AUDIT_PAGE_SIZE, ...request },
          },
        },
      ),
    ),
  );
}
