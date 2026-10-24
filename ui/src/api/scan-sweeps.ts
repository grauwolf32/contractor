import type { PublicAPI } from "./client";
import {
  RESOURCE_ID,
  conformsTo,
  schemaLimit,
  type ResponseOperation,
} from "./contract";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";

export type ScanSweep = components["schemas"]["ScanSweep"];
export type CreateScanSweepRequest =
  components["schemas"]["CreateScanSweepRequest"];
type SweepPage = components["schemas"]["ScanSweepPage"];
type MemberPage = components["schemas"]["ScanSweepMemberPage"];
export const MAX_SWEEP_PROJECTS = schemaLimit(
  "CreateScanSweepRequest",
  "maxItems",
  "properties",
  "projectIds",
);
// LIM-SCN-79: shared client response ceiling, independent of snapshot/tick limits.
const RESPONSE_BYTES = 4 * 1024 * 1024;
const PAGE_SIZE = 50;
export const sweepKeys = {
  all: ["scan-sweeps"] as const,
  list: (cursor?: string) => ["scan-sweeps", "list", cursor ?? null] as const,
  detail: (id: string) => ["scan-sweeps", "detail", id] as const,
  members: (id: string) => ["scan-sweeps", "members", id] as const,
};
export function sweepSettled(sweep: ScanSweep): boolean {
  return sweep.state === "completed" || sweep.state === "cancelled";
}
function identity(id: string) {
  if (!RESOURCE_ID.test(id)) throw new TypeError("Sweep ID is invalid");
}
function requireSweep(sweep: ScanSweep, id?: string): ScanSweep {
  const c = sweep.counts;
  if (
    (id !== undefined && sweep.sweepId !== id) ||
    c.total < 1 ||
    c.total !== c.pending + c.admitted + c.skipped + c.failed + c.cancelled ||
    c.unsettled > c.total ||
    (sweepSettled(sweep) && (c.unsettled !== 0 || sweep.settledAt === null))
  )
    throw invalidAPIResponse(200, "Sweep state or counters are inconsistent");
  return sweep;
}
async function body<T>(
  api: PublicAPI,
  operation: ResponseOperation,
  result: { data?: unknown; error?: unknown; response: Response },
): Promise<T> {
  requireData(result);
  return api.readJSON<T>(operation, result.response, RESPONSE_BYTES);
}
function page<
  T extends {
    items: unknown[];
    page: { hasMore: boolean; nextCursor?: string };
  },
>(value: T): T {
  if (
    value.items.length > PAGE_SIZE ||
    (value.page.hasMore && (value.items.length === 0 || !value.page.nextCursor))
  )
    throw invalidAPIResponse(200, "Sweep page continuation is invalid");
  return value;
}
export async function createScanSweep(
  api: PublicAPI,
  request: CreateScanSweepRequest,
  key: string,
) {
  if (
    !conformsTo("CreateScanSweepRequest", request) ||
    (request.allProjects === true) === (request.projectIds !== undefined) ||
    request.parameters?.target !== undefined
  )
    throw new TypeError(
      "Choose explicit Projects or all Projects; targets come from the Server",
    );
  const result = await api.request((client) =>
    client.POST("/v1/scan-sweeps", {
      body: request,
      params: { header: { "Idempotency-Key": key } },
      parseAs: "stream",
    }),
  );
  return requireSweep(
    await body<ScanSweep>(api, "POST /v1/scan-sweeps", result),
  );
}
export async function listScanSweeps(api: PublicAPI, cursor?: string) {
  const result = await api.request((client) =>
    client.GET("/v1/scan-sweeps", {
      params: {
        query: {
          limit: PAGE_SIZE,
          ...(cursor === undefined ? {} : { cursor }),
        },
      },
      parseAs: "stream",
    }),
  );
  const value = page(await body<SweepPage>(api, "GET /v1/scan-sweeps", result));
  if (
    cursor !== undefined &&
    value.page.hasMore &&
    value.page.nextCursor === cursor
  )
    throw invalidAPIResponse(200, "Sweep cursor did not advance");
  value.items.forEach((item) => requireSweep(item));
  return value;
}
export async function getScanSweep(
  api: PublicAPI,
  id: string,
  signal?: AbortSignal,
) {
  identity(id);
  const result = await api.request((client) =>
    client.GET("/v1/scan-sweeps/{sweepId}", {
      params: { path: { sweepId: id } },
      ...(signal === undefined ? {} : { signal }),
      parseAs: "stream",
    }),
  );
  return requireSweep(
    await body<ScanSweep>(api, "GET /v1/scan-sweeps/{sweepId}", result),
    id,
  );
}
export async function listScanSweepMembers(
  api: PublicAPI,
  id: string,
  cursor?: string,
  signal?: AbortSignal,
) {
  identity(id);
  const result = await api.request((client) =>
    client.GET("/v1/scan-sweeps/{sweepId}/members", {
      params: {
        path: { sweepId: id },
        query: {
          limit: PAGE_SIZE,
          ...(cursor === undefined ? {} : { cursor }),
        },
      },
      ...(signal === undefined ? {} : { signal }),
      parseAs: "stream",
    }),
  );
  const value = page(
    await body<MemberPage>(
      api,
      "GET /v1/scan-sweeps/{sweepId}/members",
      result,
    ),
  );
  if (
    cursor !== undefined &&
    value.page.hasMore &&
    value.page.nextCursor === cursor
  )
    throw invalidAPIResponse(200, "Sweep member cursor did not advance");
  const projects = new Set<string>();
  let ordinal = -1;
  for (const member of value.items) {
    if (
      member.ordinal <= ordinal ||
      projects.has(member.projectId) ||
      (member.state === "admitted") !== (member.runId !== null) ||
      (member.runState !== null && member.runId === null)
    )
      throw invalidAPIResponse(200, "Sweep members are inconsistent");
    ordinal = member.ordinal;
    projects.add(member.projectId);
  }
  return value;
}
export async function cancelScanSweep(api: PublicAPI, id: string) {
  identity(id);
  const result = await api.request((client) =>
    client.POST("/v1/scan-sweeps/{sweepId}/cancel", {
      params: { path: { sweepId: id } },
      parseAs: "stream",
    }),
  );
  return requireSweep(
    await body<ScanSweep>(api, "POST /v1/scan-sweeps/{sweepId}/cancel", result),
    id,
  );
}
export async function purgeScanSweep(api: PublicAPI, id: string) {
  identity(id);
  const result = await api.request((client) =>
    client.DELETE("/v1/scan-sweeps/{sweepId}", {
      params: { path: { sweepId: id } },
    }),
  );
  if (result.response.status !== 204) requireData(result);
}
