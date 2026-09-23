import type { PublicAPI } from "./client";
import { publicAPIError } from "./error";
import type { components, operations } from "./generated/public";

type Schema = components["schemas"];
export type EvalExperiment = Schema["EvalExperiment"];
export type EvalDraft = Schema["EvalDraft"];
export type EvalVariant = Schema["EvalVariant"];
export type EvalCase = Schema["EvalCase"];
export type EvalArtifact = Schema["EvalArtifact"];
export type EvalDataset = Schema["EvalDataset"];
export type EvalDatasetInput = Schema["EvalDatasetInput"];
export type EvalCapabilities = Schema["EvalCapabilities"];
export type EvalCheck = Schema["EvalCheck"];
export type EvalMember = Schema["EvalMemberView"];
export type EvalPair = Schema["EvalPair"];
export type EvalChart = Schema["EvalChart"];
export type EvalSummary = Schema["EvalSummary"];
export type EvalCommand = Schema["EvalCommand"];
export type EvalCommandReceipt = Schema["EvalCommandReceipt"];
export type EvalReceipt = Schema["EvalExperimentReceipt"];
export type EvalReview = Schema["EvalReview"];
export type EvalAssessment = Schema["EvalAssessmentInput"];
export const EVAL_PAGE_SIZE = 50;
export const EVAL_POLL_MS = 2000;

export function evalData<T>(result: {
  data?: T;
  error?: unknown;
  response: Response;
}): T {
  if (result.data === undefined)
    throw publicAPIError(result.response.status, result.error);
  return result.data;
}

export function evalNextPage(page: {
  hasMore: boolean;
  nextCursor?: string | null;
}): string | undefined {
  if (!page.hasMore) return undefined;
  if (!page.nextCursor)
    throw new Error("The next page is unavailable. Refresh to retry.");
  return page.nextCursor;
}

export async function evalInventory<T>(
  load: (cursor?: string) => Promise<{
    items: T[];
    page: { hasMore: boolean; nextCursor?: string | null };
  }>,
  signal: AbortSignal,
): Promise<T[]> {
  const items: T[] = [];
  const seen = new Set<string>();
  let cursor: string | undefined;
  do {
    signal.throwIfAborted();
    const page = await load(cursor);
    items.push(...page.items);
    cursor = evalNextPage(page.page);
    if (cursor && seen.has(cursor))
      throw new Error("The collection changed. Refresh to retry.");
    if (cursor) seen.add(cursor);
  } while (cursor);
  return items;
}

export function mutationHeaders(
  api: PublicAPI,
  key: string,
): { "Idempotency-Key": string };
export function mutationHeaders(
  api: PublicAPI,
  key: string,
  revision: number,
): { "Idempotency-Key": string; "If-Match": string };
export function mutationHeaders(
  api: PublicAPI,
  key: string,
  revision?: number,
): { "Idempotency-Key": string; "If-Match"?: string } {
  const headers = api.mutationHeaders({
    idempotencyKey: key,
    ...(revision === undefined ? {} : { ifMatch: `"${revision}"` }),
  });
  const ifMatch = headers.get("If-Match");
  return {
    "Idempotency-Key": headers.get("Idempotency-Key")!,
    ...(ifMatch === null ? {} : { "If-Match": ifMatch }),
  };
}

export async function getEvalCapabilities(
  api: PublicAPI,
  cursor?: string,
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-capabilities", {
        params: {
          query: { limit: EVAL_PAGE_SIZE, ...(cursor ? { cursor } : {}) },
        },
        ...(signal ? { signal } : {}),
      }),
    ),
  );
}

export async function listEvalDatasets(
  api: PublicAPI,
  projectId: string,
  cursor?: string,
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/projects/{projectId}/eval-datasets", {
        params: {
          path: { projectId },
          query: { limit: EVAL_PAGE_SIZE, ...(cursor ? { cursor } : {}) },
        },
        ...(signal ? { signal } : {}),
      }),
    ),
  );
}

export async function listEvalCases(
  api: PublicAPI,
  projectId: string,
  datasetId: string,
  revision: string,
  cursor?: string,
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET(
        "/v1/projects/{projectId}/eval-datasets/{datasetId}/revisions/{revision}/cases",
        {
          params: {
            path: { projectId, datasetId, revision },
            query: { limit: EVAL_PAGE_SIZE, ...(cursor ? { cursor } : {}) },
          },
          ...(signal ? { signal } : {}),
        },
      ),
    ),
  );
}

export async function importEvalDataset(
  api: PublicAPI,
  projectId: string,
  body: EvalDatasetInput,
  key: string,
) {
  return evalData(
    await api.request((client) =>
      client.POST("/v1/projects/{projectId}/eval-datasets", {
        params: {
          path: { projectId },
          header: mutationHeaders(api, key),
        },
        body,
      }),
    ),
  );
}

export type EvalListQuery = NonNullable<
  operations["listEvalExperiments"]["parameters"]["query"]
>;
export async function listEvalExperiments(
  api: PublicAPI,
  query: EvalListQuery = {},
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments", {
        params: { query: { limit: EVAL_PAGE_SIZE, ...query } },
        ...(signal ? { signal } : {}),
      }),
    ),
  );
}

export async function getEvalExperiment(
  api: PublicAPI,
  id: string,
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}", {
        params: { path: { id } },
        ...(signal ? { signal } : {}),
      }),
    ),
  );
}

export async function saveEvalDraft(
  api: PublicAPI,
  projectId: string,
  body: Schema["EvalDraftUpdate"],
  key: string,
  current?: { id: string; revision: number },
) {
  if (current)
    return evalData(
      await api.request((client) =>
        client.PATCH("/v1/eval-experiments/{id}", {
          params: {
            path: { id: current.id },
            header: mutationHeaders(api, key, current.revision),
          },
          body,
        }),
      ),
    );
  return evalData(
    await api.request((client) =>
      client.POST("/v1/projects/{projectId}/eval-experiments", {
        params: {
          path: { projectId },
          header: mutationHeaders(api, key),
        },
        body: { ...body, controlMode: "server" },
      }),
    ),
  );
}

export async function commandEvalExperiment(
  api: PublicAPI,
  id: string,
  body: EvalCommand,
  key: string,
  revision: number,
) {
  return evalData(
    await api.request((client) =>
      client.POST("/v1/eval-experiments/{id}/commands", {
        params: { path: { id }, header: mutationHeaders(api, key, revision) },
        body,
      }),
    ),
  );
}

export async function getEvalCommand(
  api: PublicAPI,
  id: string,
  commandId: string,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/commands/{commandId}", {
        params: { path: { id, commandId } },
      }),
    ),
  );
}

export type EvalMemberQuery = NonNullable<
  operations["listEvalMembers"]["parameters"]["query"]
>;
export async function listEvalMembers(
  api: PublicAPI,
  id: string,
  query: EvalMemberQuery = {},
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/members", {
        params: { path: { id }, query: { limit: EVAL_PAGE_SIZE, ...query } },
        ...(signal ? { signal } : {}),
      }),
    ),
  );
}

export type EvalPairQuery = NonNullable<
  operations["listEvalPairs"]["parameters"]["query"]
>;
export async function listEvalPairs(
  api: PublicAPI,
  id: string,
  query: EvalPairQuery = {},
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/pairs", {
        params: { path: { id }, query: { limit: EVAL_PAGE_SIZE, ...query } },
        ...(signal ? { signal } : {}),
      }),
    ),
  );
}

export async function getEvalPair(
  api: PublicAPI,
  id: string,
  pairId: string,
  viewSnapshot?: string,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/pairs/{pairId}", {
        params: {
          path: { id, pairId },
          query: { ...(viewSnapshot ? { viewSnapshot } : {}) },
        },
      }),
    ),
  );
}

export type EvalChartQuery = NonNullable<
  operations["getEvalChart"]["parameters"]["query"]
>;
export async function getEvalChart(
  api: PublicAPI,
  id: string,
  chart: EvalChart["chart"],
  query: EvalChartQuery = {},
  signal?: AbortSignal,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/charts/{chart}", {
        params: { path: { id, chart }, query },
        ...(signal ? { signal } : {}),
      }),
    ),
  );
}

export async function getEvalReview(
  api: PublicAPI,
  id: string,
  memberId: string,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/members/{memberId}/review", {
        params: { path: { id, memberId } },
      }),
    ),
  );
}

export async function submitEvalAssessment(
  api: PublicAPI,
  id: string,
  memberId: string,
  body: EvalAssessment,
  key: string,
) {
  return evalData(
    await api.request((client) =>
      client.POST("/v1/eval-experiments/{id}/members/{memberId}/assessments", {
        params: {
          path: { id, memberId },
          header: mutationHeaders(api, key),
        },
        body,
      }),
    ),
  );
}

export async function selectEvalAssessment(
  api: PublicAPI,
  id: string,
  body: Schema["EvalSelectionInput"],
  key: string,
  revision: number,
) {
  return evalData(
    await api.request((client) =>
      client.POST("/v1/eval-experiments/{id}/selections", {
        params: { path: { id }, header: mutationHeaders(api, key, revision) },
        body,
      }),
    ),
  );
}

export async function getEvalReport(
  api: PublicAPI,
  id: string,
  viewSnapshot: string,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/report", {
        params: { path: { id }, query: { viewSnapshot } },
      }),
    ),
  );
}

export async function listEvalExecutions(
  api: PublicAPI,
  id: string,
  memberId: string,
  cursor?: string,
) {
  return evalData(
    await api.request((client) =>
      client.GET("/v1/eval-experiments/{id}/members/{memberId}/executions", {
        params: {
          path: { id, memberId },
          query: { limit: EVAL_PAGE_SIZE, ...(cursor ? { cursor } : {}) },
        },
      }),
    ),
  );
}
