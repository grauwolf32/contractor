import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";
import { safeRunMetadataLabels } from "./run-metadata-labels";

export const QUEUE_PAGE_SIZE = 50;
export const QUEUE_STATES = [
  "initializing",
  "running",
  "cancelling",
] as const satisfies readonly QueueState[];
export const QUEUE_MEMBERSHIPS = [
  "standalone",
  "project",
  "evaluation",
] as const satisfies readonly QueueMembership[];

export type QueueState = components["schemas"]["NonTerminalWorkflowRunState"];
export type QueueMembership = components["schemas"]["RunQueueMembership"];
export type QueueItem = components["schemas"]["RunQueueItem"];
export type QueuePage = components["schemas"]["RunQueuePage"];
export type OwnerQueueControl = components["schemas"]["OwnerQueueControl"];

export interface QueuePageRequest {
  state?: QueueState;
  membership?: QueueMembership;
  cursor?: string;
}

const RESOURCE_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
const EVENT_SEQUENCE = /^(?:0|[1-9][0-9]{0,19})$/;
const QUEUE_CONTROL_REVISION = /^(?:0|[1-9][0-9]{0,18})$/;

function invalidQueueResponse(status: number): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message: "Server returned an invalid Queue response",
  });
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

function safeQueueItem(item: QueueItem, status: number): QueueItem {
  if (
    !RESOURCE_ID.test(item.runId) ||
    !QUEUE_STATES.includes(item.state) ||
    typeof item.workflow !== "string" ||
    item.workflow.length === 0 ||
    item.workflow.length > 193 ||
    !RESOURCE_ID.test(item.eventCursor?.generation ?? "") ||
    !EVENT_SEQUENCE.test(item.eventCursor?.sequence ?? "")
  ) {
    throw invalidQueueResponse(status);
  }
  if (
    item.project !== undefined &&
    (!RESOURCE_ID.test(item.project.projectId) ||
      typeof item.project.name !== "string" ||
      item.project.name.trim().length === 0 ||
      item.project.name.length > 160 ||
      (item.project.kind !== "project" && item.project.kind !== "evaluation"))
  ) {
    throw invalidQueueResponse(status);
  }
  return {
    runId: item.runId,
    ...(item.project === undefined
      ? {}
      : {
          project: {
            projectId: item.project.projectId,
            name: item.project.name,
            kind: item.project.kind,
          },
        }),
    workflow: item.workflow,
    state: item.state,
    labels: safeRunMetadataLabels(item.labels),
    eventCursor: { ...item.eventCursor },
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  };
}

function safeOwnerQueueControl(
  value: OwnerQueueControl,
  response: Response,
): OwnerQueueControl {
  if (
    typeof value.paused !== "boolean" ||
    !QUEUE_CONTROL_REVISION.test(value.revision) ||
    response.headers.get("ETag") !== `"${value.revision}"` ||
    (value.revision === "0" &&
      (value.paused || value.updatedAt !== undefined)) ||
    (value.revision !== "0" &&
      (typeof value.updatedAt !== "string" ||
        !Number.isFinite(Date.parse(value.updatedAt))))
  ) {
    throw invalidQueueResponse(response.status);
  }
  return { ...value };
}

export async function getOwnerQueueControl(
  api: PublicAPI,
): Promise<OwnerQueueControl> {
  const result = await api.request((client) => client.GET("/v1/queue/control"));
  return safeOwnerQueueControl(requireData(result), result.response);
}

export async function setOwnerQueuePaused(
  api: PublicAPI,
  paused: boolean,
  expectedRevision: string,
): Promise<OwnerQueueControl> {
  if (!QUEUE_CONTROL_REVISION.test(expectedRevision)) {
    throw new TypeError("Queue control revision is invalid");
  }
  const result = await api.request((client) =>
    client.PUT("/v1/queue/control", {
      params: { header: { "If-Match": `"${expectedRevision}"` } },
      body: { paused },
    }),
  );
  const control = safeOwnerQueueControl(requireData(result), result.response);
  if (control.paused !== paused) {
    throw invalidQueueResponse(result.response.status);
  }
  return control;
}

export async function listRunQueue(
  api: PublicAPI,
  request: QueuePageRequest = {},
): Promise<QueuePage> {
  if (
    (request.state !== undefined && !QUEUE_STATES.includes(request.state)) ||
    (request.membership !== undefined &&
      !QUEUE_MEMBERSHIPS.includes(request.membership)) ||
    (request.cursor !== undefined && request.cursor.length === 0)
  ) {
    throw new TypeError("Queue query is invalid");
  }
  const result = await api.request((client) =>
    client.GET("/v1/queue", {
      params: {
        query: {
          limit: QUEUE_PAGE_SIZE,
          ...(request.state === undefined ? {} : { state: request.state }),
          ...(request.membership === undefined
            ? {}
            : { membership: request.membership }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  if (!Array.isArray(page.items) || page.page === undefined) {
    throw invalidQueueResponse(result.response.status);
  }
  return {
    items: page.items.map((item) =>
      safeQueueItem(item, result.response.status),
    ),
    page: { ...page.page },
  };
}
