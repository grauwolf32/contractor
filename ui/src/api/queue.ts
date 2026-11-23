import type { PublicAPI } from "./client";
import { schemaEnum } from "./contract";
import { NON_TERMINAL_RUN_STATES } from "./states";
import { invalidAPIResponse, PublicAPIError, requireData } from "./error";
import type { components } from "./generated/public";

const QUEUE_PAGE_SIZE = 50;
// The Queue lists the Run states that can still change.
export const QUEUE_STATES = NON_TERMINAL_RUN_STATES;
export const QUEUE_MEMBERSHIPS =
  schemaEnum<QueueMembership>("RunQueueMembership");

export type QueueState = components["schemas"]["NonTerminalWorkflowRunState"];
export type QueueMembership = components["schemas"]["RunQueueMembership"];
export type QueueItem = components["schemas"]["RunQueueItem"];
type QueuePage = components["schemas"]["RunQueuePage"];
export type OwnerQueueControl = components["schemas"]["OwnerQueueControl"];

interface QueuePageRequest {
  state?: QueueState;
  membership?: QueueMembership;
  cursor?: string;
}

function invalidQueueResponse(status: number): PublicAPIError {
  return invalidAPIResponse(
    status,
    "Server returned an invalid Queue response",
  );
}

function requireOwnerQueueControl(
  value: OwnerQueueControl,
  response: Response,
): OwnerQueueControl {
  // The ETag names the representation's revision; revision 0 is the
  // never-written default, which is unpaused and has no update time.
  if (
    response.headers.get("ETag") !== `"${value.revision}"` ||
    (value.revision === "0"
      ? value.paused || value.updatedAt !== undefined
      : value.updatedAt === undefined)
  ) {
    throw invalidQueueResponse(response.status);
  }
  return value;
}

export async function getOwnerQueueControl(
  api: PublicAPI,
): Promise<OwnerQueueControl> {
  const result = await api.request((client) => client.GET("/v1/queue/control"));
  return requireOwnerQueueControl(requireData(result), result.response);
}

export async function setOwnerQueuePaused(
  api: PublicAPI,
  paused: boolean,
  expectedRevision: string,
): Promise<OwnerQueueControl> {
  // The client checks the If-Match value against the operation's schema.
  const result = await api.request((client) =>
    client.PUT("/v1/queue/control", {
      params: { header: { "If-Match": `"${expectedRevision}"` } },
      body: { paused },
    }),
  );
  const control = requireOwnerQueueControl(
    requireData(result),
    result.response,
  );
  // The replacement answers with the state it was asked to set.
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
  return requireData(result);
}
