// One fake Public API Server for unit tests. Routes are keyed by the
// operation ("METHOD /v1/path/{template}") they answer, and every JSON success
// response is checked against that operation's generated response schema, as
// the real client checks it: a fixture that breaks the contract fails the test
// that served it, with the location of the violation.
import { expect } from "vitest";

import { responseViolation, type ResponseOperation } from "../api/contract";
import { RESPONSE_SCHEMAS } from "../api/generated/response-schemas";
import type { components } from "../api/generated/public";

export const API_VERSION_HEADERS = {
  "X-Contractor-API-Version": "contractor.public.v1",
};

/** Responses a route built on purpose to break the contract. */
const unchecked = new WeakSet<Response>();

/** A JSON response with the public API version header. */
export function json(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...API_VERSION_HEADERS,
      ...headers,
    },
  });
}

/** A JSON success response the fake Server must not check (A1 probes). */
export function invalidJSON(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  const response = json(value, status, headers);
  unchecked.add(response);
  return response;
}

/** An error envelope with `status`. */
export function failure(
  status: number,
  code = status === 404 ? "not_found" : "internal_error",
  message = "Server failed",
  retryable = false,
): Response {
  return json({ code, message, retryable, requestId: "request_test" }, status);
}

/** 409: another write changed the resource or its state first. */
export function conflict(code = "conflict", message = "Changed"): Response {
  return failure(409, code, message);
}

/** 412: the If-Match revision is no longer current. */
export function preconditionFailed(message = "The resource changed"): Response {
  return failure(412, "precondition_failed", message);
}

/** 204 without a body. */
export function noContent(): Response {
  return new Response(null, { status: 204, headers: API_VERSION_HEADERS });
}

/** A gateway's error page: no API version header, no error envelope. */
export function gatewayFailure(status: number): Response {
  return new Response("Gateway error", {
    status,
    headers: { "Content-Type": "text/plain" },
  });
}

/** One keyset page; `nextCursor` marks that more pages follow. */
export function page<T>(
  items: T[],
  nextCursor?: string,
): { items: T[]; page: components["schemas"]["PageInfo"] } {
  return {
    items,
    page:
      nextCursor === undefined
        ? { hasMore: false }
        : { hasMore: true, nextCursor },
  };
}

export interface FakeCall {
  request: Request;
  url: URL;
  /** Decoded path parameters of the matched operation. */
  params: Record<string, string>;
}

export type Route = (call: FakeCall) => Response | Promise<Response>;
export type Routes = Partial<Record<ResponseOperation, Route>>;

/** Answers a request no route matched; undefined falls through to 404. */
export type Fallback = (
  request: Request,
  url: URL,
) => Response | undefined | Promise<Response | undefined>;

interface CompiledRoute {
  operation: ResponseOperation;
  method: string;
  pattern: RegExp;
  names: string[];
}

const compiled: CompiledRoute[] = Object.keys(RESPONSE_SCHEMAS).map((key) => {
  const operation = key as ResponseOperation;
  const [method = "", template = ""] = operation.split(" ");
  const names: string[] = [];
  const source = template
    .split(/(\{[^}]+\})/)
    .map((part) => {
      const name = /^\{([^}]+)\}$/.exec(part)?.[1];
      if (name === undefined) return part.replace(/[.*+?^$()|[\]\\]/g, "\\$&");
      names.push(name);
      return "([^/]+)";
    })
    .join("");
  return { operation, method, pattern: new RegExp(`^${source}$`), names };
});

/** The operation a request addresses; literal segments win over templates. */
export function operationOf(
  method: string,
  pathname: string,
):
  { operation: ResponseOperation; params: Record<string, string> } | undefined {
  let best:
    | { operation: ResponseOperation; params: Record<string, string> }
    | undefined;
  let bestNames = Number.POSITIVE_INFINITY;
  for (const route of compiled) {
    if (route.method !== method) continue;
    const match = route.pattern.exec(pathname);
    if (match === null || route.names.length >= bestNames) continue;
    bestNames = route.names.length;
    best = {
      operation: route.operation,
      params: Object.fromEntries(
        route.names.map((name, index) => [
          name,
          decodeURIComponent(match[index + 1] ?? ""),
        ]),
      ),
    };
  }
  return best;
}

async function checkResponse(
  operation: ResponseOperation | undefined,
  label: string,
  response: Response,
): Promise<void> {
  if (!response.ok || unchecked.has(response) || response.status === 204) {
    return;
  }
  if (
    response.headers.get("Content-Type")?.split(";")[0]?.trim() !==
    "application/json"
  ) {
    return;
  }
  const body = (await response.clone().json()) as unknown;
  const violation =
    operation === undefined
      ? "the request matches no public operation"
      : responseViolation(operation, response.status, body);
  // Soft: the client turns a thrown error into a network failure, which
  // would hide the fixture problem behind an unrelated UI state.
  expect
    .soft(violation, `fake Server answered ${label} ${response.status}`)
    .toBeUndefined();
}

export interface FakeServer {
  /** The fetch implementation to give PublicAPI. */
  fetch: (input: RequestInfo | URL) => Promise<Response>;
  /** Every request, in order (clones, so bodies stay readable). */
  requests: Request[];
  /** Requests with this method whose path ends with `suffix`. */
  sent(method: string, suffix: string): Request[];
  /** Adds or replaces the route of `operation`. */
  route(operation: ResponseOperation, route: Route): void;
}

/**
 * A fake Server answering `routes`; requests no route answers go to
 * `fallback`, then get a 404 error envelope.
 */
export function fakeServer(
  routes: Routes = {},
  fallback?: Fallback,
): FakeServer {
  const table = new Map<ResponseOperation, Route>(
    Object.entries(routes) as [ResponseOperation, Route][],
  );
  const requests: Request[] = [];
  return {
    requests,
    sent: (method, suffix) =>
      requests.filter(
        (request) =>
          request.method === method &&
          new URL(request.url).pathname.endsWith(suffix),
      ),
    route: (operation, route) => {
      table.set(operation, route);
    },
    fetch: async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request.clone());
      const url = new URL(request.url);
      const matched = operationOf(request.method, url.pathname);
      const route =
        matched === undefined ? undefined : table.get(matched.operation);
      const label = `${request.method} ${url.pathname}`;
      let response: Response | undefined;
      if (route !== undefined && matched !== undefined) {
        response = await route({ request, url, params: matched.params });
      } else {
        response = await fallback?.(request, url);
      }
      if (response === undefined) {
        return failure(404, "not_found", `No fake route for ${label}`);
      }
      await checkResponse(matched?.operation, label, response);
      return response;
    },
  };
}
