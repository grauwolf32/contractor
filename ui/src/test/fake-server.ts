// One fake Public API Server for unit tests. Routes are keyed by the
// operation ("METHOD /v1/path/{template}") they answer, and every JSON success
// response is checked against that operation's generated response schema, as
// the real client checks it: a fixture that breaks the contract fails the test
// that served it, with the location of the violation. A status the contract
// declares without a JSON body (artifact content, whatever its media type) is
// raw bytes and is not checked.
import { expect } from "vitest";

import { responseViolation, type ResponseOperation } from "../api/contract";
import { RESPONSE_SCHEMAS } from "../api/generated/response-schemas";
import { failure, json, operationOf } from "./responses";

/** Responses a route built on purpose to break the contract. */
const unchecked = new WeakSet<Response>();

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

async function checkResponse(
  operation: ResponseOperation | undefined,
  label: string,
  response: Response,
  owner: string | undefined,
): Promise<void> {
  if (!response.ok || unchecked.has(response) || response.status === 204) {
    return;
  }
  // Raw content: the operation declares this status without a JSON body.
  if (operation !== undefined) {
    const declared: Readonly<Record<string, string | null>> =
      RESPONSE_SCHEMAS[operation];
    if (declared[String(response.status)] === null) return;
  }
  // A read that outlives the test that served it must not fail a later test.
  if (owner !== undefined && expect.getState().currentTestName !== owner) {
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
  /** Whether a route answers `operation` (a fallback does not count). */
  handles(operation: ResponseOperation): boolean;
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
  // The test this Server serves; its late reads are not checked in another.
  const owner = expect.getState().currentTestName;
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
    handles: (operation) => table.has(operation),
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
      await checkResponse(matched?.operation, label, response, owner);
      return response;
    },
  };
}
