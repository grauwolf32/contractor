// One mock Public API for the browser specs, mocked and process alike.
//
// The production UI validates every JSON success response against the pinned
// OpenAPI document (UI-R196), so a mock body the Server could never send makes
// the page show "The Server sent an answer this page cannot read" instead of
// the content under test. Every response a spec serves for /v1 therefore goes
// through this module, which checks it against the same generated contract
// before it is served: a request that matches no public operation, a status
// the operation does not declare, or a success body that breaks the
// operation's response schema throws at once, naming the operation and the
// failing location. Playwright fails the running test with that error.
//
// The browser gate (ui/server/browser-gate.mjs) refuses spec files that call
// route.fulfill themselves, so no spec can serve an unchecked answer.
import type { Page, Request, Route, TestInfo } from "@playwright/test";

import packageMetadata from "../../package.json" with { type: "json" };
import {
  responseViolation,
  type ResponseOperation,
} from "../../src/api/contract";
import { RESPONSE_SCHEMAS } from "../../src/api/generated/response-schemas";
import type { components } from "../../src/api/generated/public";
import { conforming } from "../../src/test/builders";
import { operationOf } from "../../src/test/fake-server";

type Schemas = components["schemas"];

const API_VERSION = "contractor.public.v1";
export const CSRF_TOKEN = "a".repeat(43);

// The Server's CORS policy (internal/httpapi/public/auth_handlers.go).
const EXPOSED_HEADERS =
  "X-Request-ID, X-Contractor-API-Version, ETag, Content-Disposition";
const ALLOWED_METHODS = "GET, POST, PUT, PATCH, DELETE";
const ALLOWED_HEADERS =
  "Authorization, Content-Type, Idempotency-Key, If-Match, If-None-Match, X-CSRF-Token";

/** The origin of the UI under test, from the configured baseURL. */
export function uiOrigin(testInfo: TestInfo): string {
  const baseURL = testInfo.project.use.baseURL;
  if (typeof baseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  return new URL(baseURL).origin;
}

/**
 * The API origin the served UI is configured for (its Content-Security-Policy
 * admits no other), or a loopback stand-in when the spec runs against a
 * development server. The mock intercepts it; nothing listens there.
 */
export function mockedAPIOrigin(): string {
  return process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
}

/** A signed-in browser session; `username` doubles as the user ID. */
export function authSession(
  username = "browser",
  capabilities: Schemas["AuthPrincipal"]["capabilities"] = [
    "user",
    "operations",
  ],
): Schemas["AuthSession"] {
  return conforming("AuthSession", {
    principal: { userId: username, username, capabilities },
    csrfToken: CSRF_TOKEN,
    idleExpiresAt: "2099-01-01T00:00:00Z",
    absoluteExpiresAt: "2099-01-02T00:00:00Z",
  });
}

/** The headers the Server sends with every /v1 answer to `request`. */
function responseHeaders(
  request: Request,
  contentType: string | undefined,
  extra: Record<string, string>,
): Record<string, string> {
  const origin = request.headers()["origin"];
  return {
    "cache-control": "no-store",
    "x-contractor-api-version": API_VERSION,
    ...(contentType === undefined ? {} : { "content-type": contentType }),
    ...(origin === undefined
      ? {}
      : {
          "access-control-allow-origin": origin,
          "access-control-allow-credentials": "true",
          "access-control-expose-headers": EXPOSED_HEADERS,
        }),
    ...extra,
  };
}

/** The operation `request` addresses, or a thrown error naming the request. */
function operationFor(request: Request): ResponseOperation {
  const { pathname } = new URL(request.url());
  const matched = operationOf(request.method(), pathname);
  if (matched === undefined) {
    throw new Error(
      `Mock answered ${request.method()} ${pathname}, which matches no operation of the public contract`,
    );
  }
  return matched.operation;
}

function contractError(
  operation: ResponseOperation,
  status: number,
  violation: string,
): Error {
  return new Error(
    `Mock response to ${operation} ${status} breaks the public contract: ${violation}`,
  );
}

/**
 * Refuses a status `operation` does not declare and, for a success status,
 * a body that breaks its response schema (`body` undefined: no content).
 * Error bodies are not checked, as the UI reads them leniently.
 */
function requireConforming(
  operation: ResponseOperation,
  status: number,
  body: unknown,
): void {
  const declared: Readonly<Record<string, string | null>> =
    RESPONSE_SCHEMAS[operation];
  if (!(String(status) in declared)) {
    throw contractError(operation, status, `status ${status} is not declared`);
  }
  if (status >= 300) return;
  const violation = responseViolation(operation, status, body);
  if (violation !== undefined) {
    throw contractError(operation, status, violation);
  }
}

/** Serves `body` as the operation's JSON answer after checking it. */
export async function fulfillJSON(
  route: Route,
  body: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Promise<void> {
  const request = route.request();
  // Checked as the browser receives it: members holding undefined vanish.
  const text = JSON.stringify(body);
  requireConforming(operationFor(request), status, JSON.parse(text) as unknown);
  await route.fulfill({
    status,
    body: text,
    headers: responseHeaders(request, "application/json", headers),
  });
}

/** Serves a conforming Error envelope with `status`. */
export async function fulfillError(
  route: Route,
  status: number,
  code: string,
  message = code,
  extra: Partial<Schemas["Error"]> = {},
): Promise<void> {
  await fulfillJSON(
    route,
    conforming("Error", {
      code,
      message,
      retryable: false,
      requestId: "request-e2e",
      ...extra,
    }),
    status,
  );
}

/**
 * The answer to a read the spec does not care about, as the unit-test fake
 * Server gives it: an empty page where the operation answers with a page,
 * otherwise a 404 Error envelope.
 */
export async function fulfillEmptyPageOrNotFound(
  route: Route,
  message = "not found",
): Promise<void> {
  const request = route.request();
  const empty = { items: [], page: { hasMore: false } };
  if (
    request.method() === "GET" &&
    responseViolation(operationFor(request), 200, empty) === undefined
  ) {
    await fulfillJSON(route, empty);
    return;
  }
  await fulfillError(route, 404, "not_found", message);
}

/** Serves a declared answer without content, such as 204. */
export async function fulfillEmpty(
  route: Route,
  status = 204,
  headers: Record<string, string> = {},
): Promise<void> {
  const request = route.request();
  requireConforming(operationFor(request), status, undefined);
  await route.fulfill({
    status,
    headers: responseHeaders(request, undefined, headers),
  });
}

/**
 * Serves content that is not JSON (Artifact bytes, archives) for an
 * operation that declares no JSON body for `status`.
 */
export async function fulfillContent(
  route: Route,
  body: string | Buffer,
  contentType: string,
  status = 200,
  headers: Record<string, string> = {},
): Promise<void> {
  const request = route.request();
  const operation = operationFor(request);
  const declared: Readonly<Record<string, string | null>> =
    RESPONSE_SCHEMAS[operation];
  if (declared[String(status)] !== null) {
    throw contractError(
      operation,
      status,
      `status ${status} does not declare a non-JSON body`,
    );
  }
  await route.fulfill({
    status,
    body,
    headers: responseHeaders(request, contentType, {
      "content-length": String(Buffer.byteLength(body)),
      ...headers,
    }),
  });
}

/**
 * Serves `response` (built with the unit-test fake Server helpers in
 * src/test) after checking it as fulfillJSON does.
 */
export async function fulfillResponse(
  route: Route,
  response: Response,
): Promise<void> {
  const headers: Record<string, string> = {};
  response.headers.forEach((value, name) => {
    if (name !== "content-type") headers[name] = value;
  });
  if (response.status === 204) {
    await fulfillEmpty(route, 204, headers);
    return;
  }
  await fulfillJSON(
    route,
    (await response.json()) as unknown,
    response.status,
    headers,
  );
}

/** The runtime configuration the static UI server would serve. */
export async function routeRuntimeConfig(
  page: Page,
  apiBaseUrl?: string,
  supportedApiVersions: readonly string[] = [API_VERSION],
): Promise<void> {
  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: packageMetadata.version,
        supportedApiVersions,
        apiBaseUrl: apiBaseUrl ?? new URL(route.request().url()).origin,
      },
    }),
  );
}

interface MockAPIOptions {
  /** The API origin the runtime configuration names; the UI origin if unset. */
  apiOrigin?: string;
  /** The answer to GET /v1/auth/session. */
  session?: Schemas["AuthSession"];
}

/**
 * What a handler returns for a request it leaves unanswered on purpose, to
 * hold the UI in its waiting state.
 */
export const UNANSWERED = Symbol("unanswered");

/** Answers one /v1 request; it settles the route or returns UNANSWERED. */
type APIHandler = (
  route: Route,
  request: Request,
  url: URL,
) => Promise<unknown> | unknown;

/**
 * Serves the runtime configuration and mocks the Public API at `apiOrigin`:
 * CORS preflights and the session are answered here, every other request by
 * `handle`. A request `handle` leaves unsettled without returning UNANSWERED
 * fails the test.
 */
export async function mockAPI(
  page: Page,
  handle: APIHandler,
  options: MockAPIOptions = {},
): Promise<void> {
  const session = options.session ?? authSession();
  await routeRuntimeConfig(page, options.apiOrigin);
  await page.route(`${options.apiOrigin ?? "**"}/v1/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": request.headers()["origin"] ?? "*",
          "access-control-allow-credentials": "true",
          "access-control-allow-methods": ALLOWED_METHODS,
          "access-control-allow-headers": ALLOWED_HEADERS,
          "access-control-max-age": "600",
        },
      });
      return;
    }
    if (url.pathname === "/v1/auth/session" && request.method() === "GET") {
      await fulfillJSON(route, session);
      return;
    }
    let settled = false;
    const tracked = new Proxy(route, {
      get(target, key) {
        const value: unknown = Reflect.get(target, key, target);
        if (typeof value !== "function") return value;
        return (...args: unknown[]) => {
          if (
            key === "fulfill" ||
            key === "abort" ||
            key === "fallback" ||
            key === "continue"
          ) {
            settled = true;
          }
          return (value as (...values: unknown[]) => unknown).apply(
            target,
            args,
          );
        };
      },
    });
    const result = await handle(tracked, request, url);
    if (!settled && result !== UNANSWERED) {
      throw new Error(
        `No mock answers ${request.method()} ${url.pathname}${url.search}`,
      );
    }
  });
}
