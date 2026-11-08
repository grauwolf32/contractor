// Public API responses and request matching without a test runner, shared by
// the unit tests' fake Server (src/test/fake-server.ts) and the browser
// specs' mock API (e2e/support/mock-api.ts). Nothing here may import vitest
// or Playwright.
import { responseViolation, type ResponseOperation } from "../api/contract";
import { RESPONSE_SCHEMAS } from "../api/generated/response-schemas";
import type { components } from "../api/generated/public";
import errorRetryability from "./error-retryability.generated.json" with { type: "json" };

export const API_VERSION_HEADERS = {
  "X-Contractor-API-Version": "contractor.public.v1",
};

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

/**
 * Raw bytes as an artifact content operation serves them: `body` in
 * `mediaType`, with its UTF-8 length. A JSON artifact is no API body, so the
 * fake Server does not check it.
 */
export function bytes(
  body: string,
  mediaType: string,
  headers: Record<string, string> = {},
): Response {
  return new Response(body, {
    headers: {
      "Content-Type": mediaType,
      "Content-Length": String(new TextEncoder().encode(body).byteLength),
      ...API_VERSION_HEADERS,
      ...headers,
    },
  });
}

/** Retryability generated from internal/apierror by make docs-error-registry. */
export function errorRetryable(code: string): boolean {
  if (!Object.hasOwn(errorRetryability, code)) {
    throw new Error(`Unregistered public API error code: ${code}`);
  }
  return errorRetryability[code as keyof typeof errorRetryability];
}

/** An error envelope with `status` and the registry's retryability. */
export function failure(
  status: number,
  code = status === 404 ? "not_found" : "internal_error",
  message = "Server failed",
): Response {
  return json(
    {
      code,
      message,
      retryable: errorRetryable(code),
      requestId: "request_test",
    },
    status,
  );
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

/**
 * A fallback for reads a test does not care about: an empty page where the
 * operation answers with a page, otherwise 404. It never serves a body the
 * operation's schema rejects.
 */
export function emptyPageOrNotFound(request: Request, url: URL): Response {
  const cost = unavailableCostForURL(request.method, url);
  if (cost) return json(cost);
  if (
    request.method === "GET" &&
    url.pathname === "/v1/operations/settings/models"
  )
    return json({ configured: false, models: [] });
  const matched = operationOf(request.method, url.pathname);
  const empty = page([]);
  return matched !== undefined &&
    responseViolation(matched.operation, 200, empty) === undefined
    ? json(empty)
    : failure(
        404,
        "not_found",
        `No fake route for ${request.method} ${url.pathname}`,
      );
}

/** Unrelated fixtures have no monetary producer: keep that absence explicit. */
export function unavailableCostForURL(
  method: string,
  url: URL,
): components["schemas"]["LLMCostSummary"] | undefined {
  const match = /^\/v1\/(runs|audits|projects)\/([^/]+)\/llm-cost$/.exec(
    url.pathname,
  );
  if (method !== "GET" || !match) return undefined;
  const scope =
    match[1] === "runs" ? "run" : match[1] === "audits" ? "audit" : "project";
  return {
    scope,
    scopeId: decodeURIComponent(match[2]!),
    currency: "USD",
    amount: null,
    calls: "0",
    pricedCalls: "0",
    reports: "0",
    unknownReports: "0",
    basis: "none",
    coverage: "unavailable",
    pending: false,
    historicalGap: true,
    asOf: "2026-11-08T12:00:00Z",
    updatedAt: null,
    checksAmount: null,
    standaloneAmount: null,
    activity: [],
    models: [],
    executions: url.searchParams.getAll("executionId").map((executionId) => ({
      executionId,
      runId: null,
      amount: null,
      basis: "none",
      coverage: "unavailable",
      pending: false,
    })),
  };
}

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
