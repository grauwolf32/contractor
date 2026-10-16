import createClient, { type Client } from "openapi-fetch";

import {
  PUBLIC_API_VERSION,
  PUBLIC_API_VERSION_HEADER,
  SUPPORTED_API_VERSIONS,
} from "../build";
import {
  type RuntimeConfig,
  validateAPIBaseURL,
} from "../config/runtime-config";
import {
  isResponseOperation,
  requestHeaderViolation,
  responseViolation,
  type ResponseOperation,
} from "./contract";
import {
  APICompatibilityError,
  InvalidResponseError,
  PublicAPIError,
  publicAPIError,
} from "./error";
import type { components, operations, paths } from "./generated/public";

const MAXIMUM_ERROR_BODY_BYTES = 64 * 1024;
const SAFE_HEADER_VALUE = /^[\x21-\x7E]{1,256}$/;

export type AuthSession = components["schemas"]["AuthSession"];
export type LoginRequest = components["schemas"]["LoginRequest"];

interface MutationHeaders {
  idempotencyKey?: string;
  ifMatch?: string;
  ifNoneMatch?: string;
}

type DirectRequestInit = Omit<RequestInit, "credentials">;

class CSRFMemoryStore {
  #value: string | undefined;
  #generation = 0;

  get generation(): number {
    return this.#generation;
  }

  get(): string | undefined {
    return this.#value;
  }

  replace(value: string): void {
    this.#value = value;
    this.#generation += 1;
  }

  clear(): void {
    this.#value = undefined;
    this.#generation += 1;
  }
}

async function boundedErrorResponse(response: Response): Promise<Response> {
  if (response.ok || response.body === null) {
    return response;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  let tooLarge = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    size += value.byteLength;
    if (size > MAXIMUM_ERROR_BODY_BYTES) {
      tooLarge = true;
      await reader.cancel();
      break;
    }
    chunks.push(value);
  }
  const headers = new Headers(response.headers);
  headers.delete("content-length");
  if (tooLarge) {
    headers.set("content-type", "application/json");
    return new Response(
      JSON.stringify({
        code: "invalid_error_response",
        message: "Server error response exceeded the client limit",
        retryable: false,
        requestId: headers.get("x-request-id") ?? "unknown",
      }),
      { status: response.status, statusText: response.statusText, headers },
    );
  }
  const body = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

/**
 * Throws the invalid-response error unless `body` is what `operation`
 * declares for `status`. Every JSON success response passes here: the
 * generated client through validatingClient, direct requests through
 * PublicAPI.readJSON.
 */
function requireContractResponse(
  operation: ResponseOperation,
  status: number,
  body: unknown,
): void {
  const violation = responseViolation(operation, status, body);
  if (violation !== undefined) {
    throw new InvalidResponseError(
      status,
      `Server response to ${operation} broke the public contract: ${violation}`,
    );
  }
}

const CLIENT_METHODS = ["GET", "PUT", "POST", "DELETE", "PATCH"] as const;

type ClientInit = {
  parseAs?: string;
  params?: { header?: Record<string, unknown> };
};

type ClientCall = (
  path: string,
  init?: ClientInit,
) => Promise<{ data?: unknown; error?: unknown; response: Response }>;

/**
 * Refuses, before anything is sent, a request header value (If-Match,
 * Idempotency-Key, If-None-Match) that breaks the schema its operation
 * declares for it.
 */
function requireContractHeaders(
  operation: ResponseOperation,
  headers: Record<string, unknown> | undefined,
): void {
  for (const [name, value] of Object.entries(headers ?? {})) {
    const violation = requestHeaderViolation(operation, name, value);
    if (violation !== undefined) {
      throw publicAPIError(0, {
        code: "invalid_client_request",
        message: `${name} for ${operation} breaks the public contract at ${violation}`,
        retryable: false,
      });
    }
  }
}

/**
 * Wraps the generated client so that declared request headers are checked
 * before sending and every parsed JSON success body is validated against
 * its operation's response schema before a wrapper sees it. Bodies read as
 * text, blob or stream are left to their callers.
 */
function validatingClient(client: Client<paths>): Client<paths> {
  const wrapped: Record<string, unknown> = { ...client };
  for (const method of CLIENT_METHODS) {
    const call = client[method] as unknown as ClientCall;
    wrapped[method] = async (path: string, init?: ClientInit) => {
      const operation = `${method} ${path}`;
      if (!isResponseOperation(operation)) {
        throw publicAPIError(0, {
          code: "invalid_client_request",
          message: `${operation} is not in the generated public contract`,
          retryable: false,
        });
      }
      requireContractHeaders(operation, init?.params?.header);
      const result = await call(path, init);
      if (result.response.ok && (init?.parseAs ?? "json") === "json") {
        requireContractResponse(operation, result.response.status, result.data);
      }
      return result;
    };
  }
  return wrapped as unknown as Client<paths>;
}

function exactHeaderValue(name: string, value: string): string {
  if (!SAFE_HEADER_VALUE.test(value)) {
    throw new TypeError(
      `${name} must contain 1 through 256 visible ASCII bytes`,
    );
  }
  return value;
}

export class PublicAPI {
  readonly #client: Client<paths>;
  readonly #fetch: (request: Request) => Promise<Response>;
  readonly #apiOrigin: string;
  readonly #unauthorizedListeners = new Set<() => void>();
  readonly csrf = new CSRFMemoryStore();

  constructor(
    runtimeConfig: RuntimeConfig,
    fetchImplementation: typeof fetch = globalThis.fetch,
  ) {
    if (
      !runtimeConfig.supportedApiVersions.includes(PUBLIC_API_VERSION) ||
      !SUPPORTED_API_VERSIONS.includes(PUBLIC_API_VERSION)
    ) {
      throw new APICompatibilityError();
    }
    const apiOrigin = validateAPIBaseURL(runtimeConfig.apiBaseUrl);
    this.#apiOrigin = apiOrigin;
    const send = async (request: Request): Promise<Response> => {
      let response: Response;
      try {
        response = await fetchImplementation(request);
      } catch {
        throw publicAPIError(0, {
          code: "network_error",
          message: "Public API is unavailable",
          retryable: true,
        });
      }
      const values = response.headers
        .get(PUBLIC_API_VERSION_HEADER)
        ?.split(",")
        .map((value) => value.trim());
      if (values === undefined && response.status >= 500) {
        await response.body?.cancel().catch(() => undefined);
        throw publicAPIError(response.status, {
          code: "server_unavailable",
          message: "Server is temporarily unavailable. Try again.",
          retryable: true,
        });
      }
      if (
        values === undefined ||
        values.length !== 1 ||
        values[0] !== PUBLIC_API_VERSION
      ) {
        throw new APICompatibilityError(values?.[0]);
      }
      return response;
    };
    const checkedFetch = async (request: Request): Promise<Response> => {
      const csrfAtStart = this.csrf.get();
      const requestURL = new URL(request.url);
      if (
        requestURL.origin !== apiOrigin ||
        !requestURL.pathname.startsWith("/v1/")
      ) {
        throw publicAPIError(0, {
          code: "invalid_client_request",
          message: "Public API request escaped the configured Server boundary",
          retryable: false,
        });
      }
      let outgoingRequest = request;
      const unsafe = !["GET", "HEAD", "OPTIONS"].includes(request.method);
      const isLogin =
        request.method === "POST" && requestURL.pathname === "/v1/auth/login";
      if (unsafe && !isLogin) {
        const csrf = this.csrf.get();
        if (csrf === undefined) {
          throw new TypeError("An authenticated CSRF token is required");
        }
        const headers = new Headers(request.headers);
        headers.set("X-CSRF-Token", csrf);
        outgoingRequest = new Request(request, { headers });
      }
      const replayable =
        unsafe &&
        !isLogin &&
        ["Idempotency-Key", "If-Match", "If-None-Match"].some((name) =>
          outgoingRequest.headers.has(name),
        );
      const replayRequest = replayable ? outgoingRequest.clone() : undefined;
      let response = await send(outgoingRequest);
      let csrfAtAttempt = csrfAtStart;
      if (
        response.status === 403 &&
        unsafe &&
        !isLogin &&
        csrfAtStart === this.csrf.get()
      ) {
        // Another tab can replace the shared cookie while this tab still has
        // the previous session's token. A 403 alone is not proof of that: the
        // Server also uses it for origin and capability denials.
        let refreshed: AuthSession | null = null;
        try {
          refreshed = await this.getSession();
        } catch {
          // Preserve the original 403 when the session read is unavailable.
        }
        const freshToken = this.csrf.get();
        if (
          refreshed !== null &&
          freshToken === refreshed.csrfToken &&
          freshToken !== csrfAtStart
        ) {
          await response.body?.cancel().catch(() => undefined);
          if (replayRequest === undefined) {
            throw publicAPIError(403, {
              code: "session_refreshed",
              message:
                "Your session changed in another tab. Try this action again.",
              retryable: false,
            });
          }
          const headers = new Headers(replayRequest.headers);
          headers.set("X-CSRF-Token", freshToken);
          csrfAtAttempt = freshToken;
          response = await send(new Request(replayRequest, { headers }));
        }
      }
      if (
        response.status === 401 &&
        !isLogin &&
        requestURL.pathname !== "/v1/auth/session" &&
        csrfAtAttempt === this.csrf.get()
      ) {
        this.csrf.clear();
        for (const listener of this.#unauthorizedListeners) {
          listener();
        }
      }
      return boundedErrorResponse(response);
    };
    this.#fetch = checkedFetch;
    this.#client = validatingClient(
      createClient<paths>({
        baseUrl: apiOrigin,
        credentials: "include",
        fetch: checkedFetch,
        headers: { Accept: "application/json" },
      }),
    );
  }

  get apiBaseUrl(): string {
    return this.#apiOrigin;
  }

  subscribeUnauthorized(listener: () => void): () => void {
    this.#unauthorizedListeners.add(listener);
    return () => {
      this.#unauthorizedListeners.delete(listener);
    };
  }

  async request<T>(
    operation: (client: Client<paths>) => Promise<T>,
  ): Promise<T> {
    return this.#safeRequest(() => operation(this.#client));
  }

  async fetch(path: string, init: DirectRequestInit = {}): Promise<Response> {
    if (!path.startsWith("/v1/") || path.startsWith("//")) {
      throw publicAPIError(0, {
        code: "invalid_client_request",
        message: "Direct API path must be repository-relative public /v1",
        retryable: false,
      });
    }
    const url = new URL(path, this.#apiOrigin);
    if (url.origin !== this.#apiOrigin || url.hash !== "") {
      throw publicAPIError(0, {
        code: "invalid_client_request",
        message: "Direct API request escaped the configured Server boundary",
        retryable: false,
      });
    }
    const headers = new Headers(init.headers);
    if (headers.has("Authorization") || headers.has("Cookie")) {
      throw publicAPIError(0, {
        code: "invalid_client_request",
        message: "Browser API requests cannot supply credential headers",
        retryable: false,
      });
    }
    const request = new Request(url, {
      ...init,
      headers,
      credentials: "include",
    });
    return this.#safeRequest(() => this.#fetch(request));
  }

  /**
   * Reads a direct request's JSON success body, bounded to `maximumBytes`,
   * and validates it as `operation`'s response. Direct requests (uploads and
   * bounded previews) that answer with JSON must read it here.
   */
  async readJSON<T>(
    operation: ResponseOperation,
    response: Response,
    maximumBytes: number,
  ): Promise<T> {
    const unreadable = () =>
      new InvalidResponseError(
        response.status,
        `Server response to ${operation} was unreadable`,
      );
    if (
      response.headers.get("Content-Type")?.split(";")[0]?.trim() !==
        "application/json" ||
      response.body === null
    ) {
      await response.body?.cancel().catch(() => undefined);
      throw unreadable();
    }
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let length = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > maximumBytes) {
          await reader.cancel();
          throw new InvalidResponseError(
            response.status,
            `Server response to ${operation} exceeded the client limit`,
          );
        }
        chunks.push(value);
      }
    } finally {
      reader.releaseLock();
    }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    let body: unknown;
    try {
      body = JSON.parse(
        new TextDecoder("utf-8", { fatal: true }).decode(bytes),
      );
    } catch {
      throw unreadable();
    }
    requireContractResponse(operation, response.status, body);
    return body as T;
  }

  async error(response: Response): Promise<PublicAPIError> {
    let value: unknown;
    try {
      value = await response.json();
    } catch {
      value = undefined;
    }
    return publicAPIError(response.status, value);
  }

  async #safeRequest<T>(operation: () => Promise<T>): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      if (error instanceof PublicAPIError) {
        throw error;
      }
      throw new InvalidResponseError(
        0,
        "Public API returned an unreadable response",
      );
    }
  }

  mutationHeaders(options: MutationHeaders = {}): Headers {
    const headers = new Headers();
    const csrf = this.csrf.get();
    if (csrf === undefined) {
      throw new TypeError("An authenticated CSRF token is required");
    }
    headers.set("X-CSRF-Token", csrf);
    if (options.idempotencyKey !== undefined) {
      headers.set(
        "Idempotency-Key",
        exactHeaderValue("Idempotency-Key", options.idempotencyKey),
      );
    }
    if (options.ifMatch !== undefined) {
      headers.set("If-Match", exactHeaderValue("If-Match", options.ifMatch));
    }
    if (options.ifNoneMatch !== undefined) {
      headers.set(
        "If-None-Match",
        exactHeaderValue("If-None-Match", options.ifNoneMatch),
      );
    }
    return headers;
  }

  async getSession(): Promise<AuthSession | null> {
    const generation = this.csrf.generation;
    const result = await this.request((client) =>
      client.GET("/v1/auth/session"),
    );
    // Cancelling a query does not cancel side effects of its pending request.
    // A late lookup must not change a newer login, logout or session expiry.
    if (result.data !== undefined) {
      if (generation === this.csrf.generation) {
        this.csrf.replace(result.data.csrfToken);
      }
      return result.data;
    }
    if (result.response.status === 401) {
      if (generation === this.csrf.generation) {
        this.csrf.clear();
      }
      return null;
    }
    throw publicAPIError(result.response.status, result.error);
  }

  async login(request: LoginRequest): Promise<AuthSession> {
    const result = await this.request((client) =>
      client.POST("/v1/auth/login", { body: request }),
    );
    if (result.data === undefined) {
      throw publicAPIError(result.response.status, result.error);
    }
    this.csrf.replace(result.data.csrfToken);
    return result.data;
  }

  async logout(): Promise<void> {
    if (this.csrf.get() === undefined) {
      throw new TypeError("An authenticated CSRF token is required");
    }
    // The browser sets Origin and the checked fetch adds X-CSRF-Token.
    const header = {} as operations["logout"]["parameters"]["header"];
    const result = await this.request((client) =>
      client.POST("/v1/auth/logout", { params: { header } }),
    );
    if (result.error !== undefined) {
      throw publicAPIError(result.response.status, result.error);
    }
    this.csrf.clear();
  }
}
