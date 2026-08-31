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
import { APICompatibilityError, PublicAPIError, publicAPIError } from "./error";
import type { components, paths } from "./generated/public";

const MAXIMUM_ERROR_BODY_BYTES = 64 * 1024;
const SAFE_HEADER_VALUE = /^[\x21-\x7E]{1,256}$/;

export type AuthSession = components["schemas"]["AuthSession"];
export type LoginRequest = components["schemas"]["LoginRequest"];

export interface MutationHeaders {
  idempotencyKey?: string;
  ifMatch?: string;
  ifNoneMatch?: string;
}

export type DirectRequestInit = Omit<RequestInit, "credentials">;

export class CSRFMemoryStore {
  #value: string | undefined;

  get(): string | undefined {
    return this.#value;
  }

  replace(value: string): void {
    this.#value = value;
  }

  clear(): void {
    this.#value = undefined;
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
    const checkedFetch = async (request: Request): Promise<Response> => {
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
      let response: Response;
      try {
        response = await fetchImplementation(outgoingRequest);
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
      if (
        values === undefined ||
        values.length !== 1 ||
        values[0] !== PUBLIC_API_VERSION
      ) {
        throw new APICompatibilityError(values?.[0]);
      }
      return boundedErrorResponse(response);
    };
    this.#fetch = checkedFetch;
    this.#client = createClient<paths>({
      baseUrl: apiOrigin,
      credentials: "include",
      fetch: checkedFetch,
      headers: { Accept: "application/json" },
    });
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
      throw publicAPIError(0, {
        code: "invalid_api_response",
        message: "Public API returned an unreadable response",
        retryable: false,
      });
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
    const result = await this.request((client) =>
      client.GET("/v1/auth/session"),
    );
    if (result.data !== undefined) {
      this.csrf.replace(result.data.csrfToken);
      return result.data;
    }
    if (result.response.status === 401) {
      this.csrf.clear();
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
    const headers = this.mutationHeaders();
    const csrf = headers.get("X-CSRF-Token");
    if (csrf === null) {
      throw new TypeError("An authenticated CSRF token is required");
    }
    const result = await this.request((client) =>
      client.POST("/v1/auth/logout", {
        params: {
          header: {
            Origin: globalThis.location.origin,
            "X-CSRF-Token": csrf,
          },
        },
      }),
    );
    if (result.error !== undefined) {
      throw publicAPIError(result.response.status, result.error);
    }
    this.csrf.clear();
  }
}
