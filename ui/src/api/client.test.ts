import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { APICompatibilityError, PublicAPIError } from "./error";
import { PublicAPI } from "./client";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"] as const,
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-08-31T20:00:00Z",
  absoluteExpiresAt: "2026-09-01T12:00:00Z",
};

function apiResponse(body: unknown, status = 200): Response {
  return new Response(body === undefined ? undefined : JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      "X-Request-ID": "request-test",
    },
  });
}

describe("PublicAPI", () => {
  it("uses direct credentialed requests and keeps CSRF only in memory", async () => {
    const requests: Request[] = [];
    const fetchImplementation = vi.fn<typeof fetch>(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request);
      if (request.url.endsWith("/v1/auth/session")) {
        return apiResponse(session);
      }
      return apiResponse(undefined, 204);
    });
    const api = new PublicAPI(runtimeConfig, fetchImplementation);

    await expect(api.getSession()).resolves.toEqual(session);
    expect(requests[0]?.url).toBe("http://127.0.0.1:8080/v1/auth/session");
    expect(requests[0]?.credentials).toBe("include");
    expect(api.csrf.get()).toBe(session.csrfToken);

    await api.logout();
    expect(requests[1]?.method).toBe("POST");
    expect(requests[1]?.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
    expect(api.csrf.get()).toBeUndefined();
  });

  it("returns null for an authoritative 401 and clears stale CSRF", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        apiResponse(
          {
            code: "unauthorized",
            message: "authentication is required",
            retryable: false,
            requestId: "request-test",
          },
          401,
        ),
      ),
    );
    api.csrf.replace("stale");
    await expect(api.getSession()).resolves.toBeNull();
    expect(api.csrf.get()).toBeUndefined();
  });

  it("fails explicitly when the Server omits its compatibility signal", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () => new Response("{}", { status: 401 })),
    );
    await expect(api.getSession()).rejects.toBeInstanceOf(
      APICompatibilityError,
    );
  });

  it("bounds an oversized Server error before mapping it", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(
        async () =>
          new Response("x".repeat(70 * 1024), {
            status: 500,
            headers: {
              "X-Contractor-API-Version": "contractor.public.v1",
              "X-Request-ID": "request-large",
            },
          }),
      ),
    );
    const error = await api.getSession().catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(PublicAPIError);
    expect((error as PublicAPIError).code).toBe("invalid_error_response");
    expect((error as PublicAPIError).message).toContain("exceeded");
  });

  it("requires CSRF and validates caller-provided mutation headers", () => {
    const api = new PublicAPI(runtimeConfig, vi.fn());
    expect(() => api.mutationHeaders()).toThrow("CSRF");
    api.csrf.replace(session.csrfToken);
    const headers = api.mutationHeaders({
      idempotencyKey: "draft-123",
      ifMatch: '"revision-4"',
      ifNoneMatch: "*",
    });
    expect(headers.get("Idempotency-Key")).toBe("draft-123");
    expect(headers.get("If-Match")).toBe('"revision-4"');
    expect(headers.get("If-None-Match")).toBe("*");
    expect(() => api.mutationHeaders({ idempotencyKey: "bad\nvalue" })).toThrow(
      "visible ASCII",
    );
  });

  it("never sends cookies or CSRF outside the configured API boundary", async () => {
    const fetchImplementation = vi.fn<typeof fetch>();
    const api = new PublicAPI(runtimeConfig, fetchImplementation);
    api.csrf.replace(session.csrfToken);
    await expect(
      api.request((client) =>
        client.GET("/v1/auth/session", {
          baseUrl: "https://attacker.invalid",
        }),
      ),
    ).rejects.toMatchObject({ code: "invalid_client_request" });
    expect(fetchImplementation).not.toHaveBeenCalled();
  });

  it("maps malformed response JSON to a safe fallback error", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(
        async () =>
          new Response("not-json", {
            status: 500,
            headers: {
              "content-type": "application/json",
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          }),
      ),
    );
    await expect(api.getSession()).rejects.toMatchObject({
      code: "request_failed",
      message: "Public API request failed",
    });
  });

  it("fences direct binary requests and attaches the current CSRF", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return apiResponse(
          {
            artifact: {
              namespace: "projects",
              name: "source",
              revision: "revision-1",
            },
            mediaType: "application/zip",
            size: 3,
          },
          201,
        );
      }),
    );
    api.csrf.replace(session.csrfToken);
    const response = await api.fetch("/v1/artifacts/projects/source", {
      method: "PUT",
      headers: {
        "Content-Type": "application/zip",
        "If-None-Match": "*",
      },
      body: new Blob(["zip"]),
    });
    expect(response.status).toBe(201);
    expect(captured?.credentials).toBe("include");
    expect(captured?.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
    expect(captured?.headers.get("If-None-Match")).toBe("*");

    await expect(
      api.fetch("https://attacker.invalid/v1/artifacts/x/y"),
    ).rejects.toMatchObject({ code: "invalid_client_request" });
    await expect(
      api.fetch("/v1/artifacts/x/y", {
        headers: { Authorization: "Bearer forbidden" },
      }),
    ).rejects.toMatchObject({ code: "invalid_client_request" });
  });
});
