import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import { getSchedulerSettings, replaceSchedulerSettings } from "./operations";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function settingsResponse(
  maxConcurrentRuns: number,
  revision: string,
  overrides: Record<string, unknown> = {},
  headers: Record<string, string> = {},
): Response {
  return new Response(
    JSON.stringify({
      maxConcurrentRuns,
      maxConcurrentToolRuns: 2,
      revision,
      updatedAt: "2026-09-06T01:00:00Z",
      ...overrides,
    }),
    {
      headers: {
        "content-type": "application/json",
        "X-Contractor-API-Version": "contractor.public.v1",
        "Cache-Control": "no-store",
        ETag: `"${revision}"`,
        ...headers,
      },
    },
  );
}

describe("Scheduler settings API", () => {
  it("carries the exact GET ETag into a typed CAS replacement", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        return request.method === "GET"
          ? settingsResponse(1, "7")
          : settingsResponse(1, "8", { maxConcurrentToolRuns: 5 });
      }),
    );
    api.csrf.replace("a".repeat(43));

    const current = await getSchedulerSettings(api);
    expect(current).toEqual({
      resource: {
        maxConcurrentRuns: 1,
        maxConcurrentToolRuns: 2,
        revision: "7",
        updatedAt: "2026-09-06T01:00:00Z",
      },
      etag: '"7"',
    });
    // Changing one limit still sends both, so nothing unseen is kept.
    await expect(
      replaceSchedulerSettings(
        api,
        { maxConcurrentRuns: 1, maxConcurrentToolRuns: 5 },
        current.etag,
      ),
    ).resolves.toMatchObject({
      resource: {
        maxConcurrentRuns: 1,
        maxConcurrentToolRuns: 5,
        revision: "8",
      },
      etag: '"8"',
    });

    expect(new URL(requests[0]!.url).pathname).toBe(
      "/v1/operations/settings/scheduler",
    );
    expect(requests[1]?.method).toBe("PUT");
    expect(requests[1]?.headers.get("If-Match")).toBe('"7"');
    expect(requests[1]?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(requests[1]?.json()).resolves.toEqual({
      maxConcurrentRuns: 1,
      maxConcurrentToolRuns: 5,
    });
  });

  it.each([
    ["maxConcurrentRuns", 0],
    ["maxConcurrentRuns", 1.5],
    ["maxConcurrentRuns", 33],
    ["maxConcurrentRuns", Number.NaN],
    ["maxConcurrentToolRuns", 0],
    ["maxConcurrentToolRuns", 1.5],
    ["maxConcurrentToolRuns", 33],
    ["maxConcurrentToolRuns", Number.NaN],
  ])(
    "rejects invalid client %s %s before issuing a request",
    async (name, value) => {
      const fetcher = vi.fn(async () => settingsResponse(1, "1"));
      const api = new PublicAPI(runtimeConfig, fetcher);
      api.csrf.replace("a".repeat(43));
      await expect(
        replaceSchedulerSettings(
          api,
          { maxConcurrentRuns: 1, maxConcurrentToolRuns: 2, [name]: value },
          '"1"',
        ),
      ).rejects.toBeInstanceOf(TypeError);
      expect(fetcher).not.toHaveBeenCalled();
    },
  );

  it.each([[{ maxConcurrentRuns: 3 }], [{ maxConcurrentToolRuns: 3 }]])(
    "rejects a replacement answered with other values %o",
    async (answered) => {
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async () => settingsResponse(1, "2", answered)),
      );
      api.csrf.replace("a".repeat(43));
      await expect(
        replaceSchedulerSettings(
          api,
          { maxConcurrentRuns: 1, maxConcurrentToolRuns: 2 },
          '"1"',
        ),
      ).rejects.toMatchObject({ code: "invalid_api_response" });
    },
  );

  it("rejects malformed or inconsistent Server representations", async () => {
    for (const response of [
      settingsResponse(1, "01"),
      settingsResponse(1, "1", { unknown: true }),
      settingsResponse(1, "1", {}, { ETag: '"2"' }),
      settingsResponse(1, "1", {}, { "Cache-Control": "private" }),
      settingsResponse(33, "1"),
      settingsResponse(1, "1", { maxConcurrentToolRuns: 0 }),
      settingsResponse(1, "1", { maxConcurrentToolRuns: undefined }),
    ]) {
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async () => response),
      );
      await expect(getSchedulerSettings(api)).rejects.toMatchObject({
        code: "invalid_api_response",
      });
    }
  });
});
