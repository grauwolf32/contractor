import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import { listAwaitingReviewItems } from "./inbox";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function response(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function apiAnswering(value: unknown) {
  const requests: Request[] = [];
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input: RequestInfo | URL) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request);
      return response(value);
    }),
  );
  return { api, requests };
}

describe("Inbox API", () => {
  it("lists only the work items that wait for the owner", async () => {
    const item = { itemId: "item_1", state: "awaiting_review" };
    const { api, requests } = apiAnswering({
      items: [item],
      page: { hasMore: true, nextCursor: "next" },
    });

    await expect(
      listAwaitingReviewItems(api, "audit_1", "cursor_1"),
    ).resolves.toEqual({
      items: [item],
      page: { hasMore: true, nextCursor: "next" },
    });
    const url = new URL(requests[0]!.url);
    expect(url.pathname).toBe("/v1/audits/audit_1/items");
    expect(url.searchParams.get("state")).toBe("awaiting_review");
    expect(url.searchParams.get("limit")).toBe("50");
    expect(url.searchParams.get("cursor")).toBe("cursor_1");
  });

  it("rejects an invalid check ID and a malformed page", async () => {
    const { api, requests } = apiAnswering({ items: "nope" });

    await expect(listAwaitingReviewItems(api, "../audit")).rejects.toThrow(
      "Audit ID is invalid",
    );
    expect(requests).toHaveLength(0);
    await expect(listAwaitingReviewItems(api, "audit_1")).rejects.toThrow(
      "Server returned an invalid Audit item page",
    );
  });
});
