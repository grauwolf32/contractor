import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import { listRunQueue } from "./queue";

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

describe("Queue API", () => {
  it("requests filters and preserves only the closed safe projection", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response({
          items: [
            {
              runId: "run-project",
              project: {
                projectId: "project-example",
                name: "Payment service",
                kind: "project",
              },
              workflow: "openapi-from-source@1",
              state: "running",
              labels: { purpose: "manual" },
              eventCursor: { generation: "events-project", sequence: "7" },
              createdAt: "2026-09-05T08:00:00Z",
              updatedAt: "2026-09-05T08:01:00Z",
            },
          ],
          page: { hasMore: true, nextCursor: "next-queue" },
        });
      }),
    );

    await expect(
      listRunQueue(api, {
        state: "running",
        membership: "project",
        cursor: "cursor-queue",
      }),
    ).resolves.toMatchObject({
      items: [
        {
          runId: "run-project",
          project: { projectId: "project-example", kind: "project" },
          labels: { purpose: "manual" },
          eventCursor: { sequence: "7" },
        },
      ],
      page: { hasMore: true, nextCursor: "next-queue" },
    });
    const url = new URL(captured!.url);
    expect(url.pathname).toBe("/v1/queue");
    expect(url.searchParams.get("limit")).toBe("50");
    expect(url.searchParams.get("state")).toBe("running");
    expect(url.searchParams.get("membership")).toBe("project");
    expect(url.searchParams.get("cursor")).toBe("cursor-queue");
  });

  it("rejects terminal or malformed queue data", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          items: [
            {
              runId: "run-terminal",
              workflow: "workflow@1",
              state: "succeeded",
              labels: {},
              eventCursor: { generation: "events", sequence: "0" },
              createdAt: "2026-09-05T08:00:00Z",
              updatedAt: "2026-09-05T08:01:00Z",
            },
          ],
          page: { hasMore: false },
        }),
      ),
    );
    await expect(listRunQueue(api)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    await expect(
      listRunQueue(api, { state: "succeeded" as never }),
    ).rejects.toThrow("Queue query is invalid");
  });
});
