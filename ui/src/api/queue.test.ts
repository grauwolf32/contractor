import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  getOwnerQueueControl,
  listRunQueue,
  setOwnerQueuePaused,
} from "./queue";

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
  it("reads and updates the durable owner control with exact ETags", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const paused = request.method === "PUT";
        return new Response(
          JSON.stringify({
            paused,
            revision: paused ? "1" : "0",
            ...(paused ? { updatedAt: "2026-09-05T08:00:00Z" } : {}),
          }),
          {
            headers: {
              "content-type": "application/json",
              "X-Contractor-API-Version": "contractor.public.v1",
              ETag: paused ? '"1"' : '"0"',
            },
          },
        );
      }),
    );
    api.csrf.replace("a".repeat(43));

    await expect(getOwnerQueueControl(api)).resolves.toEqual({
      paused: false,
      revision: "0",
    });
    await expect(setOwnerQueuePaused(api, true, "0")).resolves.toMatchObject({
      paused: true,
      revision: "1",
    });
    expect(new URL(requests[0]!.url).pathname).toBe("/v1/queue/control");
    expect(requests[1]?.method).toBe("PUT");
    expect(requests[1]?.headers.get("If-Match")).toBe('"0"');
    expect(requests[1]?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(requests[1]?.json()).resolves.toEqual({ paused: true });
  });

  it("rejects a mismatched Queue control representation", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(
        async () =>
          new Response(JSON.stringify({ paused: true, revision: "1" }), {
            headers: {
              "content-type": "application/json",
              "X-Contractor-API-Version": "contractor.public.v1",
              ETag: '"2"',
            },
          }),
      ),
    );
    await expect(getOwnerQueueControl(api)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
  });

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
