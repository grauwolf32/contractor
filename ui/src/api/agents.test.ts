import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { listAgentTemplateWorkflowBindings } from "./agents";
import { PublicAPI } from "./client";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function response(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

describe("Agent catalog API", () => {
  it("reads one exact Agent binding page with cancellation", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response({
          items: [
            {
              workflow: { name: "openapi-from-source", version: "1" },
              stage: "analyze",
              logicalWorker: "builder",
            },
          ],
          page: { hasMore: true, nextCursor: "binding-next" },
        });
      }),
    );
    const abort = new AbortController();

    await expect(
      listAgentTemplateWorkflowBindings(api, "researcher", "v2", {
        cursor: "binding-current",
        signal: abort.signal,
      }),
    ).resolves.toMatchObject({ page: { nextCursor: "binding-next" } });
    const url = new URL(captured!.url);
    expect(url.pathname).toBe(
      "/v1/configurations/agent-templates/researcher/versions/v2/workflow-bindings",
    );
    expect(url.searchParams).toEqual(
      new URLSearchParams({ limit: "50", cursor: "binding-current" }),
    );
    abort.abort();
    expect(captured?.signal.aborted).toBe(true);
  });

  it("fails locally for invalid identities and malformed binding pages", async () => {
    const fetcher = vi.fn(async () =>
      response({ items: undefined, page: { hasMore: false } }),
    );
    const api = new PublicAPI(runtimeConfig, fetcher);
    await expect(
      listAgentTemplateWorkflowBindings(api, "not/a/name", "1"),
    ).rejects.toThrow("Agent version is invalid");
    expect(fetcher).not.toHaveBeenCalled();
    await expect(
      listAgentTemplateWorkflowBindings(api, "researcher", "1"),
    ).rejects.toThrow("invalid Agent Workflow bindings");
  });
});
