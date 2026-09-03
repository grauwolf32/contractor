import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  cancelRun,
  downloadRunArtifact,
  getRun,
  getRunArtifactLineage,
  getRunArtifactMetadata,
  listRunArtifacts,
  listRunArtifactVersions,
  listRuns,
} from "./runs";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function response(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(
    typeof value === "string" ? value : JSON.stringify(value),
    {
      status,
      headers: {
        "content-type": "application/json",
        "X-Contractor-API-Version": "contractor.public.v1",
        ...headers,
      },
    },
  );
}

describe("Run API", () => {
  it("uses exact list/detail and RunScope Artifact query paths", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const path = new URL(request.url).pathname;
        if (path === "/v1/runs") {
          return response({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/runs/run-1") {
          return response({
            runId: "run-1",
            workflow: "workflow@1",
            state: "running",
            runtimeLabels: [],
            runtimeConfiguration: {
              default: {
                label: "default",
                bindingRevision: "1",
                config: {
                  name: "contractor-empty",
                  version: "1",
                  digest: `sha256:${"0".repeat(64)}`,
                },
              },
              labels: [],
            },
            attempts: [],
            transitions: [],
            outputs: {},
          });
        }
        if (path === "/v1/runs/run-1/artifacts") {
          return response({ items: [], page: { hasMore: false } });
        }
        if (path.endsWith("/metadata")) {
          return response({
            artifact: {
              namespace: "builder",
              name: "openapi",
              revision: "r1",
            },
            mediaType: "application/yaml",
            size: 2,
            current: true,
            frozen: true,
            createdAt: "2026-08-31T12:00:00Z",
          });
        }
        if (path.endsWith("/versions")) {
          return response({ items: [], page: { hasMore: false } });
        }
        if (path.endsWith("/lineage")) {
          return response({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.url}`);
      }),
    );

    await listRuns(api, { state: "running", cursor: "run-next" });
    await getRun(api, "run-1");
    await listRunArtifacts(api, {
      runId: "run-1",
      namespace: "builder",
      cursor: "artifact-next",
    });
    await getRunArtifactMetadata(api, {
      runId: "run-1",
      namespace: "builder",
      name: "openapi",
      revision: "r1",
    });
    await listRunArtifactVersions(api, {
      runId: "run-1",
      namespace: "builder",
      name: "openapi",
      cursor: "version-next",
    });
    await getRunArtifactLineage(api, {
      runId: "run-1",
      namespace: "builder",
      name: "openapi",
      revision: "r1",
      cursor: "lineage-next",
    });

    expect(new URL(requests[0]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({
        limit: "50",
        state: "running",
        cursor: "run-next",
      }),
    );
    expect(requests[1]?.url).toBe("http://127.0.0.1:8080/v1/runs/run-1");
    expect(new URL(requests[2]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({
        limit: "50",
        namespace: "builder",
        cursor: "artifact-next",
      }),
    );
    expect(requests[3]?.url).toContain("revision=r1");
    expect(requests[4]?.url).toContain("cursor=version-next");
    expect(requests[5]?.url).toContain("cursor=lineage-next");
  });

  it("cancels with trimmed reason and accepts only authoritative 200/202", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response({ runId: "run-1", state: "cancelling" }, 202);
      }),
    );
    api.csrf.replace("a".repeat(43));
    await expect(cancelRun(api, "run-1", "  stop safely  ")).resolves.toEqual({
      runId: "run-1",
      state: "cancelling",
    });
    expect(captured?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(captured?.clone().json()).resolves.toEqual({
      reason: "stop safely",
    });
    await expect(cancelRun(api, "run-1", "   ")).rejects.toThrow(
      "Cancellation reason",
    );
  });

  it("downloads one exact RunScope revision through the direct API boundary", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response("ok", 200, {
          "content-type": "text/plain",
          "content-length": "2",
        });
      }),
    );
    const downloaded = await downloadRunArtifact(api, "run-1", {
      artifact: {
        namespace: "outputs",
        name: "result",
        revision: "r2",
      },
      mediaType: "text/plain",
      size: 2,
      current: true,
      frozen: true,
      createdAt: "2026-08-31T12:00:00Z",
    });
    expect(captured?.url).toBe(
      "http://127.0.0.1:8080/v1/runs/run-1/artifacts/outputs/result?revision=r2",
    );
    expect(await downloaded.blob.text()).toBe("ok");
  });
});
