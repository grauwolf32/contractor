import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  createRun,
  getWorkflow,
  listConfigurations,
  listCredentials,
  listWorkflows,
  type CreateRunRequest,
} from "./workflows";

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

describe("Workflow API", () => {
  it("uses generated exact and cursor query operations", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const path = new URL(request.url).pathname;
        if (path === "/v1/workflows") {
          return response({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/workflows/openapi-from-workspace/versions/3") {
          return response({
            ref: { name: "openapi-from-workspace", version: "3" },
            entryStage: "discover",
            parameters: {},
            inputs: {},
            outputs: {},
            stages: {},
          });
        }
        if (path === "/v1/configurations/model-policies") {
          return response({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/operations/credentials") {
          return response({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.url}`);
      }),
    );

    await listWorkflows(api, { cursor: "workflow-next" });
    await getWorkflow(api, "openapi-from-workspace", "3");
    await listConfigurations(api, "model-policies", {
      cursor: "policy-next",
    });
    await listCredentials(api, { cursor: "credential-next" });

    expect(new URL(requests[0]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({ limit: "50", cursor: "workflow-next" }),
    );
    expect(requests[1]?.url).toBe(
      "http://127.0.0.1:8080/v1/workflows/openapi-from-workspace/versions/3",
    );
    expect(new URL(requests[2]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({ limit: "50", cursor: "policy-next" }),
    );
    expect(new URL(requests[3]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({ limit: "50", cursor: "credential-next" }),
    );
  });

  it("creates one Run with generated JSON, CSRF, and exact idempotency headers", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response(
          {
            runId: "run_example",
            state: "running",
            runtimeLabels: [],
            labels: {
              "eval.id": "eval-01",
              "eval.leg": "a",
              purpose: "eval",
            },
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
          },
          202,
        );
      }),
    );
    api.csrf.replace("a".repeat(43));
    const request: CreateRunRequest = {
      workflow: "openapi-from-workspace@4",
      labels: { "eval.id": "eval-01", "eval.leg": "a", purpose: "eval" },
      parameters: { objective: "Describe the service" },
      artifacts: {
        source: {
          namespace: "projects",
          name: "source",
          revision: "revision-7",
        },
      },
    };
    await expect(createRun(api, request, "draft-exact-1")).resolves.toEqual({
      runId: "run_example",
      state: "running",
      runtimeLabels: [],
      labels: { "eval.id": "eval-01", "eval.leg": "a", purpose: "eval" },
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
    });
    expect(captured?.method).toBe("POST");
    expect(captured?.headers.get("Idempotency-Key")).toBe("draft-exact-1");
    expect(captured?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(captured?.clone().json()).resolves.toEqual(request);
  });

  it("does not retry a Run mutation after response loss", async () => {
    const fetchImplementation = vi.fn(async () => {
      throw new TypeError("connection reset");
    });
    const api = new PublicAPI(runtimeConfig, fetchImplementation);
    api.csrf.replace("a".repeat(43));
    await expect(
      createRun(api, { workflow: "artifact-copy@1" }, "draft-loss-1"),
    ).rejects.toMatchObject({ code: "network_error", status: 0 });
    expect(fetchImplementation).toHaveBeenCalledTimes(1);
  });

  it("rejects a well-shaped Run response unless the Server returned 202", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({ runId: "run_unexpected", state: "running" }, 200),
      ),
    );
    api.csrf.replace("a".repeat(43));
    await expect(
      createRun(api, { workflow: "artifact-copy@1" }, "draft-status-1"),
    ).rejects.toMatchObject({ code: "invalid_api_response", status: 200 });
  });

  it("fails closed when a creation response contains hostile metadata labels", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response(
          {
            runId: "run_hostile",
            state: "running",
            runtimeLabels: [],
            labels: { "contractor.secret": "must-not-render" },
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
          },
          202,
        ),
      ),
    );
    api.csrf.replace("a".repeat(43));
    await expect(
      createRun(api, { workflow: "artifact-copy@1" }, "draft-hostile-1"),
    ).rejects.toThrow("contractor. prefix is reserved");
  });
});
