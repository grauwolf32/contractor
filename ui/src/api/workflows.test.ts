import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  createProjectRun,
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
    const abort = new AbortController();
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

    await listWorkflows(api, {
      cursor: "workflow-next",
      q: "API purpose",
      signal: abort.signal,
    });
    await getWorkflow(api, "openapi-from-workspace", "3");
    await listConfigurations(api, "model-policies", {
      cursor: "policy-next",
      q: "strong",
      name: "worker",
      signal: abort.signal,
    });
    await listCredentials(api, { cursor: "credential-next" });

    expect(new URL(requests[0]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({
        limit: "50",
        cursor: "workflow-next",
        q: "API purpose",
      }),
    );
    expect(requests[1]?.url).toBe(
      "http://127.0.0.1:8080/v1/workflows/openapi-from-workspace/versions/3",
    );
    expect(new URL(requests[2]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({
        limit: "50",
        cursor: "policy-next",
        q: "strong",
        name: "worker",
      }),
    );
    expect(new URL(requests[3]?.url ?? "http://invalid").searchParams).toEqual(
      new URLSearchParams({ limit: "50", cursor: "credential-next" }),
    );
    abort.abort();
    expect(requests[0]?.signal.aborted).toBe(true);
    expect(requests[2]?.signal.aborted).toBe(true);
  });

  it("rejects overlong discovery queries and invalid exact names locally", async () => {
    const fetcher = vi.fn(async () =>
      response({ items: [], page: { hasMore: false } }),
    );
    const api = new PublicAPI(runtimeConfig, fetcher);
    await expect(listWorkflows(api, { q: "x".repeat(201) })).rejects.toThrow(
      "exceeds 200 Unicode characters",
    );
    await expect(
      listConfigurations(api, "agent-templates", { name: "not/a/name" }),
    ).rejects.toThrow("exact name is invalid");
    expect(fetcher).not.toHaveBeenCalled();
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
      workflow: "openapi-from-workspace@5",
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

  it("creates a Project Run only through its scope-bound endpoint", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response(
          {
            runId: "run_project_example",
            projectId: "project_example",
            state: "initializing",
            runtimeLabels: [],
            labels: {},
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
      workflow: "openapi-from-workspace@5",
      artifacts: {
        source: {
          namespace: "sources",
          name: "service",
          revision: "revision-1",
        },
      },
    };

    await expect(
      createProjectRun(api, "project_example", request, "project-run-1"),
    ).resolves.toMatchObject({
      runId: "run_project_example",
      projectId: "project_example",
    });
    expect(new URL(captured!.url).pathname).toBe(
      "/v1/projects/project_example/runs",
    );
    expect(captured?.headers.get("Idempotency-Key")).toBe("project-run-1");
    expect(await captured?.clone().json()).toEqual(request);
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
