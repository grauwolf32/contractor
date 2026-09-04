import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import type { RuntimeConfig } from "../../config/runtime-config";
import { Application } from "../../app/application";
import { createApplicationQueryClient } from "../../app/query-client";
import { applicationRoutes } from "../../app/router";
import type { WorkflowResource } from "../../api/workflows";
import { WorkflowRunForm } from "./run-form";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const digest = `sha256:${"1".repeat(64)}`;
const secondDigest = `sha256:${"2".repeat(64)}`;
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

const workerConfig = {
  modelPolicy: { policyId: "worker", version: "1", digest },
  llmGateway: { gatewayId: "local", version: "1", digest },
};

const workflow: WorkflowResource = {
  ref: { name: "openapi-from-workspace", version: "4" },
  entryStage: "build",
  parameters: {
    objective: { required: true },
    audience: { required: false },
  },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  outputs: {
    openapi: { required: true, mediaTypes: ["application/yaml"] },
  },
  stages: {
    build: {
      objective: "Build an OpenAPI contract from the supplied source tree.",
      instructions: { ref: "instructions/openapi-planner.md", digest },
      planner: { plannerId: "passthrough", version: "1" },
      agents: {
        builder: {
          template: {
            templateId: "openapi-builder",
            version: "1",
            digest,
          },
          namespace: "builder",
        },
      },
      executionConfig: { agents: { builder: workerConfig } },
      contextArtifacts: {
        source: { namespace: "inputs", name: "source", required: true },
      },
      resultArtifacts: {
        openapi: { required: true, mediaTypes: ["application/yaml"] },
      },
      workflowOutputs: { openapi: "openapi" },
      on: {
        succeeded: { kind: "succeed" },
        failed: {
          kind: "escalate",
          maxAttempts: 1,
          executionConfig: {
            ref: {
              configId: "strong-escalation",
              version: "2",
              digest,
            },
            effective: { agents: { builder: workerConfig } },
          },
          then: { kind: "fail" },
        },
        interrupted: { kind: "fail" },
      },
    },
  },
};

const sourceArtifact = {
  artifact: {
    namespace: "projects",
    name: "source",
    revision: "revision-7",
  },
  mediaType: "application/zip",
  size: 1024,
  current: true,
  frozen: false,
  createdAt: "2026-08-31T12:00:00Z",
};

const pinnedRuntimeConfiguration = {
  default: {
    label: "default",
    bindingRevision: "1",
    config: { name: "contractor-empty", version: "1", digest },
  },
  labels: [],
};

const labeledRuntimeConfiguration = {
  default: pinnedRuntimeConfiguration.default,
  labels: [
    {
      label: "caido",
      bindingRevision: "4",
      config: { name: "caido", version: "1", digest: secondDigest },
    },
    {
      label: "debug",
      bindingRevision: "7",
      config: { name: "debug", version: "2", digest },
    },
  ],
};

function apiResponse(value: unknown, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

function inventoryResponse(path: string): Response | undefined {
  if (path === "/v1/artifacts") {
    return apiResponse({
      items: [sourceArtifact],
      page: { hasMore: false },
    });
  }
  if (path === "/v1/configurations/model-policies") {
    return apiResponse({
      items: [
        {
          ref: {
            kind: "model-policies",
            name: "worker-strong",
            version: "2",
            digest,
          },
          body: { model: "strong-model", maxTotalTokens: 100_000 },
          source: "operator",
        },
      ],
      page: { hasMore: false },
    });
  }
  if (path === "/v1/configurations/llm-gateways") {
    return apiResponse({
      items: [
        {
          ref: {
            kind: "llm-gateways",
            name: "local",
            version: "1",
            digest,
          },
          body: {
            protocol: "openai-compatible@1",
            url: "http://192.0.2.1:4000",
          },
          source: "operator",
        },
      ],
      page: { hasMore: false },
    });
  }
  if (path === "/v1/operations/credentials") {
    return apiResponse({
      items: [
        {
          credentialId: "worker-budget",
          llmGateway: { gatewayId: "local", version: "1", digest },
          label: "Worker budget",
          createdAt: "2026-08-31T12:00:00Z",
          effectivePolicy: {
            modelPolicies: [
              { policyId: "worker-strong", version: "2", digest },
            ],
            models: ["strong-model"],
          },
        },
      ],
      page: { hasMore: false },
    });
  }
  if (path === "/v1/operations/runtime-labels") {
    return apiResponse({
      items: [
        {
          label: "default",
          config: pinnedRuntimeConfiguration.default.config,
          revision: "1",
          createdBy: "system",
          createdAt: "2026-08-31T12:00:00Z",
          updatedBy: "system",
          updatedAt: "2026-08-31T12:00:00Z",
        },
        {
          label: "caido",
          config: labeledRuntimeConfiguration.labels[0]!.config,
          revision: "4",
          createdBy: "user_local",
          createdAt: "2026-08-31T12:00:00Z",
          updatedBy: "user_local",
          updatedAt: "2026-08-31T12:00:00Z",
        },
        {
          label: "debug",
          config: labeledRuntimeConfiguration.labels[1]!.config,
          revision: "7",
          createdBy: "user_local",
          createdAt: "2026-08-31T12:00:00Z",
          updatedBy: "user_local",
          updatedAt: "2026-08-31T12:00:00Z",
        },
      ],
      page: { hasMore: false },
    });
  }
  return undefined;
}

function renderWorkflowApplication(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Workflow routes", () => {
  it("isolates infinite Run inventories from finite Operations picker cache", async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const request = input instanceof Request ? input : new Request(input);
      const inventory = inventoryResponse(new URL(request.url).pathname);
      if (inventory !== undefined) {
        return inventory;
      }
      throw new Error(`unexpected ${request.method} ${request.url}`);
    });
    const api = new PublicAPI(runtimeConfig, fetcher);
    const queryClient = createApplicationQueryClient();
    queryClient.setQueryData(queryKeys.configurations.picker("llm-gateways"), {
      items: [],
      page: { hasMore: false },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <PublicAPIProvider api={api}>
          <MemoryRouter>
            <WorkflowRunForm workflow={workflow} />
          </MemoryRouter>
        </PublicAPIProvider>
      </QueryClientProvider>,
    );

    expect(
      await screen.findByRole("heading", { name: "Start Workflow Run" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("0/2")).toBeInTheDocument();
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    expect(
      fetcher.mock.calls.map(
        ([input]) =>
          new URL(input instanceof Request ? input.url : input).pathname,
      ),
    ).toEqual(["/v1/artifacts"]);

    const user = userEvent.setup();
    await user.click(screen.getByText("Execution overrides", { exact: true }));
    expect(
      await screen.findAllByRole("option", { name: /worker-strong@2/ }),
    ).toHaveLength(2);
    expect(
      fetcher.mock.calls.map(
        ([input]) =>
          new URL(input instanceof Request ? input.url : input).pathname,
      ),
    ).not.toContain("/v1/operations/runtime-labels");
  });

  it("paginates exact published Workflow versions", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/workflows") {
          if (url.searchParams.get("cursor") === "next-workflow") {
            return apiResponse({
              items: [
                {
                  ...workflow,
                  ref: { name: "likec4-from-workspace", version: "4" },
                },
              ],
              page: { hasMore: false },
            });
          }
          return apiResponse({
            items: [workflow],
            page: { hasMore: true, nextCursor: "next-workflow" },
          });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderWorkflowApplication(api, "/workflows");
    expect(
      await screen.findByRole("link", { name: "openapi-from-workspace@4" }),
    ).toHaveAttribute("href", "/workflows/openapi-from-workspace/4");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "likec4-from-workspace@4" }),
    ).toBeInTheDocument();
  });

  it("submits declared strings, an exact Artifact, and published overrides", async () => {
    const posts: Array<{ request: Request; body: unknown }> = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (
          url.pathname === "/v1/workflows/openapi-from-workspace/versions/4"
        ) {
          return apiResponse(workflow);
        }
        const inventory = inventoryResponse(url.pathname);
        if (inventory !== undefined) {
          return inventory;
        }
        if (url.pathname === "/v1/runs" && request.method === "POST") {
          posts.push({ request, body: await request.clone().json() });
          return apiResponse(
            {
              runId: "run_openapi",
              state: "running",
              runtimeLabels: ["caido", "debug"],
              labels: {
                "eval.id": "eval-ui-01",
                "eval.leg": "a",
                "eval.name": "openapi-browser",
                purpose: "eval",
              },
              runtimeConfiguration: labeledRuntimeConfiguration,
            },
            { status: 202 },
          );
        }
        if (url.pathname === "/v1/runs/run_openapi") {
          return apiResponse({
            runId: "run_openapi",
            workflow: "openapi-from-workspace@4",
            state: "running",
            runtimeLabels: ["caido", "debug"],
            labels: {
              "eval.id": "eval-ui-01",
              "eval.leg": "a",
              "eval.name": "openapi-browser",
              purpose: "eval",
            },
            runtimeConfiguration: labeledRuntimeConfiguration,
            attempts: [],
            transitions: [],
            outputs: {},
          });
        }
        if (url.pathname === "/v1/runs/run_openapi/artifacts") {
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    const { router } = renderWorkflowApplication(
      api,
      "/workflows/openapi-from-workspace/4",
    );
    expect(
      await screen.findByRole("heading", { name: "openapi-from-workspace@4" }),
    ).toBeInTheDocument();
    await screen.findByText("Scheduler transitions");
    expect(
      screen.getByText(
        (_, element) =>
          element?.tagName === "SPAN" &&
          element.textContent?.includes("escalate up to 1 attempt using") ===
            true,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("strong-escalation@2", { selector: "code" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /escalate/i }),
    ).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByText("Runtime placement", { exact: true }));
    expect(
      await screen.findByText("Default · always applied"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: /debug/ }));
    await user.click(screen.getByRole("checkbox", { name: /caido/ }));
    await user.click(screen.getByText("Run metadata", { exact: true }));
    await user.click(
      screen.getByRole("button", { name: "Add eval metadata preset" }),
    );
    expect(screen.getByLabelText("Run metadata label key 1")).toHaveValue(
      "purpose",
    );
    await user.type(
      screen.getByLabelText("Run metadata label value 2"),
      "openapi-browser",
    );
    await user.type(
      screen.getByLabelText("Run metadata label value 3"),
      "eval-ui-01",
    );
    await user.type(screen.getByLabelText("Run metadata label value 4"), "a");
    await user.type(screen.getByLabelText(/^objective/i), "Build public API");
    await screen.findByRole("option", {
      name: /projects\/source@revision-7/,
    });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(screen.getByText("Execution overrides", { exact: true }));
    const workerOverrides = screen.getByRole("group", { name: "Workers" });
    await user.selectOptions(
      within(workerOverrides).getByLabelText("Model policy"),
      "worker-strong@2",
    );
    await user.selectOptions(
      within(workerOverrides).getByLabelText("LLM Gateway"),
      "local@1",
    );
    await user.selectOptions(
      within(workerOverrides).getByLabelText("Credential"),
      "worker-budget",
    );
    expect(screen.queryByLabelText(/Gateway URL/i)).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );

    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/runs/run_openapi"),
    );
    expect(posts).toHaveLength(1);
    expect(posts[0]?.request.headers.get("X-CSRF-Token")).toBe(
      session.csrfToken,
    );
    expect(posts[0]?.request.headers.get("Idempotency-Key")).toMatch(
      /^run-ui-[0-9a-f]{32}$/,
    );
    expect(posts[0]?.body).toEqual({
      workflow: "openapi-from-workspace@4",
      runtimeLabels: ["caido", "debug"],
      labels: {
        "eval.id": "eval-ui-01",
        "eval.leg": "a",
        "eval.name": "openapi-browser",
        purpose: "eval",
      },
      parameters: { objective: "Build public API" },
      artifacts: {
        source: {
          namespace: "projects",
          name: "source",
          revision: "revision-7",
        },
      },
      executionConfig: {
        workers: {
          modelPolicy: "worker-strong@2",
          llmGateway: "local@1",
          credential: "worker-budget",
        },
      },
    });
    expect(
      await screen.findByRole("heading", { name: "run_openapi" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Authoritative aggregate")).toBeInTheDocument();
    expect(await screen.findByText("binding revision 7")).toBeInTheDocument();
    const metadata = screen
      .getByRole("heading", { name: "Run metadata labels" })
      .closest("section");
    expect(metadata).not.toBeNull();
    expect(metadata).toHaveTextContent("eval.id=eval-ui-01");
    expect(metadata).toHaveTextContent("eval.leg=a");
    expect(metadata).toHaveTextContent("Immutable");
  });

  it("blocks a mutation while required declared fields are missing", async () => {
    let postCount = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (
          url.pathname === "/v1/workflows/openapi-from-workspace/versions/4"
        ) {
          return apiResponse(workflow);
        }
        if (
          url.pathname.startsWith("/v1/configurations/") ||
          url.pathname === "/v1/operations/credentials"
        ) {
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        const inventory = inventoryResponse(url.pathname);
        if (inventory !== undefined) {
          return inventory;
        }
        if (url.pathname === "/v1/runs") {
          postCount += 1;
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderWorkflowApplication(api, "/workflows/openapi-from-workspace/4");
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    expect(
      await screen.findByText("Required string parameter is missing."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Required Artifact input is missing."),
    ).toBeInTheDocument();
    expect(postCount).toBe(0);
    await user.click(screen.getByText("Execution overrides", { exact: true }));
    expect(
      await screen.findByText("No ModelPolicy versions published."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No LLMGatewayConfig versions published."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No active credentials available."),
    ).toBeInTheDocument();
  });

  it("reuses one key for exact response-loss retry and rotates it after edits", async () => {
    const keys: string[] = [];
    let postCount = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (
          url.pathname === "/v1/workflows/openapi-from-workspace/versions/4"
        ) {
          return apiResponse(workflow);
        }
        const inventory = inventoryResponse(url.pathname);
        if (inventory !== undefined) {
          return inventory;
        }
        if (url.pathname === "/v1/runs" && request.method === "POST") {
          keys.push(request.headers.get("Idempotency-Key") ?? "missing");
          postCount += 1;
          if (postCount < 3) {
            throw new TypeError("response lost");
          }
          return apiResponse(
            {
              runId: "run_changed",
              state: "running",
              runtimeLabels: [],
              labels: { "eval.leg": "b" },
              runtimeConfiguration: pinnedRuntimeConfiguration,
            },
            { status: 202 },
          );
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    const { router } = renderWorkflowApplication(
      api,
      "/workflows/openapi-from-workspace/4",
    );
    const user = userEvent.setup();
    const objective = await screen.findByLabelText(/^objective/i);
    await user.type(objective, "Initial objective");
    await user.click(screen.getByText("Run metadata", { exact: true }));
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 1"),
      "eval.leg",
    );
    const labelValue = screen.getByLabelText("Run metadata label value 1");
    await user.type(labelValue, "a");
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Retry exact request" }),
    );
    expect(
      await screen.findByRole("button", { name: "Retry exact request" }),
    ).toBeInTheDocument();
    expect(keys[0]).toBe(keys[1]);

    await user.clear(labelValue);
    await user.type(labelValue, "b");
    await user.click(
      screen.getByRole("button", {
        name: "Start changed draft with a new key",
      }),
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/runs/run_changed"),
    );
    expect(keys).toHaveLength(3);
    expect(keys[2]).not.toBe(keys[1]);
  });

  it("blocks duplicate and reserved Run metadata labels before mutation", async () => {
    let postCount = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (
          url.pathname === "/v1/workflows/openapi-from-workspace/versions/4"
        ) {
          return apiResponse(workflow);
        }
        const inventory = inventoryResponse(url.pathname);
        if (inventory !== undefined) {
          return inventory;
        }
        if (url.pathname === "/v1/runs" && request.method === "POST") {
          postCount += 1;
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderWorkflowApplication(api, "/workflows/openapi-from-workspace/4");
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Validate labels",
    );
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(screen.getByText("Run metadata", { exact: true }));
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 1"),
      "contractor.secret",
    );
    await user.type(
      screen.getByLabelText("Run metadata label value 1"),
      "must-not-submit",
    );
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 2"),
      "eval.id",
    );
    await user.type(screen.getByLabelText("Run metadata label value 2"), "one");
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 3"),
      "eval.id",
    );
    await user.type(screen.getByLabelText("Run metadata label value 3"), "two");
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );

    expect(
      await screen.findByText("The contractor. prefix is reserved."),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("Label key eval.id is duplicated."),
    ).toHaveLength(2);
    expect(postCount).toBe(0);
  });
});
