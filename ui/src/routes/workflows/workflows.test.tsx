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
import { RunDraftProvider } from "../../run-drafts/provider";
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
  ref: { name: "openapi-from-workspace", version: "fixture-1" },
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
      session: "isolated",
      agents: {
        builder: {
          template: {
            templateId: "openapi-builder",
            version: "1",
            digest,
          },
          namespace: "builder",
          skills: [{ namespace: "skills", name: "openapi-analysis" }],
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

const workflowSelector = `${workflow.ref.name}@${workflow.ref.version}`;
const workflowRoute = `/workflows/${workflow.ref.name}/${workflow.ref.version}`;
const workflowEndpoint = `/v1/workflows/${workflow.ref.name}/versions/${workflow.ref.version}`;
const nextWorkflow: WorkflowResource = {
  ...workflow,
  ref: { name: "likec4-from-workspace", version: "fixture-1" },
};
const nextWorkflowSelector = `${nextWorkflow.ref.name}@${nextWorkflow.ref.version}`;

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
  it.each([undefined, "project_example"])(
    "only offers compatible input revisions in scope %s",
    async (projectId) => {
      const textArtifact = {
        ...sourceArtifact,
        artifact: { ...sourceArtifact.artifact, name: "readme" },
        mediaType: "text/plain",
      };
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const path = new URL(request.url).pathname;
          if (path.endsWith("/artifacts")) {
            return apiResponse({
              items: [sourceArtifact, textArtifact],
              page: { hasMore: false },
            });
          }
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      render(
        <QueryClientProvider client={createApplicationQueryClient()}>
          <PublicAPIProvider api={api}>
            <RunDraftProvider ownerId="user_local">
              <MemoryRouter>
                <WorkflowRunForm
                  workflow={{
                    ...workflow,
                    inputs: {
                      ...workflow.inputs,
                      document: {
                        required: false,
                        mediaTypes: ["application/pdf"],
                      },
                      any: { required: false, mediaTypes: ["*/*"] },
                    },
                  }}
                  {...(projectId === undefined ? {} : { projectId })}
                />
              </MemoryRouter>
            </RunDraftProvider>
          </PublicAPIProvider>
        </QueryClientProvider>,
      );
      const source = screen.getByRole("combobox", { name: "source required" });
      await within(source).findByRole("option", { name: /source@revision-7/ });
      expect(within(source).getAllByRole("option")).toHaveLength(2);
      expect(
        within(source).queryByRole("option", { name: /readme/ }),
      ).toBeNull();
      const document = screen.getByRole("combobox", { name: "document" });
      expect(within(document).getAllByRole("option")).toHaveLength(1);
      expect(screen.getByText(/No matching artifacts found/)).toBeVisible();
      const any = screen.getByRole("combobox", { name: "any" });
      expect(within(any).getAllByRole("option")).toHaveLength(3);
      expect(screen.queryByRole("option", { name: /Incompatible/ })).toBeNull();
      const git = screen.getByRole("button", { name: "Import Git for source" });
      expect(git.querySelector("svg")).not.toBeNull();
      expect(
        screen.queryByRole("button", { name: "Import Git for document" }),
      ).toBeNull();
      await userEvent.setup().click(git);
      expect(
        screen.getByRole("dialog", { name: "Import Git repository" }),
      ).toBeVisible();
    },
  );

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
          <RunDraftProvider ownerId="user_local">
            <MemoryRouter>
              <WorkflowRunForm workflow={workflow} />
            </MemoryRouter>
          </RunDraftProvider>
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
              items: [nextWorkflow],
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
      await screen.findByRole("link", { name: workflowSelector }),
    ).toHaveAttribute("href", workflowRoute);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: nextWorkflowSelector }),
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
        if (url.pathname === workflowEndpoint) {
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
            workflow: workflowSelector,
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
    const { router } = renderWorkflowApplication(api, workflowRoute);
    expect(
      await screen.findByRole("heading", { name: workflowSelector }),
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
      workflow: workflowSelector,
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
      await screen.findByRole("button", { name: "Copy Run ID" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Workflow Run", { exact: true }),
    ).toBeInTheDocument();
    expect(await screen.findByText("binding revision 7")).toBeInTheDocument();
    const metadata = screen
      .getByRole("heading", { name: "Run metadata labels" })
      .closest("section");
    expect(metadata).not.toBeNull();
    expect(metadata).toHaveTextContent("eval.id:eval-ui-01");
    expect(metadata).toHaveTextContent("eval.leg:a");
    expect(metadata).toHaveTextContent("Labels are fixed at creation");
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
        if (url.pathname === workflowEndpoint) {
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
    renderWorkflowApplication(api, workflowRoute);
    expect(
      await screen.findByRole("link", { name: "skills/openapi-analysis" }),
    ).toHaveAttribute("href", "/artifacts/skills/openapi-analysis");
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
        if (url.pathname === workflowEndpoint) {
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
    const { router } = renderWorkflowApplication(api, workflowRoute);
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
    await screen.findByRole("button", { name: "Retry exact request" });
    await user.type(screen.getByLabelText("Add a label"), "pending:restored");
    await router.navigate("/artifacts");
    await screen.findByRole("heading", { name: "Artifacts" });
    await router.navigate(workflowRoute);
    expect(await screen.findByLabelText(/^objective/i)).toHaveValue(
      "Initial objective",
    );
    expect(screen.getByLabelText("Run metadata label value 1")).toHaveValue(
      "a",
    );
    expect(screen.getByLabelText("Add a label")).toHaveValue(
      "pending:restored",
    );
    expect(screen.getByLabelText(/^source/i)).toHaveValue(
      "projects/source@revision-7",
    );
    await user.click(
      await screen.findByRole("button", { name: "Retry exact request" }),
    );
    expect(
      await screen.findByRole("button", { name: "Retry exact request" }),
    ).toBeInTheDocument();
    expect(keys[0]).toBe(keys[1]);

    const restoredLabelValue = screen.getByLabelText(
      "Run metadata label value 1",
    );
    await user.clear(restoredLabelValue);
    await user.type(restoredLabelValue, "b");
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

  it("authors editable colon badges and key-only labels without losing pending input on submit", async () => {
    const requests: unknown[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return apiResponse(session);
        if (url.pathname === workflowEndpoint) return apiResponse(workflow);
        const inventory = inventoryResponse(url.pathname);
        if (inventory !== undefined) return inventory;
        if (url.pathname === "/v1/runs" && request.method === "POST") {
          requests.push(await request.json());
          throw new TypeError("response lost");
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderWorkflowApplication(api, workflowRoute);
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Badge labels",
    );
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(screen.getByText("Run metadata", { exact: true }));
    const composer = screen.getByLabelText("Add a label");
    await user.type(
      composer,
      "team:platform{Enter}debug{Enter}endpoint:https://host:8443/a=b{Enter}",
    );
    expect(requests).toHaveLength(0);
    expect(screen.getByLabelText("Run metadata label key 1")).toHaveValue(
      "team",
    );
    expect(screen.getByLabelText("Run metadata label value 2")).toHaveValue("");
    expect(screen.getByLabelText("Run metadata label value 3")).toHaveValue(
      "https://host:8443/a=b",
    );
    await user.clear(screen.getByLabelText("Run metadata label value 1"));
    await user.type(
      screen.getByLabelText("Run metadata label value 1"),
      "infra{Enter}",
    );
    expect(requests).toHaveLength(0);
    await user.click(
      screen.getByRole("button", { name: "Remove Run metadata label 3" }),
    );
    await user.type(composer, "temporary:");
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    expect(screen.getAllByLabelText(/^Run metadata label key /)).toHaveLength(
      3,
    );
    expect(screen.getByLabelText("Run metadata label value 3")).toHaveValue("");
    await user.click(
      screen.getByRole("button", { name: "Remove Run metadata label 3" }),
    );
    await user.type(composer, "release:next");
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    await screen.findByRole("button", { name: "Retry exact request" });
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      labels: { team: "infra", debug: "", release: "next" },
    });
  });

  it("uploads a local UserScope file into its originating exact input slot", async () => {
    const puts: Request[] = [];
    let stored = false;
    const uploadedArtifact = {
      artifact: {
        namespace: "inputs",
        name: "service-source",
        revision: "revision-upload-1",
      },
      mediaType: "application/zip",
      size: 3,
      current: true,
      frozen: false,
      createdAt: "2026-09-06T20:00:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return apiResponse(session);
        if (url.pathname === workflowEndpoint) return apiResponse(workflow);
        if (url.pathname === "/v1/artifacts" && request.method === "GET") {
          return apiResponse({
            items: stored ? [uploadedArtifact] : [],
            page: { hasMore: false },
          });
        }
        if (
          url.pathname === "/v1/artifacts/inputs/service-source" &&
          request.method === "PUT"
        ) {
          puts.push(request.clone());
          expect(Buffer.from(await request.arrayBuffer()).toString()).toBe(
            "zip",
          );
          expect(request.headers.get("If-None-Match")).toBe("*");
          expect(request.headers.get("Content-Type")).toBe("application/zip");
          stored = true;
          return apiResponse(
            {
              artifact: uploadedArtifact.artifact,
              mediaType: uploadedArtifact.mediaType,
              size: uploadedArtifact.size,
            },
            { status: 201, headers: { ETag: '"revision-upload-1"' } },
          );
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderWorkflowApplication(api, workflowRoute);
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Keep this field while uploading",
    );
    await user.click(
      screen.getByRole("button", { name: "Upload local file for source" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Upload local file for source",
    });
    await user.upload(
      within(dialog).getByLabelText("Drop a file here"),
      new File(["zip"], "service-source.zip", {
        type: "application/zip",
      }),
    );
    expect(within(dialog).getByLabelText("Namespace")).toHaveValue("inputs");
    expect(within(dialog).getByLabelText("Namespace")).toBeDisabled();
    expect(within(dialog).getByLabelText("Name")).toHaveValue("service-source");
    await user.click(
      within(dialog).getByRole("button", {
        name: "Upload and select exact revision",
      }),
    );

    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    expect(puts).toHaveLength(1);
    expect(screen.getByLabelText(/^objective/i)).toHaveValue(
      "Keep this field while uploading",
    );
    expect(screen.getByLabelText(/^source/i)).toHaveValue(
      "inputs/service-source@revision-upload-1",
    );

    await user.click(
      screen.getByRole("button", { name: "Discard saved draft" }),
    );
    const confirmation = screen.getByRole("alertdialog", {
      name: "Discard this Run draft?",
    });
    expect(
      within(confirmation).getByRole("button", { name: "Keep editing" }),
    ).toHaveFocus();
    await user.click(
      within(confirmation).getByRole("button", {
        name: "Discard Run draft",
      }),
    );
    expect(await screen.findByLabelText(/^objective/i)).toHaveValue("");
    expect(screen.getByLabelText(/^source/i)).toHaveValue("");
  });

  it("aborts a pending local upload without binding it into the retained draft", async () => {
    let uploadStarted: (() => void) | undefined;
    const started = new Promise<void>((resolve) => {
      uploadStarted = resolve;
    });
    let uploadAborted = false;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return apiResponse(session);
        if (url.pathname === workflowEndpoint) return apiResponse(workflow);
        if (url.pathname === "/v1/artifacts" && request.method === "GET") {
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        if (
          url.pathname === "/v1/artifacts/inputs/cancel-source" &&
          request.method === "PUT"
        ) {
          uploadStarted?.();
          return await new Promise<Response>((_resolve, reject) => {
            const abort = () => {
              uploadAborted = true;
              reject(new DOMException("Aborted", "AbortError"));
            };
            if (request.signal.aborted) {
              abort();
              return;
            }
            request.signal.addEventListener("abort", abort, { once: true });
          });
        }
        throw new Error(`unexpected ${request.method} ${url}`);
      }),
    );
    renderWorkflowApplication(api, workflowRoute);
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Retain this draft after cancellation",
    );
    await user.click(
      screen.getByRole("button", { name: "Upload local file for source" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Upload local file for source",
    });
    await user.upload(
      within(dialog).getByLabelText("Drop a file here"),
      new File(["zip"], "cancel-source.zip", { type: "application/zip" }),
    );
    await user.click(
      within(dialog).getByRole("button", {
        name: "Upload and select exact revision",
      }),
    );
    await started;
    await user.click(
      within(dialog).getByRole("button", {
        name: "Close local file upload",
      }),
    );

    await waitFor(() => expect(uploadAborted).toBe(true));
    expect(dialog).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^objective/i)).toHaveValue(
      "Retain this draft after cancellation",
    );
    expect(screen.getByLabelText(/^source/i)).toHaveValue("");
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
        if (url.pathname === workflowEndpoint) {
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
    renderWorkflowApplication(api, workflowRoute);
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
