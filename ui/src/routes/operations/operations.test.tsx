import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import type { RuntimeConfig } from "../../config/runtime-config";
import { RunEventsManager } from "../../events/run-events";

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

function apiResponse(
  value: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {},
): Response {
  return new Response(value === undefined ? undefined : JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...extraHeaders,
    },
  });
}

function snapshot(revision = "7") {
  return {
    cursor: { generation: "operations-generation-1", revision },
    runtimeAgents: [],
    allocations: [],
  };
}

class OperationsWebSocket {
  static instances: OperationsWebSocket[] = [];
  protocol = "contractor.events.v1";
  readyState = 0;
  sent: string[] = [];
  onopen: ((event: Event) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  onclose: ((event: CloseEvent) => unknown) | null = null;

  constructor(
    readonly url: string | URL,
    readonly protocols?: string | string[],
  ) {
    OperationsWebSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(value);
  }

  close(code?: number): void {
    this.readyState = 3;
    this.onclose?.({ code: code ?? 1000 } as CloseEvent);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  message(value: unknown): void {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

function renderOperations(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const events = new RunEventsManager(runtimeConfig.apiBaseUrl, {
    WebSocketImplementation: OperationsWebSocket as unknown as typeof WebSocket,
  });
  return render(
    <Application
      api={api}
      publicAPI={api}
      runEvents={events}
      router={router}
    />,
  );
}

function sessionResponse(request: Request): Response | undefined {
  return new URL(request.url).pathname === "/v1/auth/session"
    ? apiResponse(session)
    : undefined;
}

beforeEach(() => {
  OperationsWebSocket.instances = [];
});

describe("Operations routes", () => {
  it("keeps fenced Runtime observation separate from the authoritative lease", async () => {
    let snapshotReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) {
          return authenticated;
        }
        if (new URL(request.url).pathname === "/v1/operations/snapshot") {
          snapshotReads += 1;
          return apiResponse({
            ...snapshot(snapshotReads === 1 ? "7" : "8"),
            runtimeAgents: [
              {
                instanceId: "runtime-vm-1",
                softwareVersion: "0.1.0",
                supportedRuntimes: ["python@1", "adk@1"],
                supportedToolsets: [
                  {
                    ref: "likec4@1",
                    tools: ["validate_likec4", "read_likec4"],
                  },
                ],
                supportedSandboxProfiles: ["local-workdir@1"],
                supportedRuntimeAdapters: [],
                observedState: "fenced",
                slotState: "fenced",
                lastAcceptedHeartbeat: "2026-08-31T12:00:00Z",
                confirmedLeaseUntil: "2026-08-31T12:01:00Z",
                currentAllocationId: "allocation-observed",
                authoritativeAllocationId: "allocation-authoritative",
                reconciliationReason: {
                  code: "lease_confirmation_lost",
                  retryable: true,
                },
              },
              {
                instanceId: "runtime-vm-minimal",
                softwareVersion: "0.1.0",
                supportedRuntimes: ["adk@1"],
                supportedToolsets: [],
                supportedSandboxProfiles: ["local-workdir@1"],
                supportedRuntimeAdapters: [],
                observedState: "idle",
                slotState: "idle",
              },
            ],
          });
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    renderOperations(api, "/operations/runtime-agents");
    expect(
      await screen.findByRole("heading", { name: "Runtime Agents" }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("fenced")).toHaveLength(2);
    expect(screen.getByText("reconciliation pending")).toBeInTheDocument();
    expect(screen.getByText("validate_likec4")).toBeInTheDocument();
    expect(screen.getByText("No usable Toolsets reported")).toBeInTheDocument();
    expect(
      screen.queryByText(/LOCAL_PROBE_SECRET_CANARY|private\/path/),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(/Self-termination remains the recovery boundary/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /force|idle|reassign/i }),
    ).not.toBeInTheDocument();

    await waitFor(() => expect(OperationsWebSocket.instances).toHaveLength(1));
    const socket = OperationsWebSocket.instances[0];
    act(() => socket?.open());
    const subscription = JSON.parse(socket?.sent[0] ?? "{}");
    expect(subscription).toMatchObject({
      stream: { kind: "operations" },
      after: { generation: "operations-generation-1", sequence: "7" },
    });
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "subscribed",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "operations" },
        cursor: subscription.after,
      }),
    );
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "operations" },
        cursor: {
          generation: "operations-generation-1",
          sequence: "8",
        },
        kind: "operations.changed",
        occurredAt: "2026-08-31T12:00:10Z",
        data: {
          resource: "runtimeAgent",
          resourceId: "runtime-vm-1",
          revision: "8",
        },
      }),
    );
    await waitFor(() => expect(snapshotReads).toBe(2));
  });

  it("clones and publishes a new exact ModelPolicy without rendering unknown data", async () => {
    const publishedBodies: unknown[] = [];
    const original = {
      ref: {
        kind: "model-policies" as const,
        name: "worker",
        version: "1",
        digest,
      },
      body: {
        model: "qwen-worker",
        maxOutputTokens: 4096,
        maxModelCalls: 8,
        maxToolCalls: 16,
        maxTotalTokens: 32768,
        providerApiKey: "CONFIG_SECRET_CANARY",
      },
      source: "operator" as const,
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const path = new URL(request.url).pathname;
        if (path === "/v1/operations/snapshot") return apiResponse(snapshot());
        if (path === "/v1/configurations/model-policies/worker/versions/1") {
          return apiResponse(original);
        }
        if (
          path === "/v1/configurations/model-policies" &&
          request.method === "POST"
        ) {
          publishedBodies.push(await request.clone().json());
          return apiResponse(
            {
              ref: {
                kind: "model-policies",
                name: "worker",
                version: "2",
                digest: secondDigest,
              },
              body: {
                model: "qwen-worker",
                maxOutputTokens: 4096,
                maxModelCalls: 8,
                maxToolCalls: 16,
                maxTotalTokens: 32768,
              },
              source: "managed",
            },
            201,
          );
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    renderOperations(api, "/operations/configurations/model-policies/worker/1");
    expect(
      await screen.findByRole("heading", { name: "Published values" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("CONFIG_SECRET_CANARY")).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("New immutable version"), "2");
    await user.click(
      screen.getByRole("button", { name: "Publish immutable version" }),
    );
    expect(await screen.findByText("Published worker@2")).toBeInTheDocument();
    expect(screen.getByText("worker@1")).toBeInTheDocument();
    expect(publishedBodies).toEqual([
      {
        name: "worker",
        version: "2",
        modelPolicy: {
          model: "qwen-worker",
          maxOutputTokens: 4096,
          maxModelCalls: 8,
          maxToolCalls: 16,
          maxTotalTokens: 32768,
        },
      },
    ]);
  });

  it("creates a typed LiteLLM credential without accepting or rendering a secret", async () => {
    const createdBodies: unknown[] = [];
    const gateway = {
      ref: {
        kind: "llm-gateways" as const,
        name: "local-litellm",
        version: "1",
        digest,
      },
      body: {
        protocol: "openai-compatible@1" as const,
        url: "http://127.0.0.1:4000/v1",
        credentialManager: {
          implementation: "litellm-virtual-keys@1" as const,
          managementUrl: "http://127.0.0.1:4000",
        },
      },
      source: "operator" as const,
    };
    const policy = {
      ref: {
        kind: "model-policies" as const,
        name: "worker",
        version: "1",
        digest,
      },
      body: { model: "qwen-worker" },
      source: "operator" as const,
    };
    const credential = {
      credentialId: "worker-budget",
      llmGateway: { gatewayId: "local-litellm", version: "1", digest },
      label: "Worker budget",
      createdAt: "2026-08-31T12:00:00Z",
      effectivePolicy: {
        modelPolicies: [{ policyId: "worker", version: "1", digest }],
        models: ["qwen-worker"],
        maxBudget: 10,
        budgetDuration: "1d",
      },
      generatedToken: "CREDENTIAL_SECRET_CANARY",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const path = new URL(request.url).pathname;
        if (path === "/v1/operations/snapshot") return apiResponse(snapshot());
        if (path === "/v1/configurations/llm-gateways") {
          return apiResponse({ items: [gateway], page: { hasMore: false } });
        }
        if (path === "/v1/configurations/model-policies") {
          return apiResponse({ items: [policy], page: { hasMore: false } });
        }
        if (path === "/v1/operations/credentials") {
          if (request.method === "POST") {
            createdBodies.push(await request.clone().json());
            return apiResponse(credential, 201);
          }
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/operations/credentials/worker-budget") {
          return apiResponse(credential);
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    renderOperations(api, "/operations/credentials");
    expect(
      await screen.findByRole("heading", { name: "Create credential" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText(/token|secret|key value/i),
    ).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Credential ID"), "worker-budget");
    await user.type(
      screen.getByLabelText("Safe label (optional)"),
      "Worker budget",
    );
    await user.selectOptions(
      screen.getByLabelText("Exact managed LLM Gateway"),
      screen.getByRole("option", { name: /local-litellm@1/ }),
    );
    await user.click(screen.getByLabelText(/worker@1/));
    await user.type(
      screen.getByLabelText("Maximum spend · Gateway-enforced (LiteLLM)"),
      "10",
    );
    await user.type(
      screen.getByLabelText("Budget reset · Gateway-enforced (LiteLLM)"),
      "1d",
    );
    await user.click(
      screen.getByRole("button", { name: "Create active credential" }),
    );
    expect(
      await screen.findByRole("heading", { name: "LiteLLM-enforced limits" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "worker-budget" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("CREDENTIAL_SECRET_CANARY"),
    ).not.toBeInTheDocument();
    expect(JSON.stringify(createdBodies)).not.toMatch(/token|secret/i);
    expect(createdBodies).toEqual([
      {
        credentialId: "worker-budget",
        llmGateway: {
          gatewayId: "local-litellm",
          version: "1",
          digest,
        },
        label: "Worker budget",
        gatewayPolicy: {
          modelPolicies: [{ policyId: "worker", version: "1", digest }],
          maxBudget: 10,
          budgetDuration: "1d",
        },
      },
    ]);
  });

  it("keeps an in-use credential active and links only safe Run IDs", async () => {
    const credential = {
      credentialId: "worker-budget",
      llmGateway: { gatewayId: "local-litellm", version: "1", digest },
      createdAt: "2026-08-31T12:00:00Z",
      effectivePolicy: {
        modelPolicies: [{ policyId: "worker", version: "1", digest }],
        models: ["qwen-worker"],
      },
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const path = new URL(request.url).pathname;
        if (path === "/v1/operations/snapshot") return apiResponse(snapshot());
        if (path === "/v1/operations/credentials/worker-budget") {
          if (request.method === "DELETE") {
            return apiResponse(
              {
                code: "credential_in_use",
                message: "Credential is pinned by non-terminal Runs",
                retryable: false,
                requestId: "request-delete-1",
                details: {
                  kind: "credential_in_use",
                  runIds: ["run-openapi", "run-likec4"],
                },
              },
              409,
            );
          }
          return apiResponse(credential);
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    renderOperations(api, "/operations/credentials/worker-budget");
    expect(
      await screen.findByText("active", { selector: ".state-badge" }),
    ).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(
      screen.getByLabelText(/I understand that the LiteLLM key/),
    );
    await user.click(
      screen.getByRole("button", {
        name: "Delete from LiteLLM and Contractor",
      }),
    );
    expect(
      await screen.findByRole("link", { name: "run-openapi" }),
    ).toHaveAttribute("href", "/runs/run-openapi");
    expect(
      screen.getByRole("link", { name: "run-likec4" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("active", { selector: ".state-badge" }),
    ).toBeInTheDocument();
  });

  it("publishes typed Runtime configuration, erases secrets, and exposes stale binding CAS", async () => {
    const secret = "Bearer runtime-secret-canary";
    const baseResource = {
      ref: { name: "contractor-empty", version: "1", digest },
      document: {
        apiVersion: "contractor/v1alpha1",
        kind: "RuntimeConfig",
        metadata: { name: "contractor-empty", version: "1" },
        spec: {},
      },
      builtIn: true,
      createdBy: "system",
      createdAt: "2026-08-31T12:00:00Z",
    };
    const defaultBinding = {
      label: "default",
      config: baseResource.ref,
      revision: "1",
      createdBy: "system",
      createdAt: "2026-08-31T12:00:00Z",
      updatedBy: "system",
      updatedAt: "2026-08-31T12:00:00Z",
    };
    let publishedResource: typeof baseResource | undefined;
    let debugBinding: typeof defaultBinding | undefined;
    let runtimeCredentialBody: unknown;
    let bindingWrites = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const path = new URL(request.url).pathname;
        if (path === "/v1/operations/snapshot") return apiResponse(snapshot());
        if (path === "/v1/configurations/llm-gateways") {
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/operations/runtime-configs") {
          if (request.method === "POST") {
            const document = (await request
              .clone()
              .json()) as typeof baseResource.document;
            publishedResource = {
              ref: {
                name: document.metadata.name,
                version: document.metadata.version,
                digest: secondDigest,
              },
              document,
              builtIn: false,
              createdBy: "user_local",
              createdAt: "2026-08-31T12:01:00Z",
            };
            return apiResponse(publishedResource, 201);
          }
          return apiResponse({
            items: [
              baseResource,
              ...(publishedResource === undefined ? [] : [publishedResource]),
            ],
            page: { hasMore: false },
          });
        }
        if (path === "/v1/operations/runtime-labels") {
          return apiResponse({
            items: [
              defaultBinding,
              ...(debugBinding === undefined ? [] : [debugBinding]),
            ],
            page: { hasMore: false },
          });
        }
        if (path === "/v1/operations/runtime-labels/debug") {
          bindingWrites += 1;
          if (bindingWrites > 1) {
            return apiResponse(
              {
                code: "precondition_failed",
                message: "Runtime label revision changed",
                retryable: false,
              },
              412,
            );
          }
          debugBinding = {
            label: "debug",
            config: publishedResource!.ref,
            revision: "1",
            createdBy: "user_local",
            createdAt: "2026-08-31T12:02:00Z",
            updatedBy: "user_local",
            updatedAt: "2026-08-31T12:02:00Z",
          };
          expect(request.headers.get("If-None-Match")).toBe("*");
          return apiResponse(debugBinding, 201, { ETag: '"1"' });
        }
        if (path === "/v1/operations/runtime-credentials") {
          if (request.method === "POST") {
            runtimeCredentialBody = await request.clone().json();
            return apiResponse(
              {
                credentialId: "otel-debug",
                kind: "otlp-headers@1",
                createdBy: "user_local",
                createdAt: "2026-08-31T12:00:30Z",
              },
              201,
            );
          }
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    const view = renderOperations(api, "/operations/runtime-configs");
    expect(
      await screen.findByRole("heading", { name: "RuntimeConfig versions" }),
    ).toBeInTheDocument();
    const user = userEvent.setup();

    const credentialForm = view.container.querySelector(
      "form.runtime-credential-form",
    ) as HTMLFormElement;
    await user.type(
      within(credentialForm).getByLabelText("Runtime credential ID"),
      "otel-debug",
    );
    await user.type(
      within(credentialForm).getByLabelText(/Header value/),
      secret,
    );
    await user.click(
      within(credentialForm).getByRole("button", {
        name: "Create active Runtime credential",
      }),
    );
    expect(
      await within(credentialForm).findByText(/Created safe metadata/),
    ).toBeInTheDocument();
    expect(within(credentialForm).getByLabelText(/Header value/)).toHaveValue(
      "",
    );
    expect(view.container.textContent).not.toContain(secret);
    expect(runtimeCredentialBody).toEqual({
      credentialId: "otel-debug",
      kind: "otlp-headers@1",
      material: { headers: { Authorization: secret } },
    });

    await user.type(screen.getByLabelText("RuntimeConfig name"), "debug");
    const telemetryGroup = screen.getByRole("group", {
      name: /Worker telemetry/,
    });
    await user.click(within(telemetryGroup).getByRole("checkbox"));
    await user.type(
      within(telemetryGroup).getByLabelText("OTLP traces endpoint"),
      "https://otel.example/v1/traces",
    );
    await user.type(
      within(telemetryGroup).getByLabelText("Runtime credential ID (optional)"),
      "otel-debug",
    );
    await user.click(
      screen.getByRole("button", {
        name: "Publish immutable RuntimeConfig",
      }),
    );
    expect(await screen.findByText(/Published debug@1/)).toBeInTheDocument();

    const labelForm = view.container.querySelector(
      "form.runtime-label-create",
    ) as HTMLFormElement;
    await user.type(within(labelForm).getByLabelText("New label"), "debug");
    await user.selectOptions(
      within(labelForm).getByLabelText("Exact RuntimeConfig"),
      `${publishedResource!.ref.name}@${publishedResource!.ref.version}:${publishedResource!.ref.digest}`,
    );
    await user.click(
      within(labelForm).getByRole("button", { name: "Create binding" }),
    );
    const debugCard = (
      await screen.findByText("debug", { selector: "strong" })
    ).closest("article") as HTMLElement;
    expect(within(debugCard).getByText(/revision 1/)).toBeInTheDocument();
    await user.click(
      within(debugCard).getByRole("button", {
        name: "Rebind with current revision",
      }),
    );
    expect(
      await within(debugCard).findByText("Binding changed in another view."),
    ).toBeInTheDocument();
    expect(
      within(debugCard).getByRole("button", {
        name: "Reload authoritative binding",
      }),
    ).toBeInTheDocument();
  });

  it("shows durable offline and adapter-incompatible principals as separate facts", async () => {
    const principalBase = {
      labels: ["debug"],
      revision: "2",
      requiredRuntimeAdapters: ["otlp-http@1"],
      createdBy: "system",
      createdAt: "2026-08-31T12:00:00Z",
      updatedBy: "user_local",
      updatedAt: "2026-08-31T12:01:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const authenticated = sessionResponse(request);
        if (authenticated !== undefined) return authenticated;
        const path = new URL(request.url).pathname;
        if (path === "/v1/operations/snapshot") return apiResponse(snapshot());
        if (path === "/v1/operations/runtime-labels") {
          return apiResponse({
            items: [
              {
                label: "debug",
                config: { name: "debug", version: "1", digest },
                revision: "1",
                createdBy: "user_local",
                createdAt: "2026-08-31T12:00:00Z",
                updatedBy: "user_local",
                updatedAt: "2026-08-31T12:00:00Z",
              },
            ],
            page: { hasMore: false },
          });
        }
        if (path === "/v1/operations/runtime-agent-principals") {
          return apiResponse({
            items: [
              {
                ...principalBase,
                runtimeAgentId: "a".repeat(64),
                availability: "offline",
                missingRuntimeAdapters: [],
              },
              {
                ...principalBase,
                runtimeAgentId: "b".repeat(64),
                availability: "adapter_capability_mismatch",
                missingRuntimeAdapters: ["otlp-http@1"],
                live: {
                  instanceId: "runtime-incompatible",
                  softwareVersion: "0.1.0",
                  supportedRuntimes: ["adk@1"],
                  supportedToolsets: [],
                  supportedSandboxProfiles: ["none@1"],
                  supportedRuntimeAdapters: [],
                  observedState: "idle",
                  slotState: "idle",
                },
              },
            ],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.method} ${request.url}`);
      }),
    );
    renderOperations(api, "/operations/runtime-agents");
    expect(
      await screen.findByText("offline · durable labels retained"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/cannot receive matching work/),
    ).toBeInTheDocument();
    expect(
      screen.getByText("live as runtime-incompatible"),
    ).toBeInTheDocument();
  });
});
