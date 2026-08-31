import { act, render, screen, waitFor } from "@testing-library/react";
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

function apiResponse(value: unknown, status = 200): Response {
  return new Response(value === undefined ? undefined : JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
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
});
