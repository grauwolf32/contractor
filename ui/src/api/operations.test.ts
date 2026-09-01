import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  createCredential,
  createRuntimeCredential,
  deleteCredential,
  deleteRuntimeCredential,
  deleteRuntimeAgentPrincipal,
  deleteRuntimeLabel,
  getConfiguration,
  getCredential,
  getOperationsSnapshot,
  getRuntimeConfig,
  getRuntimeCredential,
  getRuntimeLabel,
  getRuntimeAgentPrincipal,
  listAllocations,
  listRuntimeAgents,
  listRuntimeConfigs,
  listRuntimeCredentials,
  listRuntimeLabels,
  listRuntimeAgentPrincipals,
  publishConfiguration,
  publishRuntimeConfig,
  putRuntimeLabel,
  replaceRuntimeAgentPrincipalLabels,
  type CreateCredentialRequest,
} from "./operations";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const digest = `sha256:${"1".repeat(64)}`;
const gateway = { gatewayId: "local-litellm", version: "1", digest };
const policy = { policyId: "worker", version: "1", digest };

function response(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(value === undefined ? undefined : JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...headers,
    },
  });
}

describe("Operations API", () => {
  it("reads one authoritative snapshot and cursor-pinned pages", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const path = new URL(request.url).pathname;
        if (path === "/v1/operations/snapshot") {
          return response({
            cursor: { generation: "operations-generation-1", revision: "7" },
            runtimeAgents: [],
            allocations: [],
          });
        }
        if (path === "/v1/operations/runtime-agents") {
          return response({
            cursor: { generation: "operations-generation-1", revision: "7" },
            items: [],
            page: { hasMore: false },
          });
        }
        if (path === "/v1/operations/allocations") {
          return response({
            cursor: { generation: "operations-generation-1", revision: "7" },
            items: [],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.url}`);
      }),
    );
    await getOperationsSnapshot(api);
    await listRuntimeAgents(api, { cursor: "agent-next" });
    await listAllocations(api, { cursor: "allocation-next" });
    expect(requests[0]?.url).toBe(
      "http://127.0.0.1:8080/v1/operations/snapshot",
    );
    expect(new URL(requests[1]!.url).searchParams).toEqual(
      new URLSearchParams({ limit: "50", cursor: "agent-next" }),
    );
    expect(new URL(requests[2]!.url).searchParams).toEqual(
      new URLSearchParams({ limit: "50", cursor: "allocation-next" }),
    );
  });

  it("copies, validates, and normalizes Runtime Agent capabilities", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          cursor: { generation: "operations-generation-1", revision: "7" },
          runtimeAgents: [
            {
              instanceId: "runtime-capable",
              softwareVersion: "0.1.0",
              supportedRuntimes: ["python@1", "adk@1"],
              supportedToolsets: [
                {
                  ref: "source-analysis@1",
                  tools: ["search_source", "read_source"],
                },
              ],
              supportedSandboxProfiles: ["remote@1", "local-workdir@1"],
              supportedRuntimeAdapters: ["otlp-http@1"],
              observedState: "idle",
              slotState: "idle",
            },
            {
              instanceId: "runtime-minimal",
              softwareVersion: "0.1.0",
              supportedRuntimes: ["adk@1"],
              supportedToolsets: [],
              supportedSandboxProfiles: ["local-workdir@1"],
              supportedRuntimeAdapters: [],
              observedState: "idle",
              slotState: "idle",
            },
          ],
          allocations: [],
        }),
      ),
    );

    const result = await getOperationsSnapshot(api);
    expect(result.runtimeAgents[0]?.supportedRuntimes).toEqual([
      "adk@1",
      "python@1",
    ]);
    expect(result.runtimeAgents[0]?.supportedToolsets[0]?.tools).toEqual([
      "read_source",
      "search_source",
    ]);
    expect(result.runtimeAgents[1]?.supportedToolsets).toEqual([]);
    const unexpected = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          cursor: { generation: "operations-generation-1", revision: "7" },
          runtimeAgents: [
            {
              instanceId: "runtime-capable",
              softwareVersion: "0.1.0",
              supportedRuntimes: ["adk@1"],
              supportedToolsets: [],
              supportedSandboxProfiles: ["local-workdir@1"],
              supportedRuntimeAdapters: [],
              observedState: "idle",
              slotState: "idle",
              localPath: "/private/runtime/path",
            },
          ],
          allocations: [],
        }),
      ),
    );
    await expect(getOperationsSnapshot(unexpected)).rejects.toThrow(
      "response shape",
    );
  });

  it("fails closed on malformed Runtime Agent capabilities", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          cursor: { generation: "operations-generation-1", revision: "7" },
          runtimeAgents: [
            {
              instanceId: "runtime-malformed",
              softwareVersion: "0.1.0",
              supportedRuntimes: ["adk@1", "adk@1"],
              supportedToolsets: [],
              supportedSandboxProfiles: ["local-workdir@1"],
              supportedRuntimeAdapters: [],
              observedState: "idle",
              slotState: "idle",
            },
          ],
          allocations: [],
        }),
      ),
    );

    await expect(getOperationsSnapshot(api)).rejects.toThrow(
      "Runtime Agent capabilities are invalid",
    );
  });

  it("gets and publishes exact immutable configuration versions", async () => {
    const requests: Request[] = [];
    const resource = {
      ref: {
        kind: "model-policies" as const,
        name: "worker",
        version: "2",
        digest,
      },
      body: {
        model: "qwen/qwen3.8-27b",
        maxOutputTokens: 4096,
        maxModelCalls: 8,
        maxToolCalls: 16,
        maxTotalTokens: 32768,
      },
      source: "managed" as const,
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        return response(resource, request.method === "POST" ? 201 : 200);
      }),
    );
    api.csrf.replace("a".repeat(43));
    await expect(
      getConfiguration(api, "model-policies", "worker", "2"),
    ).resolves.toEqual(resource);
    await expect(
      publishConfiguration(
        api,
        "model-policies",
        { name: "worker", version: "2", modelPolicy: resource.body },
        "configuration-draft-1",
      ),
    ).resolves.toEqual(resource);
    expect(requests[1]?.headers.get("Idempotency-Key")).toBe(
      "configuration-draft-1",
    );
    expect(requests[1]?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
  });

  it("creates, reads, and deletes only secret-free credential metadata", async () => {
    const requests: Request[] = [];
    const credential = {
      credentialId: "worker-budget",
      llmGateway: gateway,
      label: "Worker budget",
      createdAt: "2026-08-31T12:00:00Z",
      effectivePolicy: {
        modelPolicies: [policy],
        models: ["qwen/qwen3.8-27b"],
        maxBudget: 10,
        budgetDuration: "1d",
      },
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        if (request.method === "DELETE") {
          return response(undefined, 204);
        }
        return response(credential, request.method === "POST" ? 201 : 200);
      }),
    );
    api.csrf.replace("a".repeat(43));
    const request: CreateCredentialRequest = {
      credentialId: "worker-budget",
      llmGateway: gateway,
      label: "Worker budget",
      gatewayPolicy: {
        modelPolicies: [policy],
        maxBudget: 10,
        budgetDuration: "1d",
      },
    };
    await createCredential(api, request, "credential-create-1");
    await getCredential(api, "worker-budget");
    await deleteCredential(api, "worker-budget", "credential-delete-1");
    expect(await requests[0]?.clone().json()).toEqual(request);
    expect(requests[0]?.headers.get("Idempotency-Key")).toBe(
      "credential-create-1",
    );
    expect(requests[2]?.headers.get("Idempotency-Key")).toBe(
      "credential-delete-1",
    );
    expect(JSON.stringify(credential)).not.toMatch(/token|cipher|remoteKey/i);
  });

  it("rejects malformed snapshot cursors and read-only publication", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          cursor: { generation: "operations-generation-1", revision: "01" },
          runtimeAgents: [],
          allocations: [],
        }),
      ),
    );
    await expect(getOperationsSnapshot(api)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    await expect(
      publishConfiguration(
        api,
        "agent-templates" as "model-policies",
        { name: "worker", version: "2", modelPolicy: { model: "m" } },
        "readonly-1",
      ),
    ).rejects.toThrow("read-only");
  });

  it("manages durable Runtime Agent principals and rejects unknown fields", async () => {
    const runtimeAgentId = "a".repeat(64);
    const requests: Request[] = [];
    const principal = {
      runtimeAgentId,
      labels: ["debug"],
      revision: "2",
      availability: "adapter_capability_mismatch" as const,
      requiredRuntimeAdapters: ["otlp-http@1"],
      missingRuntimeAdapters: ["otlp-http@1"],
      live: {
        instanceId: "runtime-principal",
        softwareVersion: "0.1.0",
        supportedRuntimes: ["adk@1"],
        supportedToolsets: [],
        supportedSandboxProfiles: ["local-workdir@1"],
        supportedRuntimeAdapters: [],
        observedState: "idle" as const,
        slotState: "idle" as const,
      },
      createdBy: "runtime-registration",
      createdAt: "2026-09-01T00:00:00Z",
      updatedBy: "user-1",
      updatedAt: "2026-09-01T00:01:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        if (request.method === "DELETE") {
          return response(undefined, 204);
        }
        if (request.method === "PUT") {
          return response(principal, 200, { ETag: '"2"' });
        }
        if (new URL(request.url).pathname.endsWith(runtimeAgentId)) {
          return response(principal, 200, { ETag: '"2"' });
        }
        return response({ items: [principal], page: { hasMore: false } });
      }),
    );
    api.csrf.replace("a".repeat(43));

    const values = [
      await listRuntimeAgentPrincipals(api),
      await getRuntimeAgentPrincipal(api, runtimeAgentId),
      await replaceRuntimeAgentPrincipalLabels(
        api,
        runtimeAgentId,
        ["debug"],
        "1",
        "principal-labels-1",
      ),
    ];
    await deleteRuntimeAgentPrincipal(
      api,
      runtimeAgentId,
      "2",
      "principal-delete-1",
    );

    expect(JSON.stringify(values)).not.toMatch(/certificate|privateKey/i);
    expect(await requests[2]?.clone().json()).toEqual({ labels: ["debug"] });
    expect(requests[2]?.headers.get("If-Match")).toBe('"1"');
    expect(requests[3]?.headers.get("If-Match")).toBe('"2"');

    const unexpected = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          items: [
            { ...principal, certificatePem: "SERVER_CERTIFICATE_CANARY" },
          ],
          page: { hasMore: false },
        }),
      ),
    );
    await expect(listRuntimeAgentPrincipals(unexpected)).rejects.toThrow(
      "response shape",
    );
  });

  it("manages RuntimeConfig, labels, and write-only Runtime credentials", async () => {
    const runtimeDigest = `sha256:${"2".repeat(64)}`;
    const document = {
      apiVersion: "contractor/v1alpha1" as const,
      kind: "RuntimeConfig" as const,
      metadata: { name: "debug", version: "1" },
      spec: {
        worker: {
          telemetry: {
            adapter: "otlp-http@1" as const,
            endpoint: "https://otel.example/v1/traces",
            captureContent: false as const,
            flushTimeoutSeconds: 5,
          },
        },
      },
    };
    const resource = {
      ref: { name: "debug", version: "1", digest: runtimeDigest },
      document,
      builtIn: false,
      createdBy: "user-1",
      createdAt: "2026-09-01T00:00:00Z",
    };
    const binding = {
      label: "debug",
      config: resource.ref,
      revision: "1",
      createdBy: "user-1",
      createdAt: "2026-09-01T00:00:00Z",
      updatedBy: "user-1",
      updatedAt: "2026-09-01T00:00:00Z",
    };
    const metadata = {
      credentialId: "otel-debug",
      kind: "otlp-headers@1" as const,
      createdBy: "user-1",
      createdAt: "2026-09-01T00:00:00Z",
    };
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const path = new URL(request.url).pathname;
        if (request.method === "DELETE") {
          return response(undefined, 204);
        }
        if (path === "/v1/operations/runtime-configs") {
          return request.method === "POST"
            ? response(resource, 201)
            : response({ items: [resource], page: { hasMore: false } });
        }
        if (path.includes("/runtime-configs/")) {
          return response(resource);
        }
        if (path === "/v1/operations/runtime-labels") {
          return response({ items: [binding], page: { hasMore: false } });
        }
        if (path.endsWith("/runtime-labels/debug")) {
          return response(binding, request.method === "PUT" ? 201 : 200, {
            ETag: '"1"',
          });
        }
        if (path === "/v1/operations/runtime-credentials") {
          return request.method === "POST"
            ? response(metadata, 201)
            : response({ items: [metadata], page: { hasMore: false } });
        }
        if (path.endsWith("/runtime-credentials/otel-debug")) {
          return response(metadata);
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    api.csrf.replace("a".repeat(43));

    const values = [
      await publishRuntimeConfig(api, document, "runtime-config-create-1"),
      await getRuntimeConfig(api, "debug", "1"),
      await listRuntimeConfigs(api),
      await putRuntimeLabel(
        api,
        "debug",
        resource.ref,
        "runtime-label-create-1",
      ),
      await getRuntimeLabel(api, "debug"),
      await listRuntimeLabels(api),
      await createRuntimeCredential(
        api,
        {
          credentialId: "otel-debug",
          kind: "otlp-headers@1",
          material: { headers: { authorization: "WRITE_ONLY_CANARY" } },
        },
        "runtime-credential-create-1",
      ),
      await getRuntimeCredential(api, "otel-debug"),
      await listRuntimeCredentials(api),
    ];
    await deleteRuntimeLabel(api, "debug", "1", "runtime-label-delete-1");
    await deleteRuntimeCredential(
      api,
      "otel-debug",
      "runtime-credential-delete-1",
    );

    expect(JSON.stringify(values)).not.toMatch(/WRITE_ONLY_CANARY/);
    expect(
      requests.some((request) => request.headers.get("If-None-Match") === "*"),
    ).toBe(true);
    expect(
      requests.some((request) => request.headers.get("If-Match") === '"1"'),
    ).toBe(true);

    const malformedAPI = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({
          items: [
            {
              ...resource,
              document: { ...document, leakedToken: "MUST_FAIL_CLOSED" },
            },
          ],
          page: { hasMore: false },
        }),
      ),
    );
    await expect(listRuntimeConfigs(malformedAPI)).rejects.toThrow(
      "response shape",
    );
  });
});
