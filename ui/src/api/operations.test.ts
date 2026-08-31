import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  createCredential,
  deleteCredential,
  getConfiguration,
  getCredential,
  getOperationsSnapshot,
  listAllocations,
  listRuntimeAgents,
  publishConfiguration,
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

function response(value: unknown, status = 200): Response {
  return new Response(value === undefined ? undefined : JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
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
});
