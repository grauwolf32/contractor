import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "./client";
import {
  getRuntimeConfig,
  publishRuntimeConfig,
  type RuntimeConfigAuthorDocument,
  type RuntimeConfigResource,
} from "./operations";

const config = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};
const digest = `sha256:${"a".repeat(64)}`;
const exactGateway = { gatewayId: "local-litellm", version: "1", digest };

function response(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function resource(llmGateway: unknown) {
  return {
    ref: { name: "typed", version: "1", digest },
    document: {
      apiVersion: "contractor/v1alpha1",
      kind: "RuntimeConfig",
      metadata: { name: "typed", version: "1" },
      spec: { worker: { llmGateway } },
    },
    builtIn: false,
    createdBy: "user-1",
    createdAt: "2026-09-19T00:00:00Z",
  };
}

describe("RuntimeConfig author and read contracts", () => {
  it("sends the author selector and clears, then preserves the resolved read ref", async () => {
    const author: RuntimeConfigAuthorDocument = {
      apiVersion: "contractor/v1alpha1",
      kind: "RuntimeConfig",
      metadata: { name: "typed", version: "1" },
      spec: {
        worker: {
          llmGateway: { gateway: "local-litellm@1", credential: null },
          telemetry: null,
          httpProxy: null,
          caido: null,
        },
        planner: { telemetry: null },
      },
    };
    const resolved: RuntimeConfigResource = {
      ref: { ...author.metadata, digest },
      document: {
        ...author,
        spec: {
          ...author.spec,
          worker: {
            ...author.spec.worker,
            llmGateway: { gateway: exactGateway, credential: null },
          },
        },
      },
      builtIn: false,
      createdBy: "user-1",
      createdAt: "2026-09-19T00:00:00Z",
    };
    const requests: Request[] = [];
    const api = new PublicAPI(
      config,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request.clone());
        return response(resolved, request.method === "POST" ? 201 : 200);
      }),
    );
    api.csrf.replace("a".repeat(43));
    expect(
      await publishRuntimeConfig(api, author, "runtime-author-test"),
    ).toEqual(resolved);
    expect(await requests[0]?.json()).toEqual(author);
    expect(requests[0]?.headers.get("Idempotency-Key")).toBe(
      "runtime-author-test",
    );
    expect(await getRuntimeConfig(api, "typed", "1")).toEqual(resolved);
  });

  it.each([
    { gateway: "local-litellm@1" },
    { gateway: null },
    null,
    { gateway: { ...exactGateway, token: "UNEXPECTED_SECRET" } },
  ])("rejects unsupported gateway read shapes: %j", async (gateway) => {
    const api = new PublicAPI(
      config,
      vi.fn(async () => response(resource(gateway))),
    );
    await expect(getRuntimeConfig(api, "typed", "1")).rejects.toThrow(
      "response shape",
    );
  });
});
