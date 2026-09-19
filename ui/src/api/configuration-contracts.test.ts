import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "./client";
import type { components } from "./generated/public";
import {
  getRuntimeConfig,
  listRuntimeConfigs,
  publishRuntimeConfig,
} from "./operations";
import { safeConfigurationResource } from "./safe-resources";

const digest = `sha256:${"1".repeat(64)}`;
const config = {
  uiVersion: "0.3.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function reply(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

describe("published configuration optional fields", () => {
  it("preserves detached exact summarizer instructions without prompt content", () => {
    const instructions = {
      ref: "instructions/summarizer.md",
      digest,
      text: "PRIVATE_PROMPT_CANARY",
    };
    const resource: components["schemas"]["ConfigurationResource"] = {
      ref: { kind: "agent-templates", name: "worker", version: "1", digest },
      source: "operator",
      body: {
        description: "Worker",
        runtime: "adk@1",
        instructions: { ref: "instructions/worker.md", digest },
        modelPolicy: { policyId: "worker", version: "1", digest },
        toolsets: [],
        sandboxProfile: "local-workdir@1",
        summarizer: {
          modelPolicy: { policyId: "summary", version: "1", digest },
          contextWindowRatio: 0.8,
          instructions,
        },
      },
    };
    const safe = safeConfigurationResource(resource);
    const body = safe.body as components["schemas"]["AgentTemplateBody"];
    expect(body.summarizer?.instructions).toEqual({
      ref: instructions.ref,
      digest,
    });
    expect(body.summarizer?.instructions).not.toBe(instructions);
    expect(JSON.stringify(safe)).not.toContain("PRIVATE_PROMPT_CANARY");
  });

  it.each([
    undefined,
    {},
    { initialBackoffMilliseconds: 17, maxBackoffMilliseconds: 43 },
    { initialBackoffMilliseconds: 1, maxBackoffMilliseconds: 1 },
    { initialBackoffMilliseconds: 60000, maxBackoffMilliseconds: 60000 },
  ])(
    "preserves telemetry retry %j through publication and reads",
    async (retry) => {
      const document = {
        apiVersion: "contractor/v1alpha1" as const,
        kind: "RuntimeConfig" as const,
        metadata: { name: "retry", version: "1" },
        spec: {
          worker: {
            telemetry: {
              adapter: "otlp-http@1" as const,
              endpoint: "https://collector.example/v1/traces",
              export: { ...(retry === undefined ? {} : { retry }) },
            },
          },
        },
      };
      const resource = {
        ref: { ...document.metadata, digest },
        document,
        builtIn: false,
        createdBy: "user-1",
        createdAt: "2026-09-19T00:00:00Z",
      };
      const requests: Request[] = [];
      const api = new PublicAPI(
        config,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          requests.push(request);
          if (request.method === "POST") return reply(resource, 201);
          return reply(
            new URL(request.url).pathname.endsWith("/runtime-configs")
              ? { items: [resource], page: { hasMore: false } }
              : resource,
          );
        }),
      );
      api.csrf.replace("a".repeat(43));
      expect(
        await publishRuntimeConfig(api, document, "publish-retry"),
      ).toEqual(resource);
      expect(await requests[0]!.json()).toEqual(document);
      expect(await getRuntimeConfig(api, "retry", "1")).toEqual(resource);
      expect((await listRuntimeConfigs(api)).items).toEqual([resource]);
    },
  );

  it.each([
    null,
    { token: "RETRY_SECRET_CANARY" },
    { initialBackoffMilliseconds: null },
    { initialBackoffMilliseconds: 0 },
    { initialBackoffMilliseconds: 1.5 },
    { initialBackoffMilliseconds: 1001 },
    { maxBackoffMilliseconds: 60001 },
    { maxBackoffMilliseconds: 99 },
    { initialBackoffMilliseconds: 43, maxBackoffMilliseconds: 17 },
  ])("rejects malformed telemetry retry responses: %j", async (retry) => {
    const resource = {
      ref: { name: "retry", version: "1", digest },
      document: {
        apiVersion: "contractor/v1alpha1",
        kind: "RuntimeConfig",
        metadata: { name: "retry", version: "1" },
        spec: {
          worker: {
            telemetry: {
              adapter: "otlp-http@1",
              endpoint: "https://collector.example/v1/traces",
              export: { retry },
            },
          },
        },
      },
      builtIn: false,
      createdBy: "user-1",
      createdAt: "2026-09-19T00:00:00Z",
    };
    const api = new PublicAPI(
      config,
      vi.fn(async () => reply(resource)),
    );
    await expect(getRuntimeConfig(api, "retry", "1")).rejects.toThrow(
      TypeError,
    );
  });
});
