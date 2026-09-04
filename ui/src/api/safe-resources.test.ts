import { describe, expect, it } from "vitest";

import type { ConfigurationResource, CredentialResource } from "./workflows";
import {
  safeConfigurationResource,
  safeCredentialResource,
} from "./safe-resources";

const digest = `sha256:${"1".repeat(64)}`;

describe("safe public resource projections", () => {
  it("drops undeclared configuration fields before cache or render state", () => {
    const wire = {
      ref: {
        kind: "llm-gateways",
        name: "local",
        version: "1",
        digest,
      },
      body: {
        protocol: "openai-compatible@1",
        url: "http://127.0.0.1:4000/v1",
        apiKey: "CONFIGURATION_CANARY",
      },
      source: "operator",
      rawManifest: "ANOTHER_CANARY",
    } as ConfigurationResource;
    const projected = safeConfigurationResource(wire);
    expect(projected).toEqual({
      ref: wire.ref,
      body: {
        protocol: "openai-compatible@1",
        url: "http://127.0.0.1:4000/v1",
      },
      source: "operator",
    });
    expect(JSON.stringify(projected)).not.toMatch(/CANARY/);
  });

  it("drops undeclared credential material before cache or render state", () => {
    const wire = {
      credentialId: "worker-budget",
      llmGateway: { gatewayId: "local", version: "1", digest },
      createdAt: "2026-08-31T12:00:00Z",
      effectivePolicy: {
        modelPolicies: [{ policyId: "worker", version: "1", digest }],
        models: ["qwen-worker"],
      },
      token: "CREDENTIAL_CANARY",
      remoteKeyId: "REMOTE_CANARY",
    } as CredentialResource;
    const projected = safeCredentialResource(wire);
    expect(projected).toEqual({
      credentialId: "worker-budget",
      llmGateway: { gatewayId: "local", version: "1", digest },
      createdAt: "2026-08-31T12:00:00Z",
      effectivePolicy: {
        modelPolicies: [{ policyId: "worker", version: "1", digest }],
        models: ["qwen-worker"],
      },
    });
    expect(JSON.stringify(projected)).not.toMatch(/CANARY/);
  });

  it("retains only the resolved terminal summarizer configuration", () => {
    const policy = { policyId: "summary", version: "1", digest };
    const wire = {
      ref: {
        kind: "agent-templates",
        name: "worker",
        version: "1",
        digest,
      },
      source: "operator",
      body: {
        description: "Worker",
        runtime: "adk@1",
        instructions: { ref: "instructions/worker.md", digest },
        modelPolicy: { policyId: "worker", version: "1", digest },
        summarizer: {
          modelPolicy: policy,
          contextWindowRatio: 0.9,
          cumulativeBudget: 20_000,
          credential: "SUMMARIZER_CANARY",
        },
        skills: [
          { namespace: "skills", name: "architecture-review" },
          { namespace: "skills", name: "openapi-analysis" },
        ],
        toolsets: [],
        sandboxProfile: "local-workdir@1",
      },
    } as ConfigurationResource;

    const projected = safeConfigurationResource(wire);

    expect(projected.body).toMatchObject({
      summarizer: {
        modelPolicy: policy,
        contextWindowRatio: 0.9,
        cumulativeBudget: 20_000,
      },
      skills: [
        { namespace: "skills", name: "architecture-review" },
        { namespace: "skills", name: "openapi-analysis" },
      ],
    });
    expect(JSON.stringify(projected)).not.toMatch(/CANARY/);
  });
});
