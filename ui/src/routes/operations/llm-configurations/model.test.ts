import { describe, expect, it } from "vitest";

import { validateLLMGateway, validateModelPolicy } from "./model";

describe("Operations configuration validation", () => {
  it("requires finite limits for the selected ModelPolicy consumers", () => {
    const worker = {
      model: "qwen-worker",
      maxOutputTokens: 4096,
      maxModelCalls: 8,
      maxToolCalls: 16,
      maxTotalTokens: 32768,
    };
    expect(validateModelPolicy(worker, "worker")).toEqual([]);
    expect(validateModelPolicy(worker, "planner")).toContain(
      "Maximum Worker calls is required for a Planner policy.",
    );
    expect(
      validateModelPolicy(
        {
          ...worker,
          maxWorkerCalls: 16,
          temperature: Number.POSITIVE_INFINITY,
        },
        "both",
      ),
    ).toContain(
      "Temperature must be a finite number greater than or equal to zero.",
    );
  });

  it("mirrors the inference and LiteLLM management URL boundaries", () => {
    expect(
      validateLLMGateway({
        protocol: "openai-compatible@1",
        url: "http://127.0.0.1:4000/v1",
        credentialManager: {
          implementation: "litellm-virtual-keys@1",
          managementUrl: "http://127.0.0.1:4000",
        },
      }),
    ).toEqual([]);
    expect(
      validateLLMGateway({
        protocol: "openai-compatible@1",
        url: "https://gateway.example.test",
      }),
    ).toContain(
      "Inference URL must be an absolute HTTP(S) URL with an explicit path and no userinfo, query, or fragment.",
    );
    expect(
      validateLLMGateway({
        protocol: "openai-compatible@1",
        url: "https://gateway.example.test/v1",
        credentialManager: {
          implementation: "litellm-virtual-keys@1",
          managementUrl: "http://localhost:4000",
        },
      }),
    ).toContain(
      "HTTP credential management is allowed only for an IP-literal loopback origin.",
    );
  });
});
