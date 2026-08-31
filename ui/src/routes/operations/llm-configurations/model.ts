import type {
  ConfigurationResource,
  LLMGatewayBody,
  ModelPolicyBody,
} from "../../../api/operations";

export type ModelPolicyConsumer = "worker" | "planner" | "both";

export function modelPolicyBody(
  resource: ConfigurationResource,
): ModelPolicyBody {
  return resource.body as ModelPolicyBody;
}

export function llmGatewayBody(
  resource: ConfigurationResource,
): LLMGatewayBody {
  return resource.body as LLMGatewayBody;
}

function positiveInteger(value: number | undefined, maximum: number): boolean {
  return (
    value === undefined ||
    (Number.isSafeInteger(value) && value >= 1 && value <= maximum)
  );
}

export function validateModelPolicy(
  value: ModelPolicyBody,
  consumer: ModelPolicyConsumer,
): string[] {
  const errors: string[] = [];
  if (value.model.trim().length === 0 || value.model.length > 256) {
    errors.push("Model alias must contain 1–256 non-whitespace characters.");
  }
  const fields = [
    ["Maximum output tokens", value.maxOutputTokens, Number.MAX_SAFE_INTEGER],
    ["Maximum model calls", value.maxModelCalls, 1_000],
    ["Maximum tool calls", value.maxToolCalls, 10_000],
    ["Maximum Worker calls", value.maxWorkerCalls, 10_000],
    ["Maximum total tokens", value.maxTotalTokens, 100_000_000],
  ] as const;
  for (const [label, field, maximum] of fields) {
    if (!positiveInteger(field, maximum)) {
      errors.push(
        `${label} must be a positive integer no greater than ${maximum.toLocaleString()}.`,
      );
    }
  }
  if (
    value.temperature !== undefined &&
    (!Number.isFinite(value.temperature) || value.temperature < 0)
  ) {
    errors.push(
      "Temperature must be a finite number greater than or equal to zero.",
    );
  }
  const requireField = (
    field: number | undefined,
    label: string,
    role: string,
  ) => {
    if (field === undefined) {
      errors.push(`${label} is required for ${role}.`);
    }
  };
  if (consumer === "worker" || consumer === "both") {
    requireField(
      value.maxOutputTokens,
      "Maximum output tokens",
      "a Worker policy",
    );
    requireField(value.maxModelCalls, "Maximum model calls", "a Worker policy");
    requireField(
      value.maxToolCalls,
      "Maximum tool calls",
      "a tool-bearing Worker policy",
    );
    requireField(
      value.maxTotalTokens,
      "Maximum total tokens",
      "a Worker policy",
    );
  }
  if (consumer === "planner" || consumer === "both") {
    requireField(
      value.maxOutputTokens,
      "Maximum output tokens",
      "a Planner policy",
    );
    requireField(
      value.maxModelCalls,
      "Maximum model calls",
      "a Planner policy",
    );
    requireField(
      value.maxWorkerCalls,
      "Maximum Worker calls",
      "a Planner policy",
    );
    requireField(
      value.maxTotalTokens,
      "Maximum total tokens",
      "a Planner policy",
    );
  }
  return [...new Set(errors)];
}

function parseHTTPURL(raw: string): URL | undefined {
  if (raw.length === 0 || raw !== raw.trim()) {
    return undefined;
  }
  try {
    const parsed = new URL(raw);
    if (
      (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
      parsed.hostname.length === 0 ||
      parsed.username !== "" ||
      parsed.password !== "" ||
      parsed.search !== "" ||
      parsed.hash !== ""
    ) {
      return undefined;
    }
    return parsed;
  } catch {
    return undefined;
  }
}

function isLoopbackIP(hostname: string): boolean {
  const normalized = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  if (normalized === "::1") {
    return true;
  }
  const octets = normalized.split(".");
  return (
    octets.length === 4 &&
    octets.every((octet) => /^\d{1,3}$/.test(octet) && Number(octet) <= 255) &&
    octets[0] === "127"
  );
}

export function validateLLMGateway(value: LLMGatewayBody): string[] {
  const errors: string[] = [];
  const inference = parseHTTPURL(value.url);
  if (
    inference === undefined ||
    value.url.length > 2048 ||
    !/^https?:\/\/(?:\[[^\]]+\]|[^/?#]+)\//i.test(value.url)
  ) {
    errors.push(
      "Inference URL must be an absolute HTTP(S) URL with an explicit path and no userinfo, query, or fragment.",
    );
  }
  if (value.protocol !== "openai-compatible@1") {
    errors.push("Protocol must be openai-compatible@1.");
  }
  if (value.credentialManager !== undefined) {
    const management = parseHTTPURL(value.credentialManager.managementUrl);
    if (
      value.credentialManager.implementation !== "litellm-virtual-keys@1" ||
      management === undefined ||
      value.credentialManager.managementUrl.length > 2048 ||
      management.pathname !== "/"
    ) {
      errors.push(
        "LiteLLM management URL must be a canonical HTTP(S) root origin.",
      );
    } else if (
      management.protocol === "http:" &&
      !isLoopbackIP(management.hostname)
    ) {
      errors.push(
        "HTTP credential management is allowed only for an IP-literal loopback origin.",
      );
    }
  }
  return errors;
}

export function hasCredentialManager(resource: ConfigurationResource): boolean {
  return (
    resource.ref.kind === "llm-gateways" &&
    llmGatewayBody(resource).credentialManager !== undefined
  );
}
