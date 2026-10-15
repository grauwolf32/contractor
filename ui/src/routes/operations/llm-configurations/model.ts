import { schemaLimit } from "../../../api/contract";
import type {
  ConfigurationResource,
  LLMGatewayBody,
  ModelPolicyBody,
} from "../../../api/operations";
import { formatNumber } from "../../../app/format";

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

/** A ModelPolicyBody bound from the public contract. */
export function policyLimit(
  keyword: "maximum" | "maxLength",
  property: keyof ModelPolicyBody,
): number {
  return schemaLimit("ModelPolicyBody", keyword, "properties", property);
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
  const modelLength = policyLimit("maxLength", "model");
  if (value.model.trim().length === 0 || value.model.length > modelLength) {
    errors.push(
      `Model alias must contain 1–${modelLength} non-whitespace characters.`,
    );
  }
  const fields = [
    ["Context window tokens", value.contextWindowTokens, "contextWindowTokens"],
    ["Maximum output tokens", value.maxOutputTokens, "maxOutputTokens"],
    ["Maximum model calls", value.maxModelCalls, "maxModelCalls"],
    ["Maximum tool calls", value.maxToolCalls, "maxToolCalls"],
    ["Maximum Worker calls", value.maxWorkerCalls, "maxWorkerCalls"],
    ["Maximum total tokens", value.maxTotalTokens, "maxTotalTokens"],
  ] as const;
  for (const [label, field, property] of fields) {
    const maximum = policyLimit("maximum", property);
    if (!positiveInteger(field, maximum)) {
      errors.push(
        `${label} must be a positive integer no greater than ${formatNumber(maximum)}.`,
      );
    }
  }
  if (
    value.contextWindowTokens !== undefined &&
    value.maxOutputTokens !== undefined &&
    value.maxOutputTokens >= value.contextWindowTokens
  ) {
    errors.push("Maximum output tokens must be below the context window.");
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

export const GATEWAY_URL_LENGTH = schemaLimit(
  "LLMGatewayBody",
  "maxLength",
  "properties",
  "url",
);
export const MANAGEMENT_URL_LENGTH = schemaLimit(
  "CredentialManagerDescriptor",
  "maxLength",
  "properties",
  "managementUrl",
);

export function validateLLMGateway(value: LLMGatewayBody): string[] {
  const errors: string[] = [];
  const inference = parseHTTPURL(value.url);
  if (
    inference === undefined ||
    value.url.length > GATEWAY_URL_LENGTH ||
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
      value.credentialManager.managementUrl.length > MANAGEMENT_URL_LENGTH ||
      management.pathname !== "/"
    ) {
      errors.push("LiteLLM management URL must be an HTTP(S) root origin.");
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
