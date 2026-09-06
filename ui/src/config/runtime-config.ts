import {
  PUBLIC_API_VERSION,
  SUPPORTED_API_VERSIONS,
  UI_VERSION,
} from "../build";

const MAXIMUM_RUNTIME_CONFIG_BYTES = 8192;

export interface RuntimeConfig {
  uiVersion: string;
  supportedApiVersions: readonly string[];
  apiBaseUrl: string;
}

export class RuntimeConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RuntimeConfigError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isLoopbackIPLiteral(hostname: string): boolean {
  if (hostname === "[::1]") {
    return true;
  }
  const octets = hostname.split(".");
  return (
    octets.length === 4 &&
    octets[0] === "127" &&
    octets.every((octet) => /^(?:0|[1-9][0-9]{0,2})$/.test(octet)) &&
    octets.every((octet) => Number(octet) <= 255)
  );
}

function isPrivateIPv4Literal(hostname: string): boolean {
  const octets = hostname.split(".");
  if (
    octets.length !== 4 ||
    !octets.every((octet) => /^(?:0|[1-9][0-9]{0,2})$/.test(octet)) ||
    !octets.every((octet) => Number(octet) <= 255)
  ) {
    return false;
  }
  const first = Number(octets[0]);
  const second = Number(octets[1]);
  return (
    first === 10 ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168)
  );
}

function isExplicitDevelopmentHTTP(candidate: string, parsed: URL): boolean {
  if (
    parsed.protocol !== "http:" ||
    (!isLoopbackIPLiteral(parsed.hostname) &&
      !isPrivateIPv4Literal(parsed.hostname))
  ) {
    return false;
  }
  return candidate === parsed.origin || candidate === `${parsed.origin}/`;
}

export function validateAPIBaseURL(candidate: unknown): string {
  if (typeof candidate !== "string" || candidate !== candidate.trim()) {
    throw new RuntimeConfigError("API URL must be an absolute origin");
  }
  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new RuntimeConfigError("API URL must be an absolute origin");
  }
  if (
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.pathname !== "/" ||
    parsed.search !== "" ||
    parsed.hash !== ""
  ) {
    throw new RuntimeConfigError(
      "API URL cannot contain credentials, a path, query, or fragment",
    );
  }
  if (candidate !== parsed.origin && candidate !== `${parsed.origin}/`) {
    throw new RuntimeConfigError("API URL must use canonical origin syntax");
  }
  if (
    parsed.protocol !== "https:" &&
    !isExplicitDevelopmentHTTP(candidate, parsed)
  ) {
    throw new RuntimeConfigError(
      "API URL must use HTTPS, except for IP-literal local development",
    );
  }
  return parsed.origin;
}

export function parseRuntimeConfig(value: unknown): RuntimeConfig {
  if (!isRecord(value)) {
    throw new RuntimeConfigError("Runtime configuration must be an object");
  }
  const keys = Object.keys(value).sort();
  const expectedKeys = ["apiBaseUrl", "supportedApiVersions", "uiVersion"];
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new RuntimeConfigError("Runtime configuration has unknown fields");
  }
  if (value.uiVersion !== UI_VERSION) {
    throw new RuntimeConfigError("UI build and runtime configuration differ");
  }
  if (
    !Array.isArray(value.supportedApiVersions) ||
    value.supportedApiVersions.length !== SUPPORTED_API_VERSIONS.length ||
    !value.supportedApiVersions.every(
      (version, index) => version === SUPPORTED_API_VERSIONS[index],
    ) ||
    !value.supportedApiVersions.includes(PUBLIC_API_VERSION)
  ) {
    throw new RuntimeConfigError("Runtime configuration has incompatible APIs");
  }
  return Object.freeze({
    uiVersion: value.uiVersion,
    supportedApiVersions: Object.freeze([...value.supportedApiVersions]),
    apiBaseUrl: validateAPIBaseURL(value.apiBaseUrl),
  });
}

async function readBoundedJSON(response: Response): Promise<unknown> {
  const length = response.headers.get("content-length");
  if (length !== null && Number(length) > MAXIMUM_RUNTIME_CONFIG_BYTES) {
    throw new RuntimeConfigError("Runtime configuration is too large");
  }
  if (response.body === null) {
    throw new RuntimeConfigError("Runtime configuration is not valid JSON");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let size = 0;
  let text = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      size += value.byteLength;
      if (size > MAXIMUM_RUNTIME_CONFIG_BYTES) {
        await reader.cancel();
        throw new RuntimeConfigError("Runtime configuration is too large");
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } catch (error) {
    if (error instanceof RuntimeConfigError) {
      throw error;
    }
    throw new RuntimeConfigError("Runtime configuration is not valid UTF-8");
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new RuntimeConfigError("Runtime configuration is not valid JSON");
  }
}

export async function loadRuntimeConfig(
  fetchImplementation: typeof fetch = globalThis.fetch,
): Promise<RuntimeConfig> {
  let response: Response;
  try {
    response = await fetchImplementation("/runtime-config.json", {
      method: "GET",
      credentials: "omit",
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
  } catch {
    throw new RuntimeConfigError("Runtime configuration is unavailable");
  }
  if (!response.ok) {
    throw new RuntimeConfigError("Runtime configuration is unavailable");
  }
  return parseRuntimeConfig(await readBoundedJSON(response));
}
