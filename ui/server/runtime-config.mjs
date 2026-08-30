import { readFileSync } from "node:fs";

export const PUBLIC_API_VERSION = "contractor.public.v1";
export const SUPPORTED_API_VERSIONS = Object.freeze([PUBLIC_API_VERSION]);

const packageMetadata = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
);
export const UI_VERSION = packageMetadata.version;

export class RuntimeConfigError extends Error {
  constructor(message) {
    super(message);
    this.name = "RuntimeConfigError";
  }
}

function isLoopbackIPLiteral(hostname) {
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

function isExplicitLoopbackHTTP(candidate, parsed) {
  if (parsed.protocol !== "http:" || !isLoopbackIPLiteral(parsed.hostname)) {
    return false;
  }
  return /^http:\/\/(?:\[::1\]|127\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})(?::[0-9]{1,5})?\/?$/.test(
    candidate,
  );
}

export function validateAPIBaseURL(candidate) {
  if (
    typeof candidate !== "string" ||
    candidate.length === 0 ||
    candidate !== candidate.trim()
  ) {
    throw new RuntimeConfigError("API URL must be an absolute origin");
  }
  let parsed;
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
    !isExplicitLoopbackHTTP(candidate, parsed)
  ) {
    throw new RuntimeConfigError(
      "API URL must use HTTPS, except for IP-literal loopback development",
    );
  }
  return parsed.origin;
}

function parsePort(candidate) {
  if (!/^(?:[1-9][0-9]{0,4})$/.test(candidate)) {
    throw new RuntimeConfigError("UI listen port must be between 1 and 65535");
  }
  const port = Number(candidate);
  if (port > 65_535) {
    throw new RuntimeConfigError("UI listen port must be between 1 and 65535");
  }
  return port;
}

export function runtimeSettingsFromEnvironment(environment = process.env) {
  const apiBaseUrl = validateAPIBaseURL(environment.CONTRACTOR_UI_API_BASE_URL);
  const host = environment.CONTRACTOR_UI_HOST ?? "127.0.0.1";
  if (host.length === 0 || host.length > 255 || /[\s/\\?#@]/.test(host)) {
    throw new RuntimeConfigError("UI listen host is invalid");
  }
  const port = parsePort(environment.CONTRACTOR_UI_PORT ?? "4173");
  return Object.freeze({
    host,
    port,
    distDir: environment.CONTRACTOR_UI_DIST_DIR,
    runtimeConfig: Object.freeze({
      uiVersion: UI_VERSION,
      supportedApiVersions: SUPPORTED_API_VERSIONS,
      apiBaseUrl,
    }),
  });
}
