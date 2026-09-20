import type { components } from "./generated/public";

type Location = components["schemas"]["FindingLocation"];

// Retained finding-proposal.v1 bounds, specified in docs/spec/27.
const MAX_LOCATIONS = 256;
const MAX_PATH_BYTES = 4096;
const MAX_URL_BYTES = 8192;
const MAX_METHOD_BYTES = 64;
const HTTP_TOKEN = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;
const CONTROL = /\p{Cc}/u;
const bytes = (value: string) => new TextEncoder().encode(value).length;

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function line(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function webURL(value: unknown): value is string {
  if (
    typeof value !== "string" ||
    bytes(value) > MAX_URL_BYTES ||
    /\s|\\/u.test(value) ||
    CONTROL.test(value) ||
    /%(?![0-9a-f]{2})/i.test(value)
  )
    return false;
  try {
    const url = new URL(value);
    const authority = value.split("/")[2];
    if (authority === undefined) return false;
    return (
      /^https?:\/\//i.test(value) &&
      Boolean(url.hostname) &&
      !url.username &&
      !url.password &&
      !authority.includes("@") &&
      !authority.endsWith(":") &&
      url.port !== "0"
    );
  } catch {
    return false;
  }
}

export function isFindingLocation(value: unknown): value is Location {
  if (!object(value)) return false;
  if ("file" in value) {
    if (
      Object.keys(value).some(
        (key) => !["file", "line", "range"].includes(key),
      ) ||
      typeof value.file !== "string" ||
      !value.file ||
      bytes(value.file) > MAX_PATH_BYTES ||
      /[\\:]/.test(value.file) ||
      CONTROL.test(value.file) ||
      value.file.split("/").some((part) => ["", ".", ".."].includes(part))
    )
      return false;
    if ("line" in value && "range" in value) return false;
    if ("line" in value && !line(value.line)) return false;
    if ("range" in value) {
      const range = value.range;
      if (
        !object(range) ||
        Object.keys(range).length !== 2 ||
        !line(range.start_line) ||
        !line(range.end_line) ||
        range.end_line < range.start_line
      )
        return false;
    }
    return true;
  }
  return (
    Object.keys(value).every((key) => ["url", "method"].includes(key)) &&
    webURL(value.url) &&
    (!("method" in value) ||
      (typeof value.method === "string" &&
        value.method.length <= MAX_METHOD_BYTES &&
        HTTP_TOKEN.test(value.method)))
  );
}

export function validFindingCoordinates(document: unknown): boolean {
  if (
    !object(document) ||
    document.schema !== "contractor.audit.finding-proposal.v1"
  )
    return false;
  if (!("locations" in document)) return true;
  return (
    Array.isArray(document.locations) &&
    document.locations.length <= MAX_LOCATIONS &&
    document.locations.every(isFindingLocation)
  );
}
