import { parseDocument } from "yaml";
import { MAXIMUM_PREVIEW_BYTES } from "./artifacts";

export const HTTP_EXCHANGE_SCHEMA = "contractor.http.exchange.v1";
export const HTTP_CAPTURE_LIMITS = {
  requestBodyBytes: 1024 * 1024,
  responseBodyBytes: 16 * 1024 * 1024,
  headerBytes: 64 * 1024,
  attempts: 13,
} as const;

export interface HTTPHeader {
  name: string;
  value: string;
}
export type HTTPMessage =
  | { kind: "structured"; headers: HTTPHeader[]; body_base64: string }
  | { kind: "raw"; data_base64: string };
interface HTTPAttempt {
  request: { method: string; url: string; message: HTTPMessage };
  response?: { status: number; message: HTTPMessage; complete: boolean };
  error?: string;
}
export interface HTTPCapture {
  source: "http" | "caido";
  request_id: string;
  request_tag?: string;
  attempts: HTTPAttempt[];
}
export interface HTTPExchangeDocument {
  schema: typeof HTTP_EXCHANGE_SCHEMA;
  capture: HTTPCapture;
  finding: { proposal_id: string; receipt_id: string };
}

const TOKEN = /^[!#$%&'*+.^_\x60|~0-9A-Za-z-]+(?![\s\S])/;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}(?![\s\S])/;
const BASE64 =
  /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const OUTCOMES = new Set([
  "transport_error",
  "timeout",
  "cancelled",
  "response_invalid",
  "response_too_large",
  "response_unavailable",
  "target_denied",
  "request_invalid",
]);
const utf8 = new TextEncoder();
function bytes(value: string): number {
  return utf8.encode(value).length;
}
function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function fields(
  value: unknown,
  required: string[],
  optional: string[] = [],
): value is Record<string, unknown> {
  return (
    object(value) &&
    required.every((key) => Object.hasOwn(value, key)) &&
    Object.keys(value).every(
      (key) => required.includes(key) || optional.includes(key),
    )
  );
}
function base64(value: unknown, maximum: number): value is string {
  if (
    typeof value !== "string" ||
    value.length > 4 * Math.ceil(maximum / 3) ||
    !BASE64.test(value)
  )
    return false;
  try {
    const decoded = atob(value);
    return decoded.length <= maximum && btoa(decoded) === value;
  } catch {
    return false;
  }
}
function headers(value: unknown): value is HTTPHeader[] {
  if (!Array.isArray(value)) return false;
  let total = 0;
  for (const row of value) {
    if (
      !fields(row, ["name", "value"]) ||
      typeof row.name !== "string" ||
      !TOKEN.test(row.name) ||
      typeof row.value !== "string" ||
      /[\r\n\0]/.test(row.value)
    )
      return false;
    total += bytes(row.name) + bytes(row.value);
    if (total > HTTP_CAPTURE_LIMITS.headerBytes) return false;
  }
  return true;
}
function message(
  value: unknown,
  source: string,
  maximum: number,
): value is HTTPMessage {
  return source === "http"
    ? fields(value, ["kind", "headers", "body_base64"]) &&
        value.kind === "structured" &&
        headers(value.headers) &&
        base64(value.body_base64, maximum)
    : fields(value, ["kind", "data_base64"]) &&
        value.kind === "raw" &&
        base64(value.data_base64, maximum);
}
function webRequest(value: unknown, source: string): boolean {
  if (
    !fields(value, ["method", "url", "message"]) ||
    typeof value.method !== "string" ||
    value.method.length > 64 ||
    !TOKEN.test(value.method) ||
    typeof value.url !== "string" ||
    bytes(value.url) > 8192 ||
    /[\s\p{Cc}\\]/u.test(value.url) ||
    /%(?![A-Fa-f0-9]{2})/.test(value.url)
  )
    return false;
  try {
    const url = new URL(value.url);
    const authority = /^https?:\/\/([^/?#]*)/i.exec(value.url)?.[1];
    if (
      !["http:", "https:"].includes(url.protocol) ||
      !url.hostname ||
      authority === undefined ||
      authority.includes("@") ||
      authority.endsWith(":") ||
      url.port === "0"
    )
      return false;
  } catch {
    return false;
  }
  return (
    message(value.message, source, HTTP_CAPTURE_LIMITS.requestBodyBytes) &&
    (value.message.kind !== "raw" || value.message.data_base64 !== "")
  );
}
function attempt(value: unknown, source: string): value is HTTPAttempt {
  if (
    !fields(value, ["request"], ["response", "error"]) ||
    !webRequest(value.request, source)
  )
    return false;
  if (
    Object.hasOwn(value, "error") &&
    (typeof value.error !== "string" || !OUTCOMES.has(value.error))
  )
    return false;
  if (!Object.hasOwn(value, "response")) return typeof value.error === "string";
  const response = value.response;
  return (
    fields(response, ["status", "message", "complete"]) &&
    typeof response.status === "number" &&
    Number.isInteger(response.status) &&
    response.status >= 100 &&
    response.status <= 599 &&
    typeof response.complete === "boolean" &&
    !(response.complete && value.error !== undefined) &&
    message(response.message, source, HTTP_CAPTURE_LIMITS.responseBodyBytes)
  );
}

/** Strict current capture contract; shared Go/Python/UI cases pin its semantics. */
export function isHTTPCapture(value: unknown): value is HTTPCapture {
  if (
    !fields(value, ["source", "request_id", "attempts"], ["request_tag"]) ||
    typeof value.source !== "string" ||
    !["http", "caido"].includes(value.source) ||
    typeof value.request_id !== "string" ||
    !value.request_id ||
    bytes(value.request_id) > 256 ||
    Array.from(value.request_id).some((char) => char.charCodeAt(0) < 32)
  )
    return false;
  if (
    Object.hasOwn(value, "request_tag") &&
    (typeof value.request_tag !== "string" ||
      !IDENTIFIER.test(value.request_tag))
  )
    return false;
  if (
    value.source === "http" &&
    (!/^[1-9][0-9]*$/.test(value.request_id) ||
      !Number.isSafeInteger(Number(value.request_id)) ||
      value.request_tag === undefined)
  )
    return false;
  return (
    Array.isArray(value.attempts) &&
    value.attempts.length > 0 &&
    value.attempts.length <= HTTP_CAPTURE_LIMITS.attempts &&
    value.attempts.every((item) => attempt(item, String(value.source)))
  );
}

export function parseHTTPExchange(
  source: string,
): HTTPExchangeDocument | undefined {
  if (bytes(source) > MAXIMUM_PREVIEW_BYTES) return undefined;
  try {
    const value: unknown = JSON.parse(source);
    // JSON.parse supplies strict JSON syntax. The existing YAML parser checks
    // repeated keys, which JSON.parse alone would silently overwrite.
    if (
      parseDocument(source, { uniqueKeys: true, prettyErrors: false }).errors
        .length
    )
      return undefined;
    if (
      !fields(value, ["schema", "capture", "finding"]) ||
      value.schema !== HTTP_EXCHANGE_SCHEMA ||
      !isHTTPCapture(value.capture) ||
      !fields(value.finding, ["proposal_id", "receipt_id"]) ||
      typeof value.finding.proposal_id !== "string" ||
      !IDENTIFIER.test(value.finding.proposal_id) ||
      typeof value.finding.receipt_id !== "string" ||
      !IDENTIFIER.test(value.finding.receipt_id)
    )
      return undefined;
    return value as unknown as HTTPExchangeDocument;
  } catch {
    return undefined;
  }
}

export function messageBytes(message: HTTPMessage): Uint8Array<ArrayBuffer> {
  const value = atob(
    message.kind === "raw" ? message.data_base64 : message.body_base64,
  );
  return Uint8Array.from(value, (char) => char.charCodeAt(0));
}
