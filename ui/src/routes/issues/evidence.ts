/**
 * Reading the evidence of a possible issue for display: which captured
 * header values look like credentials (masked until the user asks), what a
 * request body holds, and how much evidence there is.
 */
import type { FindingProposalReceipt } from "../../api/audits";
import type { components } from "../../api/generated/public";

type HTTPAttempt = components["schemas"]["FindingHTTPAttempt"];

/** Header names whose values are credentials. */
const CREDENTIAL_HEADERS: ReadonlySet<string> = new Set([
  "authorization",
  "proxy-authorization",
  "cookie",
  "set-cookie",
  "x-api-key",
]);

/** Parts of a header name that mark its value as a credential. */
const CREDENTIAL_PARTS: readonly string[] = [
  "token",
  "secret",
  "password",
  "api-key",
  "apikey",
  "session",
];

/**
 * True for headers whose values look like credentials: Authorization,
 * Proxy-Authorization, Cookie, Set-Cookie, X-API-Key and any name with
 * "token", "secret", "password", "api-key", "apikey" or "session" in it.
 * Their values stay hidden until the user shows them one by one.
 */
export function isCredentialHeader(name: string): boolean {
  const key = name.trim().toLowerCase();
  return (
    CREDENTIAL_HEADERS.has(key) ||
    CREDENTIAL_PARTS.some((part) => key.includes(part))
  );
}

type DecodedBody =
  | { kind: "empty" }
  | { kind: "text"; text: string; bytes: number }
  | { kind: "binary"; bytes: number };

/** Control characters other than tab, line feed and carriage return. */
function hasControlCharacters(text: string): boolean {
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    if (
      (code < 0x20 && code !== 0x09 && code !== 0x0a && code !== 0x0d) ||
      code === 0x7f
    )
      return true;
  }
  return false;
}

/**
 * A captured request body (canonical base64) as text when it is readable
 * UTF-8, else as a number of bytes. Nothing is guessed beyond the bytes.
 */
export function decodeBody(base64: string): DecodedBody {
  if (base64 === "") return { kind: "empty" };
  let binary: string;
  try {
    binary = atob(base64);
  } catch {
    return { kind: "binary", bytes: Math.floor((base64.length * 3) / 4) };
  }
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return hasControlCharacters(text)
      ? { kind: "binary", bytes: bytes.length }
      : { kind: "text", text, bytes: bytes.length };
  } catch {
    return { kind: "binary", bytes: bytes.length };
  }
}

/** "HTTP 200", "Transport error", "Cancelled" or "No response". */
export function attemptOutcome(
  attempt: Pick<HTTPAttempt, "status" | "error">,
): string {
  if (attempt.status !== undefined) return `HTTP ${attempt.status}`;
  if (attempt.error === "transport_error") return "Transport error";
  if (attempt.error === "cancelled") return "Cancelled";
  return "No response";
}

/**
 * The pieces of evidence the Evidence tab lists: authored locations,
 * captured HTTP attempts and retained evidence files.
 */
export function evidenceCount(proposal: FindingProposalReceipt): number {
  return (
    (proposal.document.locations?.length ?? 0) +
    (proposal.document.http_exchanges?.reduce(
      (count, exchange) => count + exchange.attempts.length,
      0,
    ) ?? 0) +
    (proposal.projectEvidence?.length ?? 0)
  );
}

const INLINE_SNIPPET_BYTES = 64 * 1024;
/** Small saved source text only; structured/HTTP evidence remains explicit. */
export function canShowEvidenceSnippet(
  evidence: FindingProposalReceipt["evidence"][number],
): boolean {
  return (
    evidence.mediaType === "text/plain" &&
    evidence.sizeBytes <= INLINE_SNIPPET_BYTES
  );
}
