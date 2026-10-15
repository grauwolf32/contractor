import { schemaViolation } from "./contract";
import type { components } from "./generated/public";

type ErrorEnvelope = components["schemas"]["Error"];

const MAXIMUM_ERROR_TEXT = 512;

export type InUseDetails =
  | {
      kind: "credential_in_use";
      runIds: readonly string[];
      auditIds: readonly string[];
      bindingLabels: readonly string[];
    }
  | {
      kind: "runtime_credential_in_use";
      bindingLabels: readonly string[];
      projectIds: readonly string[];
      runIds: readonly string[];
      auditIds: readonly string[];
      allocationIds: readonly string[];
    }
  | {
      kind: "runtime_label_in_use";
      runtimeAgentIds: readonly string[];
    };

/** In-use references, kept only when the details conform to the contract. */
function parseInUseDetails(value: unknown): InUseDetails | undefined {
  const details = value as ErrorEnvelope["details"];
  switch (details?.kind) {
    case "credential_in_use":
      if (schemaViolation("CredentialInUseDetails", details) !== undefined)
        return undefined;
      return {
        kind: details.kind,
        runIds: [...new Set(details.runIds)],
        auditIds: [...new Set(details.auditIds)],
        bindingLabels: [...new Set(details.bindingLabels)],
      };
    case "runtime_credential_in_use":
      if (
        schemaViolation("RuntimeCredentialInUseDetails", details) !== undefined
      ) {
        return undefined;
      }
      return {
        kind: details.kind,
        bindingLabels: [...new Set(details.bindingLabels)],
        projectIds: [...new Set(details.projectIds)],
        runIds: [...new Set(details.runIds)],
        auditIds: [...new Set(details.auditIds)],
        allocationIds: [...new Set(details.allocationIds)],
      };
    case "runtime_label_in_use":
      if (schemaViolation("RuntimeLabelInUseDetails", details) !== undefined)
        return undefined;
      return {
        kind: details.kind,
        runtimeAgentIds: [...new Set(details.runtimeAgentIds)],
      };
    default:
      return undefined;
  }
}

function boundedText(value: unknown, fallback: string): string {
  if (typeof value !== "string" || value.length === 0) {
    return fallback;
  }
  return value.slice(0, MAXIMUM_ERROR_TEXT);
}

export class PublicAPIError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  readonly requestId?: string;
  readonly inUse?: InUseDetails;
  readonly status: number;

  constructor(options: {
    status: number;
    code: string;
    message: string;
    retryable?: boolean;
    requestId?: string;
    inUse?: InUseDetails;
  }) {
    super(boundedText(options.message, "Public API request failed"));
    this.name = "PublicAPIError";
    this.status = options.status;
    this.code = boundedText(options.code, "request_failed");
    this.retryable = options.retryable ?? false;
    if (options.requestId !== undefined) {
      this.requestId = boundedText(options.requestId, "unknown");
    }
    if (options.inUse !== undefined) this.inUse = options.inUse;
  }
}

export class APICompatibilityError extends PublicAPIError {
  readonly reportedVersion?: string;

  constructor(reportedVersion?: string) {
    super({
      status: 0,
      code: "incompatible_api",
      message:
        reportedVersion === undefined
          ? "Server did not report a supported public API version"
          : `Server public API version is not supported: ${boundedText(reportedVersion, "unknown")}`,
    });
    this.name = "APICompatibilityError";
    if (reportedVersion !== undefined) {
      this.reportedVersion = boundedText(reportedVersion, "unknown");
    }
  }
}

export function publicAPIError(status: number, value: unknown): PublicAPIError {
  const envelope = value as Partial<ErrorEnvelope> | null;
  const details = envelope?.details;
  const inUse = parseInUseDetails(details);
  return new PublicAPIError({
    status,
    code: boundedText(envelope?.code, "request_failed"),
    message: boundedText(envelope?.message, "Public API request failed"),
    retryable: envelope?.retryable === true,
    ...(typeof envelope?.requestId === "string"
      ? { requestId: envelope.requestId }
      : {}),
    ...(inUse === undefined ? {} : { inUse }),
  });
}

/** Returns a response's data, or throws the error envelope it carried. */
export function requireData<T>(result: {
  data?: T;
  error?: unknown;
  response: Response;
}): T {
  if (result.data === undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
  return result.data;
}

/**
 * The one client error for a Server response the UI cannot accept: a body
 * that breaks the operation's generated response schema, an unreadable body,
 * or a cross-field invariant the schema cannot express. Its message names the
 * operation and the failing location, never the body. A status of 0 means no
 * readable response arrived.
 */
export class InvalidResponseError extends PublicAPIError {
  constructor(status: number, message: string) {
    super({ status, code: "invalid_api_response", message });
    this.name = "InvalidResponseError";
  }
}

/** An error for a response body that does not match the public contract. */
export function invalidAPIResponse(
  status: number,
  message: string,
): InvalidResponseError {
  return new InvalidResponseError(status, message);
}
