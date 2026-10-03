import type { components } from "./generated/public";

type ErrorEnvelope = components["schemas"]["Error"];

const MAXIMUM_ERROR_TEXT = 512;
const SAFE_RESOURCE_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
const SAFE_RUNTIME_LABEL = /^[a-z][a-z0-9_-]{0,62}$/;
const SAFE_RUNTIME_AGENT_ID = /^[0-9a-f]{64}$/;

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

function safeReferences(value: unknown, pattern: RegExp): string[] {
  if (!Array.isArray(value) || value.length > 128) return [];
  return [
    ...new Set(
      value.filter(
        (item): item is string =>
          typeof item === "string" && pattern.test(item),
      ),
    ),
  ];
}

function parseInUseDetails(
  value: ErrorEnvelope["details"],
): InUseDetails | undefined {
  switch (value?.kind) {
    case "credential_in_use":
      return {
        kind: value.kind,
        runIds: safeReferences(value.runIds, SAFE_RESOURCE_ID),
        auditIds: safeReferences(value.auditIds, SAFE_RESOURCE_ID),
        bindingLabels: safeReferences(value.bindingLabels, SAFE_RUNTIME_LABEL),
      };
    case "runtime_credential_in_use":
      return {
        kind: value.kind,
        bindingLabels: safeReferences(value.bindingLabels, SAFE_RUNTIME_LABEL),
        projectIds: safeReferences(value.projectIds, SAFE_RESOURCE_ID),
        runIds: safeReferences(value.runIds, SAFE_RESOURCE_ID),
        auditIds: safeReferences(value.auditIds, SAFE_RESOURCE_ID),
        allocationIds: safeReferences(value.allocationIds, SAFE_RESOURCE_ID),
      };
    case "runtime_label_in_use":
      return {
        kind: value.kind,
        runtimeAgentIds: safeReferences(
          value.runtimeAgentIds,
          SAFE_RUNTIME_AGENT_ID,
        ),
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

/** An error for a response body that does not match the public contract. */
export function invalidAPIResponse(
  status: number,
  message: string,
): PublicAPIError {
  return new PublicAPIError({ status, code: "invalid_api_response", message });
}
