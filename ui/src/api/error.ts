import type { components } from "./generated/public";

type ErrorEnvelope = components["schemas"]["Error"];

const MAXIMUM_ERROR_TEXT = 512;
const SAFE_RESOURCE_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;

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
  readonly referencedRunIds?: readonly string[];
  readonly status: number;

  constructor(options: {
    status: number;
    code: string;
    message: string;
    retryable?: boolean;
    requestId?: string;
    referencedRunIds?: readonly string[];
  }) {
    super(boundedText(options.message, "Public API request failed"));
    this.name = "PublicAPIError";
    this.status = options.status;
    this.code = boundedText(options.code, "request_failed");
    this.retryable = options.retryable ?? false;
    if (options.requestId !== undefined) {
      this.requestId = boundedText(options.requestId, "unknown");
    }
    if (options.referencedRunIds !== undefined) {
      this.referencedRunIds = [...options.referencedRunIds];
    }
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
  const referencedRunIds =
    details?.kind === "credential_in_use" &&
    Array.isArray(details.runIds) &&
    details.runIds.length <= 128 &&
    details.runIds.every(
      (runId) => typeof runId === "string" && SAFE_RESOURCE_ID.test(runId),
    )
      ? details.runIds
      : undefined;
  return new PublicAPIError({
    status,
    code: boundedText(envelope?.code, "request_failed"),
    message: boundedText(envelope?.message, "Public API request failed"),
    retryable: envelope?.retryable === true,
    ...(typeof envelope?.requestId === "string"
      ? { requestId: envelope.requestId }
      : {}),
    ...(referencedRunIds === undefined ? {} : { referencedRunIds }),
  });
}
