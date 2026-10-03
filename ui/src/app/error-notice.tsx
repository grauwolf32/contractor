import { PublicAPIError } from "../api/error";

export function ErrorNotice({
  error,
  reconcileWrite = false,
  context,
  onRetry,
  retryLabel = "Try again",
  retryPending = false,
}: {
  error: unknown;
  reconcileWrite?: boolean;
  context?: string;
  onRetry?: () => void;
  retryLabel?: string;
  retryPending?: boolean;
}) {
  const message = error instanceof Error ? error.message : "Request failed";
  const requestId =
    error instanceof PublicAPIError ? error.requestId : undefined;
  return (
    <div className="notice notice-error" role="alert">
      <strong>{context ?? message}</strong>
      {context === undefined ? null : <p>{message}</p>}
      {reconcileWrite &&
      error instanceof PublicAPIError &&
      error.code === "conflict" ? (
        <p>
          The record or binding changed. Refresh its current revision before
          choosing an explicit new update; this change was not retried.
        </p>
      ) : reconcileWrite &&
        error instanceof PublicAPIError &&
        error.status === 0 ? (
        <p>
          The Server response was not received, so the change may have been
          applied. Refresh the current state before deciding whether to submit
          it again.
        </p>
      ) : null}
      {onRetry === undefined || reconcileWrite ? null : (
        <button
          className="secondary-button"
          type="button"
          disabled={retryPending}
          onClick={onRetry}
        >
          {retryPending ? "Loading…" : retryLabel}
        </button>
      )}
      {error instanceof PublicAPIError ? (
        <details className="error-details">
          <summary>Request details</summary>
          <small>
            Code {error.code} · Status {error.status}
          </small>
          {requestId === undefined ? null : <small>Request {requestId}</small>}
        </details>
      ) : null}
    </div>
  );
}
