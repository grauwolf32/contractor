import { PublicAPIError } from "../../api/error";

/**
 * The "Request details" disclosure of a refused or failed request, as
 * ErrorNotice shows it: code, status and request ID. The Server's own message
 * is added when the explanation on screen does not already carry it, so
 * replacing it with a plain sentence loses nothing. Errors that did not come
 * from the Public API have no details.
 */
export function RequestDetails({
  error,
  explanation,
  className,
}: {
  error: unknown;
  /** The sentence shown for this error. */
  explanation?: string | undefined;
  /** Extra classes, e.g. to line up with the decision bar's insets. */
  className?: string | undefined;
}) {
  if (!(error instanceof PublicAPIError)) return null;
  return (
    <details
      className={
        className === undefined
          ? "decisions-error-details"
          : `decisions-error-details ${className}`
      }
    >
      <summary>Request details</summary>
      <small>
        Code {error.code} · Status {error.status}
      </small>
      {error.requestId === undefined ? null : (
        <small>Request {error.requestId}</small>
      )}
      {explanation?.includes(error.message) ? null : (
        <small>Message: {error.message}</small>
      )}
    </details>
  );
}
