import type { ReactNode } from "react";

import { ErrorNotice } from "../routes/artifacts/common";

/** The subset of a TanStack Query result that QueryView reads. */
export interface QueryViewState<T> {
  data: T | undefined;
  error: Error | null;
  isFetching: boolean;
}

/**
 * Renders a query's loading, error, and loaded states. The error panel
 * replaces content only when no data has been loaded; a failed background
 * refetch keeps the last loaded data visible with an inline stale-data
 * warning and retry.
 */
export function QueryView<T>({
  query,
  loading,
  errorContext,
  onRetry,
  children,
}: {
  query: QueryViewState<T>;
  loading: ReactNode;
  errorContext?: string;
  onRetry: () => void;
  children: (data: T) => ReactNode;
}) {
  if (query.data === undefined) {
    if (query.error === null) return loading;
    return (
      <ErrorNotice
        error={query.error}
        {...(errorContext === undefined ? {} : { context: errorContext })}
        onRetry={onRetry}
        retryPending={query.isFetching}
      />
    );
  }
  return (
    <>
      {query.error === null ? null : (
        <StaleDataWarning
          error={query.error}
          onRetry={onRetry}
          retryPending={query.isFetching}
        />
      )}
      {children(query.data)}
    </>
  );
}

export function StaleDataWarning({
  error,
  onRetry,
  retryPending = false,
}: {
  error: Error;
  onRetry: () => void;
  retryPending?: boolean;
}) {
  return (
    <div className="notice notice-warning" role="status">
      <strong>Could not refresh; showing the last loaded data.</strong>
      <p>{error.message}</p>
      <button
        className="secondary-button"
        type="button"
        disabled={retryPending}
        onClick={onRetry}
      >
        {retryPending ? "Loading…" : "Try again"}
      </button>
    </div>
  );
}
