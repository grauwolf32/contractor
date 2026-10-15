import type { ReactNode } from "react";

import { Button } from "../ui/button";
import { ErrorNotice } from "./error-notice";

/** The subset of a TanStack Query result that QueryView reads. */
interface QueryViewState<T> {
  data: T | undefined;
  error: Error | null;
  isFetching: boolean;
  /** Retries the read; a TanStack Query result brings its own. */
  refetch?: () => unknown;
}

/**
 * Renders a query's loading, error, empty, and loaded states. The error panel
 * replaces content only when no data has been loaded; a failed background
 * refetch keeps the last loaded data visible with an inline stale-data
 * warning and retry. Pass the useQuery result itself as `query`: Try again
 * then calls its refetch unless `onRetry` says otherwise, and the loading
 * state reads "Loading…" unless `loading` replaces it.
 */
export function QueryView<T>({
  query,
  loading,
  errorContext,
  onRetry,
  isEmpty,
  empty,
  children,
}: {
  query: QueryViewState<T>;
  loading?: ReactNode;
  errorContext?: string;
  onRetry?: () => void;
  /** Selects loaded data that renders `empty` instead of `children`. */
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  const retry = onRetry ?? (() => void query.refetch?.());
  if (query.data === undefined) {
    if (query.error === null)
      return (
        loading ?? (
          <p className="ui-caption" role="status">
            Loading…
          </p>
        )
      );
    return (
      <ErrorNotice
        error={query.error}
        {...(errorContext === undefined ? {} : { context: errorContext })}
        onRetry={retry}
        retryPending={query.isFetching}
      />
    );
  }
  return (
    <>
      {query.error === null ? null : (
        <StaleDataWarning
          error={query.error}
          onRetry={retry}
          retryPending={query.isFetching}
        />
      )}
      {isEmpty?.(query.data) === true ? empty : children(query.data)}
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
      <Button
        size="sm"
        pending={retryPending}
        pendingLabel="Loading…"
        onClick={onRetry}
      >
        Try again
      </Button>
    </div>
  );
}
