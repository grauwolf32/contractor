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
  /** An infinite query whose next page failed: the loaded pages stay. */
  isFetchNextPageError?: boolean;
  /** Reads the next page again; an infinite query result brings its own. */
  fetchNextPage?: () => unknown;
}

/**
 * Renders a query's loading, error, empty, and loaded states. The error panel
 * replaces content only when no data has been loaded; a failed background
 * refetch keeps the last loaded data visible with an inline stale-data
 * warning and retry, and a failed next page of an infinite query keeps the
 * loaded pages with an error under them that reads that page again. Pass the
 * useQuery result itself as `query`: Try again then calls its refetch unless
 * `onRetry` says otherwise, and the loading state reads "Loading…" unless
 * `loading` replaces it.
 */
export function QueryView<T>({
  query,
  loading,
  errorContext,
  errorHint,
  renderError,
  continuationContext = "More items could not be loaded.",
  retryLabel = "Try again",
  onRetry,
  isEmpty,
  empty,
  children,
}: {
  query: QueryViewState<T>;
  /** Replaces "Loading…"; null shows nothing while loading. */
  loading?: ReactNode;
  errorContext?: string;
  /** A line under the error panel, e.g. what the user can do about it. */
  errorHint?: ((error: Error) => ReactNode) | undefined;
  /**
   * Replaces the error panel and the stale-data warning, for an area whose
   * errors need their own words or actions; `retry` is the read's retry.
   */
  renderError?:
    | ((error: Error, retry: () => void, stale: boolean) => ReactNode)
    | undefined;
  /** The heading of a failed next page's error. */
  continuationContext?: string | undefined;
  /** The label of every retry button. */
  retryLabel?: string | undefined;
  onRetry?: () => void;
  /** Selects loaded data that renders `empty` instead of `children`. */
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  const retry = onRetry ?? (() => void query.refetch?.());
  if (query.data === undefined) {
    if (query.error === null)
      return loading === undefined ? (
        <p className="ui-caption" role="status">
          Loading…
        </p>
      ) : (
        loading
      );
    if (renderError !== undefined)
      return renderError(query.error, retry, false);
    return (
      <>
        <ErrorNotice
          error={query.error}
          {...(errorContext === undefined ? {} : { context: errorContext })}
          onRetry={retry}
          retryLabel={retryLabel}
          retryPending={query.isFetching}
        />
        {errorHint === undefined ? null : (
          <p className="ui-caption">{errorHint(query.error)}</p>
        )}
      </>
    );
  }
  const continuation =
    query.error !== null && query.isFetchNextPageError === true;
  const stale = query.error !== null && !continuation;
  return (
    <>
      {!stale || query.error === null ? null : renderError !== undefined ? (
        renderError(query.error, retry, true)
      ) : (
        <StaleDataWarning
          error={query.error}
          onRetry={retry}
          retryLabel={retryLabel}
          retryPending={query.isFetching}
        />
      )}
      {isEmpty?.(query.data) === true ? empty : children(query.data)}
      {!continuation || query.error === null ? null : (
        <ErrorNotice
          error={query.error}
          context={continuationContext}
          onRetry={() => void query.fetchNextPage?.()}
          retryLabel={retryLabel}
          retryPending={query.isFetching}
        />
      )}
    </>
  );
}

export function StaleDataWarning({
  error,
  onRetry,
  retryLabel = "Try again",
  retryPending = false,
}: {
  error: Error;
  onRetry: () => void;
  retryLabel?: string;
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
        {retryLabel}
      </Button>
    </div>
  );
}
