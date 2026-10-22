import {
  useInfiniteQuery,
  useQueryClient,
  type QueryKey,
} from "@tanstack/react-query";
import type { ReactNode } from "react";

import { invalidAPIResponse, PublicAPIError } from "../../api/error";
import { ErrorNotice } from "../../app/error-notice";
import { QueryView } from "../../app/query-view";
import { Button } from "../../ui";

interface Page {
  page: { hasMore: boolean; nextCursor?: string };
  fence?: unknown;
}
type Position = { cursor: string; fence: string | undefined } | undefined;

/** User-driven pages: no background replay of an unbounded loaded chain. */
export function CorrelationPages<T extends Page>({
  queryKey,
  load,
  children,
  moreLabel = "Load more",
  onRestart,
  refreshable = true,
}: {
  queryKey: QueryKey;
  load: (cursor: string | undefined, signal: AbortSignal) => Promise<T>;
  children: (pages: T[]) => ReactNode;
  moreLabel?: string;
  onRestart?: () => void;
  refreshable?: boolean;
}) {
  const client = useQueryClient();
  const query = useInfiniteQuery({
    queryKey,
    initialPageParam: undefined as Position,
    queryFn: async ({ pageParam, signal }) => {
      const value = await load(pageParam?.cursor, signal);
      if (
        pageParam !== undefined &&
        pageParam.fence !== JSON.stringify(value.fence)
      )
        throw new PublicAPIError({
          status: 409,
          code: "finding_correlation_stale",
          message:
            "The related observations changed. Restart to read their current results.",
        });
      if (
        value.page.hasMore &&
        (!value.page.nextCursor || value.page.nextCursor === pageParam?.cursor)
      )
        throw invalidAPIResponse(
          200,
          "Correlation continuation did not advance",
        );
      return value;
    },
    getNextPageParam: (last): Position =>
      last.page.hasMore
        ? { cursor: last.page.nextCursor!, fence: JSON.stringify(last.fence) }
        : undefined,
    staleTime: Infinity,
    gcTime: 0,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: (count, error) =>
      count < 2 &&
      error instanceof PublicAPIError &&
      error.retryable &&
      (error.retryAfterMs ?? 0) <= 2_147_483_647,
    retryDelay: (attempt, error) =>
      Math.max(
        1000 * 2 ** attempt,
        error instanceof PublicAPIError ? (error.retryAfterMs ?? 0) : 0,
      ),
  });
  const restart = () => {
    onRestart?.();
    void client.resetQueries({ queryKey, exact: true });
  };
  const permanent =
    query.error instanceof PublicAPIError && !query.error.retryable;
  // A stale/invalid cursor is never repeated, and its old rows cannot look current.
  if (permanent)
    return (
      <ErrorNotice
        error={query.error}
        context="These related observations are no longer available at the displayed revision."
        onRetry={restart}
        retryLabel="Restart from current results"
      />
    );
  return (
    <QueryView
      query={query}
      errorContext="Related observations could not be loaded."
    >
      {(data) => (
        <>
          {query.isFetching && !query.isFetchingNextPage ? (
            <p className="ui-caption" role="status">
              Refreshing related observations…
            </p>
          ) : null}
          {children(data.pages)}
          <div className="issues-duplicates-actions">
            {query.hasNextPage ? (
              <Button
                size="sm"
                pending={query.isFetching}
                pendingLabel="Loading…"
                onClick={() => void query.fetchNextPage()}
              >
                {moreLabel}
              </Button>
            ) : null}
            {refreshable ? (
              <Button size="sm" disabled={query.isFetching} onClick={restart}>
                Refresh related observations
              </Button>
            ) : null}
          </div>
        </>
      )}
    </QueryView>
  );
}
