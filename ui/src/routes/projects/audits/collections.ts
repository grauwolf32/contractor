import {
  useInfiniteQuery,
  type InfiniteData,
  type QueryKey,
} from "@tanstack/react-query";

import {
  mergeAuditCollections,
  type AuditCollection,
} from "../../../api/audit-collections";

/** A bounded Audit collection with its continuation state. */
export interface AuditCollectionQuery<T> {
  items: T[];
  /** More records exist beyond the loaded batches. */
  truncated: boolean;
  /** Read the next batch from the last cursor (or retry a failed batch). */
  loadMore: () => void;
  isLoadingMore: boolean;
  /** A continuation batch failed; the loaded batches stay visible. */
  moreError: Error | null;
  isPending: boolean;
  isSuccess: boolean;
  isFetching: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => Promise<unknown>;
}

type ItemOf<B> = B extends AuditCollection<infer T> ? T : never;

function selectBatches<T>(data: InfiniteData<AuditCollection<T>>) {
  return mergeAuditCollections(data.pages);
}

/**
 * One Audit collection read in batches of at most the page cap. Refetches
 * (polling, invalidation, manual refresh) re-read every loaded batch so the
 * view never mixes a fresh head with a stale tail. `loadBatch` receives the
 * previous batch so it can carry consistency data (such as a revision) across
 * continuations.
 */
export function useAuditCollection<B extends AuditCollection<unknown>>({
  queryKey,
  loadBatch,
  enabled = true,
  refetchInterval,
}: {
  queryKey: QueryKey;
  loadBatch: (
    cursor: string | undefined,
    previous: B | undefined,
  ) => Promise<B>;
  enabled?: boolean;
  refetchInterval: number | false | ((items: ItemOf<B>[]) => number | false);
}): AuditCollectionQuery<ItemOf<B>> {
  const query = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam }) => loadBatch(pageParam.cursor, pageParam.previous),
    initialPageParam: { cursor: undefined, previous: undefined } as {
      cursor: string | undefined;
      previous: B | undefined;
    },
    getNextPageParam: (last: B) =>
      last.truncated && last.nextCursor !== undefined
        ? { cursor: last.nextCursor, previous: last }
        : undefined,
    enabled,
    refetchInterval:
      typeof refetchInterval === "function"
        ? (current) =>
            refetchInterval(
              (current.state.data?.pages.flatMap((page) => page.items) ??
                []) as ItemOf<B>[],
            )
        : refetchInterval,
    refetchOnReconnect: true,
    select: selectBatches as (
      data: InfiniteData<B>,
    ) => AuditCollection<ItemOf<B>>,
  });
  const loadingMore = query.isFetchingNextPage;
  return {
    items: query.data?.items ?? [],
    truncated: query.hasNextPage && !loadingMore,
    loadMore: () => {
      if (query.hasNextPage && !loadingMore) void query.fetchNextPage();
    },
    isLoadingMore: loadingMore,
    moreError: query.isFetchNextPageError ? query.error : null,
    isPending: query.isPending,
    isSuccess: query.isSuccess,
    isFetching: query.isFetching,
    isError: query.isError && !query.isFetchNextPageError,
    error: query.isFetchNextPageError ? null : query.error,
    refetch: () => query.refetch(),
  };
}
