import {
  useInfiniteQuery,
  useQueries,
  useQueryClient,
  type InfiniteData,
  type QueryKey,
} from "@tanstack/react-query";
import { useState } from "react";

import {
  collectAuditPages,
  mergeAuditCollections,
  type AuditCollection,
  type AuditCollectionPage,
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

/** Reads one page of a collection endpoint for `collectAuditPages`. */
export type AuditPageLoader<T> = (
  cursor?: string,
) => Promise<AuditCollectionPage<T>>;

/**
 * Several Audit collections (one per source) loaded side by side, each capped
 * per source. Continuations follow the cursor returned by the preceding
 * batch, so a refreshed head replaces the entire revision-bound cursor chain.
 * The number of requested batches survives refreshes; their old cursors do
 * not. `identity` drops a record that a moved page boundary repeats.
 */
export function useAuditCollections<S, T>(
  sources: readonly S[],
  {
    id,
    queryKey,
    load,
    identity,
    enabled = true,
    refetchInterval,
  }: {
    id: (source: S) => string;
    queryKey: (source: S) => QueryKey;
    load: (source: S) => AuditPageLoader<T>;
    identity: (item: T) => string;
    enabled?: boolean;
    refetchInterval: (source: S) => number | false;
  },
): {
  results: AuditCollectionQuery<T>[];
  /** At least one source has more records than its loaded batches. */
  truncated: boolean;
  isLoadingMore: boolean;
  /** Read the next batch of every truncated source. */
  loadMore: () => void;
} {
  const queryClient = useQueryClient();
  const [continuationCounts, setContinuationCounts] = useState<
    Record<string, number>
  >({});
  const first = useQueries({
    queries: sources.map((source) => ({
      queryKey: queryKey(source),
      queryFn: () => collectAuditPages(load(source)),
      enabled,
      refetchInterval: refetchInterval(source),
      refetchOnReconnect: true,
    })),
  });
  const continuationKey = (source: S, cursor: string) => [
    ...queryKey(source),
    "from",
    cursor,
  ];
  const continuationEntries = sources.flatMap((source, index) => {
    const entries: { index: number; cursor: string; source: S }[] = [];
    let batch = first[index]!.data;
    const count = continuationCounts[id(source)] ?? 0;
    for (let position = 0; position < count; position += 1) {
      if (!batch?.truncated || batch.nextCursor === undefined) break;
      const cursor = batch.nextCursor;
      entries.push({ index, cursor, source });
      // An uncached batch is observed below. Its completion renders the hook
      // again, revealing the next cursor without consulting the old chain.
      batch = queryClient.getQueryData<AuditCollection<T>>(
        continuationKey(source, cursor),
      );
    }
    return entries;
  });
  const later = useQueries({
    queries: continuationEntries.map(({ source, cursor }) => ({
      queryKey: continuationKey(source, cursor),
      queryFn: () => collectAuditPages(load(source), { cursor }),
      enabled,
      refetchInterval: false as const,
      refetchOnReconnect: true,
    })),
  });
  const results = sources.map((source, index): AuditCollectionQuery<T> => {
    const head = first[index]!;
    const tail = continuationEntries.flatMap((entry, position) =>
      entry.index === index ? [later[position]!] : [],
    );
    const batches: AuditCollection<T>[] = [];
    let pending = false;
    let moreError: Error | null = null;
    if (head.data !== undefined) {
      batches.push(head.data);
      for (const batch of tail) {
        if (batch.data !== undefined) batches.push(batch.data);
        if (batch.isError) {
          moreError = batch.error;
          break;
        } else if (batch.data === undefined) {
          pending = true;
          break;
        }
      }
    }
    const merged = mergeAuditCollections(batches);
    const seen = new Set<string>();
    const items = merged.items.filter((item) => {
      const key = identity(item);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    const failed = tail.find((batch) => batch.isError);
    const loadMore = () => {
      if (failed !== undefined) {
        void failed.refetch();
        return;
      }
      const cursor = merged.nextCursor;
      if (pending || !merged.truncated || cursor === undefined) return;
      const key = id(source);
      const count = (continuationCounts[key] ?? 0) + 1;
      setContinuationCounts((current) =>
        current[key] === count ? current : { ...current, [key]: count },
      );
    };
    return {
      items,
      truncated: merged.truncated && !pending && moreError === null,
      loadMore,
      isLoadingMore: pending,
      moreError,
      isPending: head.isPending,
      isSuccess: head.isSuccess,
      isFetching: head.isFetching || tail.some((batch) => batch.isFetching),
      isError: head.isError,
      error: head.error,
      refetch: async () => {
        const refreshed = await head.refetch();
        // A changed cursor installs fresh continuation queries on render.
        // Only refresh existing tails when they still belong to this head.
        if (
          refreshed.isSuccess &&
          refreshed.data.nextCursor === head.data?.nextCursor
        ) {
          await Promise.all(tail.map((batch) => batch.refetch()));
        }
      },
    };
  });
  return {
    results,
    truncated: results.some((result) => result.truncated),
    isLoadingMore: results.some((result) => result.isLoadingMore),
    loadMore: () => {
      for (const result of results)
        if (result.truncated || result.moreError !== null) result.loadMore();
    },
  };
}
