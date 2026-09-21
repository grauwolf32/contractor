import {
  useInfiniteQuery,
  useQueries,
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
 * per source. The first batch of every source polls according to
 * `refetchInterval`; continuation batches are read once from their cursor and
 * refreshed with the manual refresh. `identity` drops a record that a moved
 * page boundary repeats across batches.
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
  const [continuations, setContinuations] = useState<Record<string, string[]>>(
    {},
  );
  const first = useQueries({
    queries: sources.map((source) => ({
      queryKey: queryKey(source),
      queryFn: () => collectAuditPages(load(source)),
      enabled,
      refetchInterval: refetchInterval(source),
      refetchOnReconnect: true,
    })),
  });
  const continuationEntries = sources.flatMap((source, index) =>
    (continuations[id(source)] ?? []).map((cursor) => ({
      index,
      cursor,
      source,
    })),
  );
  const later = useQueries({
    queries: continuationEntries.map(({ source, cursor }) => ({
      queryKey: [...queryKey(source), "from", cursor],
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
        else if (batch.isError) {
          moreError = batch.error;
          break;
        } else {
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
      setContinuations((current) =>
        (current[key] ?? []).includes(cursor)
          ? current
          : { ...current, [key]: [...(current[key] ?? []), cursor] },
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
      refetch: () =>
        Promise.all([head.refetch(), ...tail.map((batch) => batch.refetch())]),
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
