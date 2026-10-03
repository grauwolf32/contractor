/** The part of an infinite query result that LoadMoreButton reads. */
export interface NextPageQuery {
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  fetchNextPage: () => Promise<unknown>;
}

/** Fetches an infinite query's next page; renders nothing on the last page. */
export function LoadMoreButton({
  query,
  label,
  pendingLabel = "Loading…",
}: {
  query: NextPageQuery;
  label: string;
  pendingLabel?: string;
}) {
  if (!query.hasNextPage) return null;
  return (
    <button
      className="secondary-button"
      type="button"
      disabled={query.isFetchingNextPage}
      onClick={() => void query.fetchNextPage()}
    >
      {query.isFetchingNextPage ? pendingLabel : label}
    </button>
  );
}
