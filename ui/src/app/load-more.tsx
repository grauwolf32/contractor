import type { ComponentProps } from "react";

import { Button } from "../ui/button";

/** The part of an infinite query result that LoadMoreButton reads. */
interface NextPageQuery {
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  fetchNextPage: () => Promise<unknown>;
}

/** Fetches an infinite query's next page; renders nothing on the last page. */
export function LoadMoreButton({
  query,
  label,
  pendingLabel = "Loading…",
  size,
}: {
  query: NextPageQuery;
  label: string;
  pendingLabel?: string;
  size?: ComponentProps<typeof Button>["size"];
}) {
  if (!query.hasNextPage) return null;
  return (
    <Button
      size={size}
      pending={query.isFetchingNextPage}
      pendingLabel={pendingLabel}
      onClick={() => void query.fetchNextPage()}
    >
      {label}
    </Button>
  );
}
