import { useInfiniteQuery } from "@tanstack/react-query";

import {
  auditPollInterval,
  listAuditEvents,
  type Audit,
  type AuditEventPage,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice } from "../../../app/error-notice";
import { checkStateLabel } from "../../../app/vocabulary";
import { ActivityLog } from "../../../ui";
import { type CheckLinks } from "./check-links";
import { eventEntry } from "./event-model";
import { useAuditProjectionRefresh } from "./projection-refresh";

export function CheckEventHistory({
  audit,
  now,
  links,
}: {
  audit: Audit;
  now: string | undefined;
  links: CheckLinks;
}) {
  const api = usePublicAPI();
  const queryKey = queryKeys.audits.events(audit.auditId);
  type Position =
    | {
        cursor: string;
        through: number;
        before: number;
        total: number;
        seen: string[];
      }
    | undefined;
  const query = useInfiniteQuery({
    queryKey,
    initialPageParam: undefined as Position,
    queryFn: async ({ pageParam, signal }) => {
      const page = await listAuditEvents(
        api,
        audit.auditId,
        pageParam === undefined ? {} : { cursor: pageParam.cursor },
        signal,
      );
      if (
        pageParam !== undefined &&
        (page.throughSequence !== pageParam.through ||
          page.total !== pageParam.total ||
          page.items.some((event) => event.sequence >= pageParam.before) ||
          (page.page.nextCursor !== undefined &&
            pageParam.seen.includes(page.page.nextCursor)))
      ) {
        throw new Error("The server returned inconsistent activity pages.");
      }
      return page;
    },
    getNextPageParam: (
      last: AuditEventPage,
      _pages,
      previous: Position,
    ): Position => {
      if (!last.page.hasMore) return undefined;
      const cursor = last.page.nextCursor!;
      return {
        cursor,
        through: last.throughSequence,
        before: last.items.at(-1)!.sequence,
        total: last.total,
        seen: [...(previous?.seen ?? []), cursor],
      };
    },
    refetchInterval: auditPollInterval([audit], 5_000),
    refetchOnReconnect: true,
    refetchOnWindowFocus: false,
    retry: false,
  });
  useAuditProjectionRefresh(audit, queryKey);
  const pages = query.data?.pages ?? [];
  const events = pages.flatMap((page) => page.items);
  const entries = events.map((event) => eventEntry(event, links));
  if (now !== undefined)
    entries.unshift({
      id: "now",
      time: "now",
      tone: checkStateLabel(audit.state).tone,
      title: now,
    });
  return (
    <>
      {query.error === null ? null : (
        <ErrorNotice
          error={query.error}
          context={
            query.isFetchNextPageError
              ? "Older activity could not be loaded."
              : "Activity could not be refreshed."
          }
          onRetry={() => {
            void (query.isFetchNextPageError
              ? query.fetchNextPage()
              : query.refetch());
          }}
          retryPending={query.isFetching}
        />
      )}
      {query.isPending ? (
        <p className="checks-quiet" role="status">
          Loading activity…
        </p>
      ) : null}
      <ActivityLog aria-label="Activity on this check" entries={entries} />
      {query.isSuccess && events.length === 0 ? (
        <p className="checks-quiet">No recorded events.</p>
      ) : null}
      {pages.length === 0 ? null : (
        <p className="checks-quiet">
          Showing {events.length.toLocaleString("en-US")} of{" "}
          {pages[0]!.total.toLocaleString("en-US")} recorded events.
        </p>
      )}
      <button
        type="button"
        className="secondary-button"
        disabled={query.isFetching}
        onClick={() => {
          void query.refetch();
        }}
      >
        Refresh activity
      </button>{" "}
      {query.hasNextPage ? (
        <button
          type="button"
          className="secondary-button"
          disabled={query.isFetching}
          onClick={() => {
            void query.fetchNextPage();
          }}
        >
          {query.isFetchingNextPage
            ? "Loading older activity…"
            : "Load older activity"}
        </button>
      ) : null}
    </>
  );
}
