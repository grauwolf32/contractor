import { useInfiniteQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import {
  auditPollInterval,
  listAuditEvents,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { nextPageCursor } from "../../../app/pagination";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { Button, TechnicalDetails, StatusGlyph } from "../../../ui";
import { readableValue } from "../../../app/vocabulary";
import { RecordedTime } from "../../../app/recorded-time";
import { useAuditProjectionRefresh } from "./projection-refresh";

/** Only the user loads older pages; refreshes reread the already loaded chain. */
export function AuditActivity({
  audit,
  expanded = false,
}: {
  audit: Audit;
  expanded?: boolean;
}) {
  const api = usePublicAPI();
  const [open, setOpen] = useState(expanded);
  const queryKey = queryKeys.audits.events(audit.auditId);
  const changing = auditPollInterval([audit]) !== false;
  const query = useInfiniteQuery({
    queryKey,
    enabled: open,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) =>
      listAuditEvents(
        api,
        audit.auditId,
        pageParam === undefined ? {} : { cursor: pageParam },
        signal,
      ),
    getNextPageParam: (page) => nextPageCursor(page.page),
    refetchInterval: changing ? pollInterval({ interval: 5_000 }) : false,
    refetchOnReconnect: true,
    retry: false,
  });
  useAuditProjectionRefresh(audit, queryKey, open);
  return (
    <section className="ui-stack" aria-label="Check activity">
      {expanded ? (
        <header className="research-section-head">
          <div>
            <p className="eyebrow">Recorded events</p>
            <h2>Activity</h2>
          </div>
        </header>
      ) : (
        <Button size="sm" onClick={() => setOpen(!open)}>
          {open ? "Hide activity" : "Show activity"}
        </Button>
      )}
      {open ? (
        <QueryView
          query={query}
          loading={<p role="status">Loading activity…</p>}
        >
          {(data) => (
            <>
              <ul className="research-timeline">
                {data.pages
                  .flatMap((page) => page.items)
                  .map((event) => (
                    <li key={event.sequence}>
                      <span
                        className="research-timeline-mark"
                        data-tone={
                          event.summary.disposition?.includes("invalid")
                            ? "warning"
                            : event.summary.state === "active"
                              ? "info"
                              : "neutral"
                        }
                      >
                        <StatusGlyph
                          tone={
                            event.summary.disposition?.includes("invalid")
                              ? "warning"
                              : "info"
                          }
                        />
                      </span>
                      <div className="research-timeline-copy">
                        <strong>
                          {readableValue(event.kind.replaceAll(".", " "))}
                        </strong>
                        <p className="ui-caption">
                          <RecordedTime value={event.createdAt} />
                        </p>
                        {event.summary.runId === undefined ? null : (
                          <>
                            {" "}
                            ·{" "}
                            <Link
                              to={`/runs/${encodeURIComponent(event.summary.runId)}`}
                            >
                              Open Run
                            </Link>
                          </>
                        )}
                        <TechnicalDetails description="Recorded event identity and summary.">
                          <pre className="checks-task-json">
                            {JSON.stringify(event, null, 2)}
                          </pre>
                        </TechnicalDetails>
                      </div>
                    </li>
                  ))}
              </ul>
              <Button
                size="sm"
                pending={query.isFetchingNextPage}
                pendingLabel="Loading older activity…"
                disabled={
                  !query.hasNextPage ||
                  query.isFetching ||
                  query.isFetchNextPageError
                }
                onClick={() => void query.fetchNextPage()}
              >
                Load older activity
              </Button>
            </>
          )}
        </QueryView>
      ) : null}
    </section>
  );
}
