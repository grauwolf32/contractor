import { useInfiniteQuery } from "@tanstack/react-query";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { PublicAPIError } from "../../api/error";
import { getRuntimeDiagnostics } from "../../api/run-diagnostics";
import type { StageAttempt } from "../../api/runs";
import { ErrorNotice } from "../../app/error-notice";
import { Button, Facts } from "../../ui";

const DROP_LABELS = {
  queueOverflow: "Queue overflow",
  encodingFailed: "Encoding failed",
  nonRetryable: "Permanent delivery failure",
  retryExhausted: "Retry attempts exhausted",
  collectorRejected: "Rejected by collector",
  deadlineExceeded: "Deadline exceeded",
  cancelled: "Cancelled",
  shutdown: "Discarded at shutdown",
} as const;

/** On-demand retained facts; their availability never claims a Run outcome. */
export function RuntimeDiagnostics({
  runId,
  attempt,
  open,
}: {
  runId: string;
  attempt: Pick<StageAttempt, "stageExecutionId" | "state">;
  open: boolean;
}) {
  const api = usePublicAPI();
  const query = useInfiniteQuery({
    queryKey: queryKeys.runs.runtimeDiagnostics(
      runId,
      attempt.stageExecutionId,
    ),
    queryFn: ({ pageParam, signal }) =>
      getRuntimeDiagnostics(
        api,
        runId,
        attempt.stageExecutionId,
        pageParam,
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => last.page.nextCursor,
    enabled: open,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
    retry: (failures, error) =>
      failures < 2 &&
      ((error instanceof PublicAPIError && error.retryable) ||
        error instanceof TypeError),
    retryDelay: (attempt, error) =>
      error instanceof PublicAPIError && error.retryAfterMs !== undefined
        ? error.retryAfterMs
        : Math.min(1000 * 2 ** attempt, 5000),
  });
  const pages = query.data?.pages;
  const items = pages?.flatMap((page) => page.items);
  return (
    <div className="runs-block runs-block-wide">
      <h4 className="runs-block-title">Runtime adapter diagnostics</h4>
      <p className="runs-block-note">
        Retained delivery and shutdown facts. Missing measurements are unknown;
        discarded spans count local decisions, not confirmed remote loss.
      </p>
      <Button
        size="sm"
        disabled={query.isFetching || !open}
        onClick={() => {
          void query.refetch();
        }}
      >
        Refresh Runtime diagnostics
      </Button>
      {open && query.isPending ? <p>Loading Runtime diagnostics…</p> : null}
      {query.isRefetching && pages !== undefined ? (
        <p>
          Refreshing Runtime diagnostics. Previously loaded facts remain
          visible.
        </p>
      ) : null}
      {query.error === null ? null : <ErrorNotice error={query.error} />}
      {query.error !== null && items !== undefined ? (
        <p>Refresh failed. Previously loaded facts are shown.</p>
      ) : null}
      {pages?.[0]?.status === "available" &&
      pages[0].reason === "invalid_report" ? (
        <p>Some malformed participant reports could not be displayed.</p>
      ) : null}
      {pages?.[0]?.status === "pending" ? (
        <p>Reports have not arrived yet.</p>
      ) : null}
      {pages?.[0]?.status === "unavailable" ? (
        <p>
          Runtime diagnostics are unavailable; retained data may have expired.
        </p>
      ) : null}
      {pages?.[0]?.status === "available" && items?.length === 0 ? (
        <p>No Runtime participants were recorded.</p>
      ) : null}
      <ol className="runs-record-list">
        {items?.map((item) => (
          <li key={item.logicalAgent}>
            <strong>Logical Worker {item.logicalAgent}</strong>
            <p>Report {item.status}</p>
            <Facts
              items={[
                {
                  term: "Runtime duration (ms)",
                  value: item.durationMs ?? "unknown",
                },
                { term: "Stop reason", value: item.stopReason ?? "unknown" },
              ]}
            />
            {item.stopReasonTruncated ? (
              <p>Stop reason was shortened.</p>
            ) : null}
            {item.adapters.map((adapter) => (
              <div key={adapter.ref}>
                <code>{adapter.ref}</code>
                <Facts
                  items={[
                    { term: "Operations", value: adapter.operations },
                    {
                      term: "Failed operations",
                      value: adapter.failedOperations,
                    },
                    {
                      term: "Flush attempted",
                      value:
                        adapter.flushAttempted === undefined
                          ? "unknown"
                          : String(adapter.flushAttempted),
                    },
                    {
                      term: "Flush succeeded",
                      value:
                        adapter.flushSucceeded === undefined
                          ? "unknown"
                          : String(adapter.flushSucceeded),
                    },
                    {
                      term: "Last error",
                      value: adapter.lastErrorCode ?? "none reported",
                    },
                    ...Object.entries(adapter.droppedSpans ?? {}).map(
                      ([name, value]) => ({
                        term: DROP_LABELS[name as keyof typeof DROP_LABELS],
                        value,
                      }),
                    ),
                  ]}
                />
              </div>
            ))}
          </li>
        ))}
      </ol>
      {query.hasNextPage ? (
        <Button
          size="sm"
          disabled={query.isFetching}
          onClick={() => {
            void query.fetchNextPage();
          }}
        >
          Load more Runtime diagnostics
        </Button>
      ) : null}
    </div>
  );
}
