import {
  useInfiniteQuery,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { queryKeys } from "../../api/query-keys";
import { getToolCallDiagnostics } from "../../api/tool-diagnostics";
import type { StageAttempt } from "../../api/runs";
import { ErrorNotice } from "../../app/error-notice";
import { Button, Facts, Field } from "../../ui";

export function ToolCallDiagnostics({
  runId,
  attempt,
  open,
}: {
  runId: string;
  attempt: Pick<StageAttempt, "stageExecutionId" | "executionConfig">;
  open: boolean;
}) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const workers = Object.keys(attempt.executionConfig.agents).sort();
  // Keep the user's participant choice through live-projection remounts. This
  // local cache has no server request and is separate from diagnostic facts.
  const selectionKey = queryKeys.runs.toolCallSelection(
    runId,
    attempt.stageExecutionId,
  );
  const { data: selection } = useQuery({
    queryKey: selectionKey,
    queryFn: () => "planner",
    initialData: "planner",
    enabled: false,
    staleTime: Infinity,
  });
  const participant = selection === "planner" ? "planner" : "worker";
  const logicalAgent =
    participant === "planner" ? undefined : selection.slice(7);
  const query = useInfiniteQuery({
    queryKey: [
      ...queryKeys.runs.toolCalls(runId, attempt.stageExecutionId),
      participant,
      logicalAgent ?? null,
    ],
    queryFn: ({ pageParam, signal }) =>
      getToolCallDiagnostics(
        api,
        runId,
        attempt.stageExecutionId,
        participant,
        logicalAgent,
        pageParam,
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => last.page.nextCursor,
    enabled: open,
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
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
      <h4 className="runs-block-title">Tool calls</h4>
      <p className="runs-block-note">
        Retained sanitized call detail, not a complete event history. Durations
        are measurements; call start times were not recorded. Missing
        measurements are unknown.
      </p>
      <Field label="Tool-call participant">
        {(control) => (
          <select
            {...control}
            value={selection}
            onChange={(event) =>
              client.setQueryData(selectionKey, event.target.value)
            }
          >
            <option value="planner">Planner</option>
            {workers.map((name) => (
              <option key={name} value={`worker:${name}`}>
                Logical Worker {name}
              </option>
            ))}
          </select>
        )}
      </Field>
      <Button
        size="sm"
        disabled={query.isFetching || !open}
        onClick={() => {
          void query.refetch();
        }}
      >
        Refresh tool calls
      </Button>
      {open && query.isPending ? <p>Loading retained tool calls…</p> : null}
      {query.isRefetching && pages !== undefined ? (
        <p>
          Refreshing retained tool calls. Previously loaded facts remain
          visible.
        </p>
      ) : null}
      {query.error === null ? null : (
        <>
          <ErrorNotice error={query.error} />
          {items?.length ? (
            <p>
              Previously loaded call detail remains visible; refresh failed.
            </p>
          ) : null}
        </>
      )}
      {pages?.[0]?.status === "pending" ? (
        <p>The selected report has not arrived yet.</p>
      ) : null}
      {pages?.[0]?.status === "unavailable" ? (
        <p>
          Retained call detail is unavailable ({pages[0].reason}); telemetry may
          have expired.
        </p>
      ) : null}
      {pages?.[0]?.complete === false ? (
        <p>The selected report is incomplete.</p>
      ) : null}
      {pages?.[0]?.detailTruncated ? (
        <p>
          Retained detail was shortened; older records may have been removed.
        </p>
      ) : null}
      {pages?.[0]?.status === "available" && items?.length === 0 ? (
        <p>No retained tool calls in the selected report.</p>
      ) : null}
      <ol className="runs-record-list">
        {items?.map((item) => (
          <li key={item.callReference}>
            <strong>{item.tool}</strong>
            <p>Outcome {item.outcome}</p>
            {item.toolTruncated ? (
              <p>
                The source tool name could not be shown within the diagnostic
                bound.
              </p>
            ) : null}
            <Facts
              items={[
                { term: "Call reference", value: item.callReference },
                { term: "Duration (ms)", value: item.durationMs ?? "unknown" },
                {
                  term: "Result size (bytes)",
                  value: item.resultSizeBytes ?? "unknown",
                },
                ...(item.error === undefined
                  ? []
                  : [
                      { term: "Error code", value: item.error.code },
                      { term: "Error", value: item.error.message },
                      {
                        term: "Retryable",
                        value:
                          item.error.retryable === undefined
                            ? "unknown"
                            : String(item.error.retryable),
                      },
                    ]),
              ]}
            />
            {item.error?.messageTruncated ? (
              <p>Error message was shortened.</p>
            ) : null}
            {item.argumentsJson === undefined ? (
              <p>No argument summary was retained.</p>
            ) : (
              <details>
                <summary>Sanitized arguments</summary>
                <p className="runs-tool-arguments">
                  <code>{item.argumentsJson}</code>
                </p>
              </details>
            )}
            {item.argumentsTruncated ? (
              <p>Argument summary was shortened or redacted.</p>
            ) : null}
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
          Load more tool calls
        </Button>
      ) : null}
    </div>
  );
}
