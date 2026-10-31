import { useQuery } from "@tanstack/react-query";
import { type ReactNode } from "react";
import { Link } from "react-router";
import { usePublicAPI } from "../../api/context";
import { listProjectRuns } from "../../api/projects";
import { listRunQueue } from "../../api/queue";
import { queryKeys } from "../../api/query-keys";
import { isTerminalRunState, listRuns, type RunPage } from "../../api/runs";
import { compactId } from "../../app/format";
import { pollInterval } from "../../app/polling";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { RefreshButton } from "../../app/refresh-button";
import { EmptyState, ListPane, ListRow, PaneLayout } from "../../ui";
import { RunStateChip } from "./components";
import type { RunListOrigin } from "./source-context";

type SourcePage = RunPage | Awaited<ReturnType<typeof listRunQueue>>;

/** The same one page that supplied the clicked Run, with no view switcher. */
function SourceRunList({
  origin,
  runId,
}: {
  origin: RunListOrigin;
  runId: string;
}) {
  const api = usePublicAPI();
  const source = origin.runSource;
  const queryKey =
    source.kind === "queue"
      ? queryKeys.queue.list(
          source.options.state,
          source.options.membership,
          source.options.cursor,
        )
      : source.kind === "completed"
        ? queryKeys.runs.list(
            source.options.state,
            source.options.cursor,
            (source.options.labelSelectors ?? []).map(
              ({ key, value }) => `${key}=${value}`,
            ),
            "terminal",
          )
        : queryKeys.projects.runView(source.projectId, source.options);
  const query = useQuery<SourcePage>({
    queryKey,
    queryFn: () =>
      source.kind === "queue"
        ? listRunQueue(api, source.options)
        : source.kind === "completed"
          ? listRuns(api, source.options)
          : listProjectRuns(api, {
              projectId: source.projectId,
              ...source.options,
            }),
    refetchInterval:
      source.refreshMs === undefined
        ? false
        : pollInterval<SourcePage>({
            interval: source.refreshMs,
            ...(source.settleWhenTerminal
              ? {
                  settled: (page: SourcePage) =>
                    page.items.every((run) => isTerminalRunState(run.state)),
                }
              : {}),
          }),
  });
  return (
    <ListPane
      title={source.label}
      titleAs="h2"
      subtitle={
        query.data === undefined
          ? undefined
          : `${query.data.items.length} shown`
      }
      actions={
        <RefreshButton
          size="sm"
          isFetching={query.isFetching}
          onRefresh={() => void query.refetch()}
        />
      }
      footer={
        <Link to={origin.returnTo} state={origin.returnState}>
          Back to {origin.returnLabel}
        </Link>
      }
    >
      <QueryView
        query={query}
        errorContext="Could not load the source Run list"
        isEmpty={(page) => page.items.length === 0}
        empty={
          <EmptyState title="No Runs in this source list">
            The selected Run remains open on the right.
          </EmptyState>
        }
      >
        {(data) => (
          <>
            {!data.items.some((run) => run.runId === runId) ? (
              <p className="runs-source-note">
                This Run is not in the current source list.
              </p>
            ) : null}
            <ul role="list" className="runs-source-list">
              {data.items.map((run) => {
                const finished =
                  "finishedAt" in run ? run.finishedAt : undefined;
                return (
                  <ListRow
                    key={run.runId}
                    to={`/runs/${encodeURIComponent(run.runId)}`}
                    state={origin}
                    selected={run.runId === runId}
                    title={
                      <>
                        {run.workflow}
                        <span className="ui-visually-hidden">
                          {" "}
                          · {run.runId}
                        </span>
                      </>
                    }
                    meta={[
                      <RunStateChip key="state" state={run.state} size="sm" />,
                      <code key="id" title={run.runId}>
                        {compactId(run.runId)}
                      </code>,
                      <span key="time">
                        {finished ? "Finished" : "Updated"}{" "}
                        <RecordedTime
                          display="clock"
                          value={finished ?? run.updatedAt}
                        />
                      </span>,
                    ]}
                  />
                );
              })}
            </ul>
            {data.page.hasMore ? (
              <Link
                className="runs-source-note"
                to={origin.returnTo}
                state={origin.returnState}
              >
                More Runs in the source list
              </Link>
            ) : null}
          </>
        )}
      </QueryView>
    </ListPane>
  );
}

export function RunSourceWorkspace({
  origin,
  runId,
  children,
}: {
  origin: RunListOrigin;
  runId: string;
  children: ReactNode;
}) {
  return (
    <PaneLayout
      className="runs-source-workspace"
      listLabel="Source Runs"
      detailLabel="Run"
      showDetail
      backLink={{
        to: origin.returnTo,
        label: `Back to ${origin.returnLabel}`,
        state: origin.returnState,
      }}
      list={<SourceRunList origin={origin} runId={runId} />}
      detail={children}
    />
  );
}
