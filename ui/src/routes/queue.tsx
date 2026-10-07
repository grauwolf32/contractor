import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { usePublicAPI } from "../api/context";
import {
  listRunQueue,
  QUEUE_MEMBERSHIPS,
  QUEUE_STATES,
  type QueueItem,
  type QueueMembership,
  type QueueState,
} from "../api/queue";
import { queryKeys } from "../api/query-keys";
import { useRunEvents } from "../events/context";
import { CursorControls } from "../app/cursor-controls";
import { useURLCursorStack } from "../app/pagination";
import { RefreshButton } from "../app/refresh-button";
import { RecordedTime } from "../app/recorded-time";
import { QueryView } from "../app/query-view";
import { EmptyState } from "../ui";
import { RunMetadataLabelChips, RunStateChip } from "./runs/components";
import { RunIdLink, RunsFilter } from "./runs/list-parts";
import { MEMBERSHIP_LABELS, RUN_STATE_LABELS } from "./runs/run-state";

const LIVE_SUBSCRIPTION_LIMIT = 24;
// Live subscriptions left waiting by a failed resync, or by a live session
// the Server refused, are retried at the Queue's own polling pace so neither
// a failing read nor a refused socket can spin.
const LIVE_RETRY_DELAY_MS = 10_000;

const STATE_OPTIONS = [
  { value: "" as const, label: "All" },
  ...QUEUE_STATES.map((state) => ({
    value: state,
    label: RUN_STATE_LABELS[state].label,
  })),
];

const MEMBERSHIP_OPTIONS = [
  { value: "" as const, label: "All" },
  ...QUEUE_MEMBERSHIPS.map((membership) => ({
    value: membership,
    label: MEMBERSHIP_LABELS[membership],
  })),
];

function QueueContext({ item }: { item: QueueItem }) {
  if (item.project === undefined) {
    return (
      <span className="runs-context">
        <strong>Standalone</strong>
        <small>No Project</small>
      </span>
    );
  }
  return (
    <span className="runs-context">
      <Link
        to={`${item.project.kind === "evaluation" ? "/evals" : "/projects"}/${encodeURIComponent(item.project.projectId)}`}
      >
        {item.project.name}
      </Link>
      <small>
        {item.project.kind === "evaluation" ? "Evaluation" : "Project"}
      </small>
    </span>
  );
}

function useQueueInvalidation(items: readonly QueueItem[]): void {
  const events = useRunEvents();
  const queryClient = useQueryClient();
  const targets = useMemo(
    () =>
      items.slice(0, LIVE_SUBSCRIPTION_LIMIT).map((item) => ({
        runId: item.runId,
        generation: item.eventCursor.generation,
        sequence: item.eventCursor.sequence,
      })),
    [items],
  );
  const targetFingerprint = targets
    .map(
      ({ runId, generation, sequence }) => `${runId}:${generation}:${sequence}`,
    )
    .join("\u0000");

  // A resync leaves subscriptions waiting for an authoritative cursor, so
  // resubscribe after the refetch even when the cursors did not change.
  const [resyncs, setResyncs] = useState(0);

  useEffect(() => {
    let active = true;
    let resyncing = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const invalidate = () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.queue.all });
    };
    const resubscribe = () => {
      if (active) setResyncs((count) => count + 1);
    };
    const resubscribeLater = () => {
      if (!active || retry !== undefined) return;
      retry = setTimeout(resubscribe, LIVE_RETRY_DELAY_MS);
    };
    const resync = () => {
      if (resyncing) return;
      resyncing = true;
      void queryClient
        .refetchQueries(
          { queryKey: queryKeys.queue.all },
          { throwOnError: true },
        )
        .then(resubscribe, resubscribeLater);
    };
    const subscriptions = targets.map((target) =>
      events.subscribeRun(
        target.runId,
        { generation: target.generation, sequence: target.sequence },
        {
          onPlannerEvent: () => true,
          onLifecycleEvent: invalidate,
          onResync: resync,
          // The manager stops without a resync request after the Server
          // refuses the live session (close 1008); a still-valid session
          // resubscribes, an expired one surfaces through the Queue reads.
          onStateChange: (state) => {
            if (state === "error") resubscribeLater();
          },
          onError: () => undefined,
        },
      ),
    );
    return () => {
      active = false;
      clearTimeout(retry);
      for (const subscription of subscriptions) {
        subscription.unsubscribe();
      }
    };
    // The fingerprint intentionally makes exact cursors the subscription key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events, queryClient, targetFingerprint, resyncs]);
}

/**
 * Runs → Queue: the owner-scoped active queue (S17/S18), oldest first,
 * polled and refreshed by live lifecycle events for its first rows. State
 * and context filters and the page cursor live in the URL.
 */
export function QueuePanel() {
  const api = usePublicAPI();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedState = searchParams.get("state");
  const state = QUEUE_STATES.find(
    (candidate) => candidate === requestedState,
  ) as QueueState | undefined;
  const requestedMembership = searchParams.get("membership");
  const membership = QUEUE_MEMBERSHIPS.find(
    (candidate) => candidate === requestedMembership,
  ) as QueueMembership | undefined;
  const pages = useURLCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.queue.list(state, membership, cursor),
    queryFn: () =>
      listRunQueue(api, {
        ...(state === undefined ? {} : { state }),
        ...(membership === undefined ? {} : { membership }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
    refetchInterval: 10_000,
  });

  useQueueInvalidation(query.data?.items ?? []);

  function replaceFilter(name: "state" | "membership", value: string): void {
    const next = new URLSearchParams(searchParams);
    next.delete("cursor");
    if (value === "") {
      next.delete(name);
    } else {
      next.set(name, value);
    }
    setSearchParams(next, { replace: true });
  }

  return (
    <div className="runs-view">
      <div className="runs-toolbar">
        <div className="runs-filters">
          <RunsFilter
            label="State"
            options={STATE_OPTIONS}
            value={state ?? ""}
            onChange={(value) => replaceFilter("state", value)}
          />
          <RunsFilter
            label="Context"
            options={MEMBERSHIP_OPTIONS}
            value={membership ?? ""}
            onChange={(value) => replaceFilter("membership", value)}
          />
        </div>
        <RefreshButton
          className="runs-icon-button"
          isFetching={query.isFetching}
          onRefresh={() => void query.refetch()}
        />
      </div>
      <p className="runs-note">
        Oldest first. Position here is not Scheduler priority.
      </p>

      <div className="runs-results">
        <QueryView
          query={query}
          loading={
            <p className="runs-loading" role="status">
              Loading Queue…
            </p>
          }
          onRetry={() => void query.refetch()}
          isEmpty={(queryData) => queryData.items.length === 0}
          empty={
            <EmptyState title="No active Runs match this view.">
              <p>Terminal Runs remain available in Completed.</p>
            </EmptyState>
          }
        >
          {(queryData) => (
            <table className="runs-table">
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Workflow</th>
                  <th scope="col">State</th>
                  <th scope="col">Context</th>
                  <th scope="col">Labels</th>
                  <th scope="col">Created</th>
                  <th scope="col">Updated</th>
                </tr>
              </thead>
              <tbody>
                {queryData.items.map((item) => {
                  const unlabelled = Object.keys(item.labels).length === 0;
                  return (
                    <tr key={item.runId}>
                      <td className="run-list-id-cell">
                        <RunIdLink runId={item.runId} returnLabel="Queue" />
                      </td>
                      <td className="run-list-workflow-cell">
                        <span className="runs-cell-label">Workflow</span>
                        <code>{item.workflow}</code>
                      </td>
                      <td className="run-list-state-cell">
                        <RunStateChip state={item.state} />
                      </td>
                      <td className="queue-context-cell">
                        <span className="runs-cell-label">Context</span>
                        <QueueContext item={item} />
                      </td>
                      <td
                        className="run-list-labels-cell"
                        data-empty={unlabelled ? "" : undefined}
                      >
                        <span className="runs-cell-label">Labels</span>
                        <RunMetadataLabelChips
                          labels={item.labels}
                          empty="None"
                        />
                      </td>
                      <td className="run-list-created-cell">
                        <RecordedTime value={item.createdAt} />
                      </td>
                      <td className="run-list-updated-cell">
                        <span className="runs-cell-label">Updated</span>
                        <RecordedTime value={item.updatedAt} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </QueryView>
      </div>

      <CursorControls
        label="Queue pages"
        {...pages.controls(query.data?.page)}
      />
    </div>
  );
}
