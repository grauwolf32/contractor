import { ContextLink } from "../app/context-navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { usePublicAPI } from "../api/context";
import {
  getOwnerQueueControl,
  listRunQueue,
  QUEUE_MEMBERSHIPS,
  QUEUE_STATES,
  setOwnerQueuePaused,
  type QueueItem,
  type QueueMembership,
  type QueueState,
} from "../api/queue";
import { queryKeys } from "../api/query-keys";
import { useRunEvents } from "../events/context";
import { CursorControls, ErrorNotice } from "./artifacts/common";
import { RunMetadataLabelChips, StateBadge } from "./runs/components";
import { RefreshButton } from "../app/refresh-button";
import { RecordedTime } from "../app/recorded-time";

const LIVE_SUBSCRIPTION_LIMIT = 24;
// Live subscriptions left waiting by a failed resync, or by a live session
// the Server refused, are retried at the Queue's own polling pace so neither
// a failing read nor a refused socket can spin.
const LIVE_RETRY_DELAY_MS = 10_000;

function compactRunId(runId: string): string {
  return runId.length <= 24
    ? runId
    : `${runId.slice(0, 12)}…${runId.slice(-8)}`;
}

function membershipLabel(value: QueueMembership): string {
  switch (value) {
    case "standalone":
      return "Standalone";
    case "project":
      return "Projects";
    case "evaluation":
      return "Evaluations";
  }
}

function QueueContext({ item }: { item: QueueItem }) {
  if (item.project === undefined) {
    return (
      <span className="queue-context">
        <strong>Standalone</strong>
        <small>No Project</small>
      </span>
    );
  }
  return (
    <span className="queue-context">
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

export function QueuePanel() {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedState = searchParams.get("state");
  const state = QUEUE_STATES.find(
    (candidate) => candidate === requestedState,
  ) as QueueState | undefined;
  const requestedMembership = searchParams.get("membership");
  const membership = QUEUE_MEMBERSHIPS.find(
    (candidate) => candidate === requestedMembership,
  ) as QueueMembership | undefined;
  // Page cursors live in the URL next to the filters they were issued for,
  // so any navigation that changes the filters (tab links, history) drops
  // them together.
  const cursors = searchParams.getAll("cursor");
  const cursor = cursors.at(-1);
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
  const controlQuery = useQuery({
    queryKey: queryKeys.queue.control,
    queryFn: () => getOwnerQueueControl(api),
  });
  const controlMutation = useMutation({
    mutationFn: ({ paused, revision }: { paused: boolean; revision: string }) =>
      setOwnerQueuePaused(api, paused, revision),
    onSuccess: (control) => {
      queryClient.setQueryData(queryKeys.queue.control, control);
    },
    onError: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.queue.control,
      });
    },
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

  function changePage(nextCursors: readonly string[]): void {
    const next = new URLSearchParams(searchParams);
    next.delete("cursor");
    for (const value of nextCursors) next.append("cursor", value);
    setSearchParams(next, { preventScrollReset: true });
  }

  return (
    <div className="panel queue-library run-view-panel">
      <div className="section-heading queue-heading">
        <div>
          <p className="eyebrow">Display order</p>
          <h3>Active queue · oldest first</h3>
          <p className="muted-copy">Position here is not Scheduler priority.</p>
        </div>
        <div className="queue-filters run-view-controls">
          <div
            className={`queue-control-state ${controlQuery.data?.paused === true ? "paused" : ""}`}
          >
            <span className="queue-control-label" aria-live="polite">
              <span aria-hidden="true" />
              {controlQuery.isPending
                ? "Checking admission…"
                : controlQuery.data?.paused === true
                  ? "Admission paused"
                  : "Admission running"}
            </span>
            <button
              className="secondary-button queue-control-button"
              type="button"
              disabled={
                controlQuery.data === undefined ||
                controlQuery.error !== null ||
                controlMutation.isPending
              }
              onClick={() => {
                const control = controlQuery.data;
                if (control !== undefined) {
                  controlMutation.mutate({
                    paused: !control.paused,
                    revision: control.revision,
                  });
                }
              }}
            >
              {controlMutation.isPending
                ? controlQuery.data?.paused === true
                  ? "Resuming…"
                  : "Pausing…"
                : controlQuery.data?.paused === true
                  ? "Resume queue"
                  : "Pause queue"}
            </button>
          </div>
          <label className="compact-select">
            State
            <select
              value={state ?? ""}
              onChange={(event) => replaceFilter("state", event.target.value)}
            >
              <option value="">All active states</option>
              {QUEUE_STATES.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {candidate}
                </option>
              ))}
            </select>
          </label>
          <label className="compact-select">
            Context
            <select
              value={membership ?? ""}
              onChange={(event) =>
                replaceFilter("membership", event.target.value)
              }
            >
              <option value="">All contexts</option>
              {QUEUE_MEMBERSHIPS.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {membershipLabel(candidate)}
                </option>
              ))}
            </select>
          </label>
          <RefreshButton
            isFetching={query.isFetching}
            onRefresh={() => void query.refetch()}
          />
        </div>
      </div>

      {controlQuery.error !== null ? (
        <div className="queue-control-error">
          <ErrorNotice error={controlQuery.error} />
        </div>
      ) : controlMutation.error !== null ? (
        <div className="queue-control-error">
          <ErrorNotice error={controlMutation.error} />
        </div>
      ) : controlQuery.data?.paused === true ? (
        <p className="queue-pause-note" role="status">
          Running Stages and cleanup will finish. No next Stage starts until you
          resume the queue.
        </p>
      ) : null}

      {query.isPending ? (
        <p className="loading-copy" role="status">
          Loading Queue…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="compact-empty">
          <strong>No active Runs match this view.</strong>
          <p>Terminal Runs remain available in execution history.</p>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="responsive-table run-list-table queue-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Workflow</th>
                <th>State</th>
                <th>Context</th>
                <th>Run metadata labels</th>
                <th>Created</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((item) => (
                <tr key={item.runId}>
                  <td className="run-list-id-cell" data-label="Run">
                    <ContextLink
                      returnLabel="Active queue"
                      className="run-list-id-link"
                      to={`/runs/${encodeURIComponent(item.runId)}`}
                      aria-label={item.runId}
                      title={item.runId}
                    >
                      {compactRunId(item.runId)}
                    </ContextLink>
                  </td>
                  <td className="run-list-workflow-cell" data-label="Workflow">
                    <span className="run-list-mobile-label">Workflow</span>
                    <code>{item.workflow}</code>
                  </td>
                  <td className="run-list-state-cell" data-label="State">
                    <StateBadge state={item.state} />
                  </td>
                  <td className="queue-context-cell" data-label="Context">
                    <QueueContext item={item} />
                  </td>
                  <td
                    className={`run-list-labels-cell ${Object.keys(item.labels).length === 0 ? "run-list-labels-empty" : ""}`}
                    data-label="Run metadata labels"
                  >
                    <span className="run-list-mobile-label">Labels</span>
                    <RunMetadataLabelChips labels={item.labels} />
                  </td>
                  <td className="run-list-created-cell" data-label="Created">
                    <RecordedTime value={item.createdAt} />
                  </td>
                  <td className="run-list-updated-cell" data-label="Updated">
                    <span className="run-list-mobile-label">Updated</span>
                    <RecordedTime value={item.updatedAt} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <CursorControls
        label="Queue pages"
        canGoBack={cursors.length > 0}
        {...(query.data?.page.hasMore === true &&
        query.data.page.nextCursor !== undefined
          ? { nextCursor: query.data.page.nextCursor }
          : {})}
        onBack={() => changePage(cursors.slice(0, -1))}
        onNext={(next) => changePage([...cursors, next])}
      />
    </div>
  );
}
