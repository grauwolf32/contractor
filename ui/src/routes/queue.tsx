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
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "./artifacts/common";
import { RunMetadataLabelChips, StateBadge } from "./runs/components";

const LIVE_SUBSCRIPTION_LIMIT = 24;

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
      <Link to={`/projects/${encodeURIComponent(item.project.projectId)}`}>
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

  useEffect(() => {
    const invalidate = () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.queue.all });
    };
    const subscriptions = targets.map((target) =>
      events.subscribeRun(
        target.runId,
        { generation: target.generation, sequence: target.sequence },
        {
          onPlannerEvent: () => true,
          onLifecycleEvent: invalidate,
          onResync: invalidate,
          onStateChange: () => undefined,
          onError: () => undefined,
        },
      ),
    );
    return () => {
      for (const subscription of subscriptions) {
        subscription.unsubscribe();
      }
    };
    // The fingerprint intentionally makes exact cursors the subscription key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events, queryClient, targetFingerprint]);
}

export function QueueRoute() {
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
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
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

  useQueueInvalidation(query.data?.items ?? []);

  function replaceFilter(name: "state" | "membership", value: string): void {
    const next = new URLSearchParams(searchParams);
    if (value === "") {
      next.delete(name);
    } else {
      next.set(name, value);
    }
    setSearchParams(next, { replace: true });
    setCursors([undefined]);
  }

  return (
    <section className="route-page queue-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Owner-scoped active work</p>
          <h2>Queue</h2>
          <p className="lede">
            One read-only view of standalone, Project, and evaluation Runs that
            are still in progress.
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      <div className="panel queue-library">
        <div className="section-heading queue-heading">
          <div>
            <p className="eyebrow">Display order</p>
            <h3>Oldest created first</h3>
            <p className="muted-copy">
              This stable display order is not Scheduler priority, rank, or an
              estimated start position.
            </p>
          </div>
          <div className="queue-filters">
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
          </div>
        </div>

        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
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
                      <Link
                        className="run-list-id-link"
                        to={`/runs/${encodeURIComponent(item.runId)}`}
                        aria-label={item.runId}
                        title={item.runId}
                      >
                        {compactRunId(item.runId)}
                      </Link>
                    </td>
                    <td
                      className="run-list-workflow-cell"
                      data-label="Workflow"
                    >
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
                      <time dateTime={item.createdAt}>
                        {formatTimestamp(item.createdAt)}
                      </time>
                    </td>
                    <td className="run-list-updated-cell" data-label="Updated">
                      <span className="run-list-mobile-label">Updated</span>
                      <time dateTime={item.updatedAt}>
                        {formatTimestamp(item.updatedAt)}
                      </time>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <CursorControls
          label="Queue pages"
          canGoBack={cursors.length > 1}
          {...(query.data?.page.hasMore === true &&
          query.data.page.nextCursor !== undefined
            ? { nextCursor: query.data.page.nextCursor }
            : {})}
          onBack={() =>
            setCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setCursors((current) => [...current, next])}
        />
      </div>
    </section>
  );
}
