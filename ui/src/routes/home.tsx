import { useDocumentTitle } from "../app/document-title";
import { RecordedTime } from "../app/recorded-time";
import { getOwnerQueueControl } from "../api/queue";
import { ContextLink } from "../app/context-navigation";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";

import { usePublicAPI } from "../api/context";
import {
  getOperationsSnapshot,
  type OperationsSnapshot,
} from "../api/operations";
import { queryKeys } from "../api/query-keys";
import {
  listRuns,
  type RunPage,
  type RunSummary,
  type WorkflowRunState,
} from "../api/runs";
import type { WorkflowSummary } from "../api/workflows";
import { useWorkflowInventory } from "./workflows/inventory";
import { groupWorkflowVersions } from "./workflows/families";
import { workflowDisplayName } from "./workflows/presentation";
import { useSession } from "../auth/session";
import { ErrorNotice, formatTimestamp } from "./artifacts/common";
import { StateBadge } from "./runs/components";
import { formatRunDuration } from "./runs/triage";
import { RefreshButton } from "../app/refresh-button";

type ActiveRunState = Extract<
  WorkflowRunState,
  "initializing" | "pending" | "running" | "waiting" | "cancelling"
>;

function compactRunId(runId: string): string {
  return runId.length <= 20 ? runId : `${runId.slice(0, 16)}…`;
}

function runDuration(run: RunSummary): string {
  const start = Date.parse(run.createdAt);
  const end = Date.parse(run.finishedAt ?? run.updatedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
    return "—";
  }
  return formatRunDuration(end - start);
}

function boundedCount(pages: Array<RunPage | undefined>): string {
  const count = pages.reduce(
    (total, page) => total + (page?.items.length ?? 0),
    0,
  );
  return `${count}${pages.some((page) => page?.page.hasMore === true) ? "+" : ""}`;
}

function sortRuns(runs: RunSummary[]): RunSummary[] {
  return [...runs].sort((left, right) => {
    const byCreated = right.createdAt.localeCompare(left.createdAt);
    return byCreated === 0 ? right.runId.localeCompare(left.runId) : byCreated;
  });
}

function RunRows({
  runs,
  emptyTitle,
  emptyCopy,
}: {
  runs: RunSummary[];
  emptyTitle: string;
  emptyCopy: string;
}) {
  if (runs.length === 0) {
    return (
      <div className="action-empty">
        <strong>{emptyTitle}</strong>
        <span>{emptyCopy}</span>
      </div>
    );
  }
  return (
    <div className="action-run-list">
      {runs.map((run) => (
        <ContextLink
          returnLabel="Action center"
          className="action-run-row"
          key={run.runId}
          to={`/runs/${encodeURIComponent(run.runId)}`}
        >
          <span className="action-run-identity">
            <strong>{run.workflow}</strong>
            <code title={run.runId}>{compactRunId(run.runId)}</code>
          </span>
          <span className="action-run-state">
            <StateBadge state={run.state} />
            <small>{runDuration(run)}</small>
          </span>
          <RecordedTime value={run.finishedAt ?? run.updatedAt} />
        </ContextLink>
      ))}
    </div>
  );
}

function AttentionPanel({
  query,
}: {
  query: ReturnType<typeof useRecentRuns>;
}) {
  const failed =
    query.data?.items.filter((run) => run.state === "failed").slice(0, 5) ?? [];
  return (
    <section className="panel action-panel action-attention">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Execution history</p>
          <h3>Recent failures</h3>
        </div>
        <Link to="/runs?view=completed&state=failed">View failed Runs →</Link>
      </div>
      <p className="action-panel-copy">
        Failed executions among the newest 50 owned Runs. Dates indicate their
        age; these are historical outcomes.
      </p>
      {query.isPending ? (
        <p className="loading-copy" role="status">
          Loading recent activity…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <RunRows
          runs={failed}
          emptyTitle="No recent failures"
          emptyCopy="The latest execution window has nothing requiring triage."
        />
      )}
    </section>
  );
}

type ActiveQuery = ReturnType<typeof useActiveRun>;

function ActivePanel({ queries }: { queries: ActiveQuery[] }) {
  const runs = sortRuns(
    queries.flatMap((query) => query.data?.items ?? []),
  ).slice(0, 6);
  const pending = queries.some((query) => query.isPending);
  const errors = queries.flatMap((query) =>
    query.error === null ? [] : [query.error],
  );
  return (
    <section className="panel action-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">In progress</p>
          <h3>Active Runs</h3>
        </div>
        <Link to="/runs">Open active queue →</Link>
      </div>
      <p className="action-panel-copy">
        Runs that are starting, running, or stopping.
      </p>
      {pending && runs.length === 0 ? (
        <p className="loading-copy" role="status">
          Loading active Runs…
        </p>
      ) : errors.length === queries.length ? (
        <ErrorNotice error={errors[0]} />
      ) : (
        <>
          {errors.length === 0 ? null : (
            <div className="action-inline-warning" role="status">
              Some active Runs could not be loaded.
            </div>
          )}
          <RunRows
            runs={runs}
            emptyTitle="No active Runs"
            emptyCopy="Start a published Workflow when work is ready."
          />
        </>
      )}
    </section>
  );
}

function RecentSuccessPanel({
  query,
}: {
  query: ReturnType<typeof useRecentRuns>;
}) {
  const succeeded =
    query.data?.items.filter((run) => run.state === "succeeded").slice(0, 5) ??
    [];
  return (
    <section className="panel action-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Ready to inspect</p>
          <h3>Recent successful Runs</h3>
        </div>
        <Link to="/runs?view=completed&state=succeeded">View successes →</Link>
      </div>
      <p className="action-panel-copy">Open a Run to preview its results.</p>
      {query.isPending ? (
        <p className="loading-copy" role="status">
          Loading recent activity…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <RunRows
          runs={succeeded}
          emptyTitle="No recent successes"
          emptyCopy="Completed output-producing Runs will appear here."
        />
      )}
    </section>
  );
}

interface RuntimeHealth {
  idle: number;
  occupied: number;
  unavailable: number;
  mismatches: number;
}

function runtimeHealth(snapshot: OperationsSnapshot): RuntimeHealth {
  return {
    idle: snapshot.runtimeAgents.filter((agent) => agent.slotState === "idle")
      .length,
    occupied: snapshot.runtimeAgents.filter(
      (agent) => agent.slotState === "reserved" || agent.slotState === "busy",
    ).length,
    unavailable: snapshot.runtimeAgents.filter(
      (agent) => agent.slotState === "draining" || agent.slotState === "fenced",
    ).length,
    mismatches: snapshot.runtimeAgents.filter(
      (agent) =>
        agent.reconciliationReason !== undefined ||
        agent.currentAllocationId !== agent.authoritativeAllocationId,
    ).length,
  };
}

function RuntimeHealthPanel({
  authorized,
  query,
}: {
  authorized: boolean;
  query: ReturnType<typeof useOperations>;
}) {
  if (!authorized) {
    return (
      <section className="panel action-panel">
        <p className="eyebrow">Runtime health</p>
        <h3>Operations access required</h3>
        <p className="action-panel-copy">
          Run activity remains available, but this session cannot observe
          Control Plane capacity.
        </p>
      </section>
    );
  }
  if (query.isPending) {
    return (
      <section className="panel action-panel">
        <p className="eyebrow">Runtime health</p>
        <h3>Loading capacity…</h3>
      </section>
    );
  }
  if (query.error !== null) {
    return (
      <section className="panel action-panel">
        <p className="eyebrow">Runtime health</p>
        <h3>Capacity unavailable</h3>
        <ErrorNotice error={query.error} />
      </section>
    );
  }
  const health = runtimeHealth(query.data);
  const attention = health.mismatches > 0 || health.unavailable > 0;
  return (
    <section className="panel action-panel runtime-health-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Runtime health</p>
          <h3>
            {attention ? "Capacity needs attention" : "Runtime slots available"}
          </h3>
        </div>
        <span
          className={`health-indicator ${attention ? "health-attention" : "health-ready"}`}
        >
          {attention ? "Attention" : "Runtime healthy"}
        </span>
      </div>
      <dl className="runtime-health-grid">
        <div>
          <dt>Idle slots</dt>
          <dd>{health.idle}</dd>
        </div>
        <div>
          <dt>Occupied</dt>
          <dd>{health.occupied}</dd>
        </div>
        <div>
          <dt>Unavailable</dt>
          <dd>{health.unavailable}</dd>
        </div>
        <div>
          <dt>Mismatches</dt>
          <dd>{health.mismatches}</dd>
        </div>
        <div>
          <dt>Allocations</dt>
          <dd>{query.data.allocations.length}</dd>
        </div>
      </dl>
      <Link className="action-panel-footer-link" to="/operations">
        Open Operations snapshot →
      </Link>
    </section>
  );
}

function workflowContractSummary(workflow: WorkflowSummary): string {
  const requiredParameters = Object.values(workflow.parameters).filter(
    (slot) => slot.required,
  ).length;
  const requiredInputs = Object.values(workflow.inputs).filter(
    (slot) => slot.required,
  ).length;
  return `${requiredParameters} required parameters · ${requiredInputs} required inputs · ${Object.keys(workflow.outputs).length} outputs`;
}

function QuickStartPanel({
  query,
}: {
  query: ReturnType<typeof useWorkflowInventory>;
}) {
  const workflows = groupWorkflowVersions(query.data ?? [])
    .slice(0, 6)
    .map((family) => family.versions[0]!);
  return (
    <section className="panel action-panel quick-start-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Quick start</p>
          <h3>Choose a Workflow</h3>
        </div>
        <Link to="/catalog/workflows">Browse catalog →</Link>
      </div>
      <p className="action-panel-copy">
        Start with a Project to keep inputs and results together.{" "}
        <Link to="/projects">Open Projects →</Link>
      </p>
      {query.isPending ? (
        <p className="loading-copy" role="status">
          Loading Workflow catalog…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : workflows.length === 0 ? (
        <div className="action-empty">
          <strong>No published Workflows</strong>
          <span>Publish a Workflow version before starting work.</span>
        </div>
      ) : (
        <div className="quick-workflow-grid">
          {workflows.map((workflow) => (
            <Link
              key={`${workflow.ref.name}@${workflow.ref.version}`}
              to={`/catalog/workflows/${encodeURIComponent(workflow.ref.name)}/${encodeURIComponent(workflow.ref.version)}`}
            >
              <span>
                <strong>{workflowDisplayName(workflow)}</strong>
                <code>@{workflow.ref.version}</code>
              </span>
              <small>{workflowContractSummary(workflow)}</small>
              <em>Inspect workflow →</em>
            </Link>
          ))}
        </div>
      )}
    </section>
  );
}

function useRecentRuns() {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.runs.list(undefined, undefined),
    queryFn: () => listRuns(api),
  });
}

function useActiveRun(state: ActiveRunState) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.runs.list(state, undefined),
    queryFn: () => listRuns(api, { state }),
  });
}

function useOperations(authorized: boolean) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.operations.snapshot,
    queryFn: () => getOperationsSnapshot(api),
    enabled: authorized,
  });
}

export function HomeRoute() {
  useDocumentTitle("Home");
  const { session } = useSession();
  const operationsAuthorized =
    session?.principal.capabilities.includes("operations") === true;
  const api = usePublicAPI();
  const admission = useQuery({
    queryKey: queryKeys.queue.control,
    queryFn: () => getOwnerQueueControl(api),
    refetchInterval: 10000,
  });
  const recent = useRecentRuns();
  const initializing = useActiveRun("initializing");
  const running = useActiveRun("running");
  const cancelling = useActiveRun("cancelling");
  const pending = useActiveRun("pending");
  const waiting = useActiveRun("waiting");
  const activeQueries = [initializing, pending, running, waiting, cancelling];
  const workflows = useWorkflowInventory();
  const operations = useOperations(operationsAuthorized);
  const activePages = activeQueries.map((query) => query.data);
  const activeCount = activePages.every((page) => page !== undefined)
    ? boundedCount(activePages)
    : "—";
  const recentFailures = recent.data?.items.filter(
    (run) => run.state === "failed",
  ).length;
  const health =
    operations.data === undefined ? undefined : runtimeHealth(operations.data);
  const fetching =
    recent.isFetching ||
    activeQueries.some((query) => query.isFetching) ||
    workflows.isFetching ||
    (operationsAuthorized && operations.isFetching);
  const updatedAt = Math.max(
    recent.dataUpdatedAt,
    ...activeQueries.map((query) => query.dataUpdatedAt),
    workflows.dataUpdatedAt,
    operationsAuthorized ? operations.dataUpdatedAt : 0,
  );

  function refresh(): void {
    const refreshes: Array<Promise<unknown>> = [
      admission.refetch(),
      recent.refetch(),
      ...activeQueries.map((query) => query.refetch()),
      workflows.refetch(),
    ];
    if (operationsAuthorized) {
      refreshes.push(operations.refetch());
    }
    void Promise.all(refreshes);
  }

  return (
    <section className="route-page action-center-page">
      <header className="route-header-row action-center-header">
        <div>
          <p className="eyebrow">Operator workspace</p>
          <h2>Action center</h2>
          <p className="lede">
            Current work, recent results and the next Run to start.
          </p>
        </div>
        <div className="action-refresh">
          <small>
            {updatedAt === 0
              ? "Waiting for snapshots"
              : `Updated ${formatTimestamp(new Date(updatedAt).toISOString())}`}
          </small>
          <RefreshButton isFetching={fetching} onRefresh={refresh} />
        </div>
      </header>

      <div
        className={`notice ${admission.data?.paused ? "notice-warning" : ""}`}
        role="status"
      >
        <strong>
          {admission.isPending
            ? "Checking queue admission…"
            : admission.error
              ? "Queue admission unavailable"
              : admission.data.paused
                ? "Queue paused"
                : "Queue admission open"}
        </strong>
        <p>
          {admission.data?.paused
            ? "New stage attempts are paused for your Runs. Idle Runtime slots do not bypass this pause."
            : "Queue admission and Runtime health are independent. Check the queue for current work."}{" "}
          <Link to="/runs">Open queue →</Link>
        </p>
      </div>
      <section className="action-metric-grid" aria-label="Current overview">
        <Link className="action-metric-card" to="/runs">
          <span>Active Runs</span>
          <strong>{activeCount}</strong>
          <small>initializing, pending, running, waiting, cancelling</small>
        </Link>
        <Link
          className={`action-metric-card ${recent.data?.items.some((run) => run.state === "failed" && recent.dataUpdatedAt - Date.parse(run.finishedAt ?? run.updatedAt) < 86400000) ? "metric-attention" : ""}`}
          to="/runs?view=completed&state=failed"
        >
          <span>Recent failures</span>
          <strong>{recentFailures ?? "—"}</strong>
          <small>among the newest 50 Runs</small>
        </Link>
        <Link
          className={`action-metric-card ${health !== undefined && (health.mismatches > 0 || health.unavailable > 0) ? "metric-attention" : ""}`}
          to={
            operationsAuthorized ? "/operations/runtime-agents" : "/operations"
          }
        >
          <span>Idle Runtime slots</span>
          <strong>{health?.idle ?? "—"}</strong>
          <small>
            {!operationsAuthorized
              ? "operations access required"
              : operations.data === undefined
                ? "snapshot unavailable"
                : `${operations.data.runtimeAgents.length} observed agents`}
          </small>
        </Link>
        <Link className="action-metric-card" to="/catalog/workflows">
          <span>Published versions</span>
          <strong>{workflows.data?.length ?? "—"}</strong>
          <small>complete Workflow catalog</small>
        </Link>
      </section>

      <div className="action-center-grid">
        <AttentionPanel query={recent} />
        <ActivePanel queries={activeQueries} />
        <RecentSuccessPanel query={recent} />
        <RuntimeHealthPanel
          authorized={operationsAuthorized}
          query={operations}
        />
        <QuickStartPanel query={workflows} />
      </div>
    </section>
  );
}
