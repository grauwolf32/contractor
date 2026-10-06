import { type UseQueryResult, useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { getRun, type RunPage, type RunSummary } from "../../api/runs";
import { getWorkflow } from "../../api/workflows";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice } from "../../app/error-notice";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { IdChip, StatusGlyph } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import {
  organizeRunOutputs,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
} from "../runs/output-model";
import { workflowFormats } from "../workflows/formats";
import { MaterialKindIcon } from "./material-icon";
import { projectPath } from "./project-sections";
import { ProjectRunIdentity } from "./run-history";
import { runStateLabel } from "./run-state";

/** A Run's exact outputs and its Workflow's published output roles. */
function useRunOutputs(summary: RunSummary) {
  const api = usePublicAPI();
  const identity = parseWorkflowIdentity(summary.workflow);
  const run = useQuery({
    queryKey: queryKeys.runs.detail(summary.runId),
    queryFn: () => getRun(api, summary.runId),
  });
  const contract = useQuery({
    queryKey: [
      ...queryKeys.workflows.detail(
        identity?.name ?? "",
        identity?.version ?? "",
      ),
      "outputs",
    ],
    queryFn: async () => {
      if (!identity) throw new Error("Workflow identity is invalid");
      return requireWorkflowOutputs(
        await getWorkflow(api, identity.name, identity.version),
        identity,
      );
    },
    enabled: identity !== undefined,
  });
  return { run, contract };
}

/**
 * The published primary results of a succeeded Run, as links: nothing while
 * the roles are unknown or when no primary output was published, so no other
 * file is ever offered in its place (US-05).
 */
export function RunPrimaryResult({
  summary,
  projectId,
}: {
  summary: RunSummary;
  projectId: string;
}) {
  const { run, contract } = useRunOutputs(summary);
  if (run.data === undefined || run.data.projectId !== projectId) return null;
  const primary = organizeRunOutputs(run.data.outputs, contract.data).flatMap(
    ({ kind, slot, artifact }) =>
      kind === "primary" && artifact !== undefined ? [{ slot, artifact }] : [],
  );
  if (primary.length === 0) return null;
  return (
    <>
      {" "}
      Primary result:{" "}
      {primary.map(({ slot, artifact }, position) => (
        <span key={slot}>
          {position > 0 ? ", " : null}
          <ContextLink
            returnLabel="Project Overview"
            to={artifactDetailPath(
              { kind: "run", id: summary.runId },
              artifact,
            )}
          >
            {slot}
          </ContextLink>
        </span>
      ))}
      .
    </>
  );
}

/**
 * The outputs of one successful Run: its primary results by the Workflow's
 * published output roles (US-05). A missing primary output is said to be
 * missing; no other file is offered in its place.
 */
function RecentRunResult({
  summary,
  projectId,
}: {
  summary: RunSummary;
  projectId: string;
}) {
  const { run, contract } = useRunOutputs(summary);
  const entries = organizeRunOutputs(run.data?.outputs ?? {}, contract.data);
  const primary = entries.filter((entry) => entry.kind === "primary");
  const shown = (primary.length ? primary : entries).slice(0, 2);
  return (
    <li className="projects-result">
      <div className="projects-result-head">
        <ContextLink
          returnLabel="Project Overview"
          to={`/runs/${encodeURIComponent(summary.runId)}`}
          className="projects-result-workflow"
        >
          {summary.workflow}
        </ContextLink>
        <span className="projects-row-meta">
          <IdChip value={summary.runId} label="Run ID" />
          <RecordedTime value={summary.finishedAt ?? summary.updatedAt} />
        </span>
      </div>
      {run.isPending ? (
        <p className="loading-copy" role="status">
          Loading outputs…
        </p>
      ) : run.error ? (
        <ErrorNotice
          error={run.error}
          onRetry={() => void run.refetch()}
          retryLabel="Retry outputs"
          retryPending={run.isFetching}
        />
      ) : run.data.projectId !== projectId ? (
        <p className="form-error">Run does not belong to this project.</p>
      ) : (
        <>
          {contract.error ? (
            <p className="projects-caption">
              Output roles unavailable.{" "}
              <button
                type="button"
                className="ui-btn"
                data-size="xs"
                onClick={() => void contract.refetch()}
              >
                Retry output roles
              </button>
            </p>
          ) : null}
          {shown.length === 0 ? (
            <p className="projects-caption">No output published by this Run.</p>
          ) : (
            <ul role="list" className="projects-outputs">
              {shown.map((entry) => (
                <li key={entry.slot}>
                  {entry.artifact ? (
                    <ContextLink
                      className="projects-output"
                      returnLabel="Project Overview"
                      to={artifactDetailPath(
                        { kind: "run", id: summary.runId },
                        entry.artifact,
                      )}
                    >
                      <span className="projects-output-icon">
                        <MaterialKindIcon kind="results" />
                      </span>
                      <span className="projects-output-text">
                        <strong>{entry.slot}</strong>
                        <small>
                          {entry.kind === "primary"
                            ? "Primary result"
                            : entry.kind === "declared"
                              ? "Supporting result"
                              : "Run output"}
                          {entry.declaration
                            ? ` · ${entry.declaration.mediaTypes.map((type) => workflowFormats[type] ?? type).join(" / ")}`
                            : ""}
                        </small>
                      </span>
                    </ContextLink>
                  ) : (
                    <p className="projects-caption">
                      {entry.kind === "primary" ? "Primary result" : "Output"}{" "}
                      <code>{entry.slot}</code> was not published.
                    </p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </li>
  );
}

/** Outputs of the three most recent successful Runs. */
export function RecentResults({
  projectId,
  results,
}: {
  projectId: string;
  results: UseQueryResult<RunPage>;
}) {
  const heading = useId();
  return (
    <section className="projects-panel" aria-labelledby={heading}>
      <div className="projects-section-heading">
        <h3 id={heading}>Recent results</h3>
        <Link
          to={`${projectPath(projectId, "runs")}?view=completed&state=succeeded`}
        >
          Successful Runs
        </Link>
      </div>
      <p className="projects-caption">
        Outputs of the three most recent successful Runs.
      </p>
      <QueryView
        query={results}
        loading={
          <p className="loading-copy" role="status">
            Loading successful Runs…
          </p>
        }
        onRetry={() => void results.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <p className="projects-caption">
            No successful Runs yet. Published results appear here.
          </p>
        }
      >
        {(data) => (
          <ul role="list" className="projects-results">
            {data.items.map((run) => (
              <RecentRunResult
                key={run.runId}
                summary={run}
                projectId={projectId}
              />
            ))}
          </ul>
        )}
      </QueryView>
    </section>
  );
}

/** The five most recent Runs of the project, in any state. */
export function RecentRuns({
  projectId,
  runs,
}: {
  projectId: string;
  runs: UseQueryResult<RunPage>;
}) {
  const heading = useId();
  return (
    <section className="projects-panel" aria-labelledby={heading}>
      <div className="projects-section-heading">
        <h3 id={heading}>Recent Runs</h3>
        <Link to={projectPath(projectId, "runs")}>All Runs</Link>
      </div>
      <QueryView
        query={runs}
        loading={
          <p className="loading-copy" role="status">
            Loading recent Runs…
          </p>
        }
        onRetry={() => void runs.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <p className="projects-caption">
            No Runs yet. Checks and Workflows start Runs.
          </p>
        }
      >
        {(data) => (
          <ul role="list" className="projects-rows">
            {data.items.map((run) => {
              const state = runStateLabel(run.state);
              return (
                <li className="projects-row" key={run.runId}>
                  <StatusGlyph tone={state.tone} />
                  <div className="projects-row-main">
                    <ProjectRunIdentity
                      run={run}
                      returnLabel="Project Overview"
                    />
                    <span className="projects-row-meta">
                      <span>{state.label}</span>
                      <RecordedTime value={run.updatedAt} />
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </QueryView>
    </section>
  );
}
